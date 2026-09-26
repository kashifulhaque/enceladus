"""`@enceladus.jit`: kernel functions, argument specialization, and launch.

A `JITFunction` wraps a Python kernel. `kernel[grid](*args, **kwargs)` launches it on the
GPU, or in the NumPy interpreter with `ENCELADUS_INTERPRET=1` or `@enceladus.jit(interpret=True)`.

The IR for one specialization comes from `build_ir(fn, arg_types, arg_facts, constexprs,
num_warps, math_mode)`, which is re-exported here from `enceladus.compiler.frontend`. It does
no NumPy or device work. `JITFunction.specialize(bound_args)` computes its inputs from
launch arguments.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from enceladus.compiler import ir
from enceladus.compiler.frontend import SourceInfo, build_ir, is_jit_function, parse_function
from enceladus.interpreter import interp
from enceladus.language import core
from enceladus.runtime import interop

__all__ = [
    "JITFunction",
    "KernelParam",
    "arg_type",
    "arg_facts",
    "build_ir",
    "cdiv",
    "jit",
    "next_power_of_2",
]

COMPILED_AVAILABLE = True
"""Whether compiled (GPU) execution is implemented."""
RECOMPILE_WARNING = 16

MATH_MODES = ("relaxed", "fast")

# The raw ENCELADUS_DEBUG value is part of the in-memory specialization key, because the
# flag decides whether tl.device_assert compiles. Reading it through a bound C method
# avoids a Python call per launch.
_debug_key = core.env_lookup


def cdiv(x: int, div: int) -> int:
    """Returns `ceil(x / div)` for positive integers: the number of blocks that cover `x`."""
    return -(-x // div)


def next_power_of_2(n: int) -> int:
    """Returns the smallest power of two that's greater than or equal to `n`."""
    return 1 if n <= 1 else 1 << (int(n) - 1).bit_length()


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0") not in ("", "0")


# ---------------------------------------------------------------------------
# Argument typing and specialization
# ---------------------------------------------------------------------------


def _is_array_like(v: Any) -> bool:
    if isinstance(v, (np.generic, str, bytes)):
        return False
    return interp.is_array_like(v) and hasattr(v, "dtype")


def _elem_dtype(v: Any) -> core.dtype:
    d = v.dtype
    if isinstance(d, core.dtype):
        return d
    fd = interop.element_dtype(v)  # PyTorch and MLX dtypes aren't NumPy dtypes
    return core.dtype_from_numpy(fd if fd is not None else np.dtype(d))


def arg_type(value: Any) -> ir.Type:
    """Returns the IR type of a runtime kernel argument.

    An array-like becomes a pointer to its element type. A Python `int` becomes `i32` if it
    fits in signed 32 bits and `i64` otherwise. A `bool` becomes `i1`, and a `float`
    becomes `f32`. An `ir.Type` passes through, which lets callers build IR from types.

    Raises:
        TypeError: The value can't be a runtime kernel argument.
    """
    if isinstance(value, ir.Type):
        return value
    if isinstance(value, (bool, np.bool_)):
        return ir.i1
    if isinstance(value, (int, np.integer)):
        v = int(value)
        if -(1 << 31) <= v < (1 << 31):
            return ir.i32
        if -(1 << 63) <= v < (1 << 63):
            return ir.i64
        raise TypeError(f"integer argument {v} doesn't fit in 64 bits")
    if isinstance(value, (float, np.floating)):
        return ir.f32
    if _is_array_like(value):
        return ir.PointerType(ir.scalar(_elem_dtype(value)))
    raise TypeError(
        f"a {type(value).__name__} can't be a runtime kernel argument. If it's a compile-time "
        "value, annotate the parameter as tl.constexpr."
    )


def _data_ptr(value: Any) -> int | None:
    if isinstance(value, np.ndarray):
        return value.__array_interface__["data"][0]
    p = getattr(value, "data_ptr", None)
    if callable(p):
        p = p()
    if isinstance(p, int):
        return p
    ai = getattr(value, "__array_interface__", None)
    if isinstance(ai, dict):
        return ai["data"][0]
    return None


def arg_facts(value: Any) -> dict[str, Any]:
    """Returns the specialization facts of a runtime argument.

    Integers get `divisibility = 16` when `value % 16 == 0` and `equal_to_1` when
    `value == 1`. Arrays get `divisibility = 16` when their data pointer is 16-byte
    aligned. Facts become IR attributes; they never remove an argument.
    """
    if isinstance(value, (bool, np.bool_, float, np.floating)) or isinstance(value, ir.Type):
        return {}
    if isinstance(value, (int, np.integer)):
        v = int(value)
        facts: dict[str, Any] = {}
        if v % 16 == 0:
            facts["divisibility"] = 16
        if v == 1:
            facts["equal_to_1"] = True
        return facts
    if _is_array_like(value):
        aligned = interop.aligned16(value)
        if aligned is None:
            p = _data_ptr(value)
            aligned = p is not None and p % 16 == 0
        return {"divisibility": 16} if aligned else {}
    return {}


# ---------------------------------------------------------------------------
# JITFunction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KernelParam:
    """One parameter of a kernel's signature."""

    name: str
    index: int
    is_constexpr: bool
    default: Any
    do_not_specialize: bool


@dataclass(frozen=True)
class Specialization:
    """The inputs to `build_ir` for one set of launch arguments."""

    arg_types: dict[str, ir.Type]
    arg_facts: dict[str, dict[str, Any]]
    constexprs: dict[str, Any]

    def key(self, num_warps: int, math_mode: str) -> str:
        """Returns a deterministic string key for caches."""
        types = ",".join(f"{k}:{t}" for k, t in self.arg_types.items())
        facts = ",".join(f"{k}:{sorted(f.items())}" for k, f in self.arg_facts.items() if f)
        consts = ",".join(f"{k}={v!r}" for k, v in self.constexprs.items())
        return f"{types}|{facts}|{consts}|w{num_warps}|{math_mode}"


class JITFunction:
    """A Enceladus kernel or device function created by `@enceladus.jit`.

    Launch a kernel with `kernel[grid](*args, **constexprs, num_warps=4)`. `grid` is a
    tuple of 1 to 3 ints, or a callable that takes a dict of all arguments by name
    (constexprs included) and returns such a tuple.
    """

    _is_enceladus_jit = True

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        interpret: bool | None = None,
        do_not_specialize: Iterable[str | int] = (),
        math_mode: str = "relaxed",
    ) -> None:
        if not inspect.isfunction(fn):
            raise TypeError(f"@enceladus.jit needs a function, but got {type(fn).__name__}")
        if math_mode not in MATH_MODES:
            raise ValueError(f"math_mode must be one of {MATH_MODES}, but got {math_mode!r}")
        self.fn = fn
        functools.update_wrapper(self, fn)
        self.interpret = interpret
        self.math_mode = math_mode
        self.signature = inspect.signature(fn)
        names = list(self.signature.parameters)
        dns: set[str] = set()
        for d in do_not_specialize:
            if isinstance(d, int):
                if not 0 <= d < len(names):
                    raise ValueError(f"do_not_specialize index {d} is out of range")
                d = names[d]
            if d not in names:
                raise ValueError(f"do_not_specialize names unknown parameter {d!r}")
            dns.add(d)
        params = []
        for i, (n, p) in enumerate(self.signature.parameters.items()):
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                raise TypeError(
                    f"kernel `{fn.__name__}` can't take *args or **kwargs. List every "
                    "parameter by name."
                )
            params.append(
                KernelParam(n, i, core.is_constexpr_annotation(p.annotation), p.default, n in dns)
            )
        self.params: list[KernelParam] = params
        self.arg_names = names
        self._src: SourceInfo | None = None
        self._cache_key: str | None = None
        # (globals dict, name, value) for each global that `cache_key` covers.
        self._deps: tuple[tuple[dict, str, Any], ...] = ()
        self._ir_cache: dict[str, ir.Module] = {}
        interp.register_kernel_code(fn.__code__)

    def __repr__(self) -> str:
        return f"JITFunction({self.__module__}:{self.__qualname__})"

    def source_info(self) -> SourceInfo:
        """Returns the parsed source of the kernel."""
        if self._src is None:
            self._src = parse_function(self.fn)
        return self._src

    @property
    def cache_key(self) -> str:
        """A SHA-256 over the source and every referenced @enceladus.jit function and global
        constant. Changing a helper function changes the key of its callers."""
        if self._cache_key is None:
            deps: list[tuple[dict, str, Any]] = []
            self._cache_key = _dependency_hash(self, set(), deps)
            self._deps = tuple(deps)
        return self._cache_key

    def _check_globals(self) -> None:
        """Drops compiled kernels if a global that they depend on was reassigned.

        Compiled code bakes in global constants and helper functions, so a kernel
        recompiles after, for example, `SCALE = 5` replaces `SCALE = 3`, as the
        interpreter would see the new value.
        """
        for g, n, v in self._deps:
            cur = core.unwrap(g.get(n))
            if cur is not v and not (type(cur) is type(v) and cur == v):
                break
        else:
            return
        self._cache_key = None
        self._deps = ()
        self._ir_cache.clear()
        if "_binder_fn" in self.__dict__:  # the compiled-launch state exists
            self._compiled.clear()
            self._spec_history.clear()

    # ---- calling and launching ----

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if core.INTERP.depth > 0:
            return self.fn(*args, **kwargs)
        raise RuntimeError(
            f"`{self.__name__}` is a @enceladus.jit function. Launch it with "
            f"`{self.__name__}[grid](...)`, or call it from another @enceladus.jit function."
        )

    def __getitem__(self, grid: Any) -> Callable[..., None]:
        return lambda *args, **kwargs: self.run(*args, grid=grid, **kwargs)

    def bind(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Binds launch arguments to parameter names, applying defaults."""
        try:
            ba = self.signature.bind(*args, **kwargs)
        except TypeError as e:
            raise TypeError(f"{self.__name__}: {e}") from None
        ba.apply_defaults()
        return dict(ba.arguments)

    def specialize(self, bound: Mapping[str, Any]) -> Specialization:
        """Computes argument types, facts, and constexpr values from bound arguments."""
        types: dict[str, ir.Type] = {}
        facts: dict[str, dict[str, Any]] = {}
        consts: dict[str, Any] = {}
        for p in self.params:
            v = bound[p.name]
            if p.is_constexpr:
                consts[p.name] = core.unwrap(v)
                continue
            try:
                types[p.name] = arg_type(v)
            except TypeError as e:
                raise TypeError(f"{self.__name__}: argument `{p.name}`: {e}") from None
            facts[p.name] = {} if p.do_not_specialize else arg_facts(v)
        return Specialization(types, facts, consts)

    def ir(self, *args: Any, num_warps: int = 4, **kwargs: Any) -> ir.Module:
        """Returns the verified IR for these arguments without running the kernel.

        Array arguments can be NumPy arrays, `enceladus.Tensor` objects, or `ir.Type`
        objects such as `ir.PointerType(ir.f32)`.
        """
        spec = self.specialize(self.bind(args, kwargs))
        return build_ir(self, spec.arg_types, spec.arg_facts, spec.constexprs, num_warps,
                        self.math_mode)  # fmt: skip

    def run(self, *args: Any, grid: Any, num_warps: int = 4, num_stages: int | None = None,
            dot_warps: tuple[int, int] | None = None, dot_backend: str = "auto",
            **kwargs: Any) -> None:  # fmt: skip
        """Launches the kernel.

        `dot_warps=(WM, WN)` arranges the SIMD groups of every `tl.dot` as a WM x WN grid.
        `dot_backend` selects the `tl.dot` lowering: "auto" (the default), "simdgroup", or
        "mpp" (Metal 4 `matmul2d`); see `enceladus.runtime.dot_backend`. `num_stages` is
        accepted for Triton compatibility and has no effect.
        """
        interpret = self.interpret
        if interpret is None:
            interpret = _env_flag("ENCELADUS_INTERPRET")
        if not interpret:
            self._run_compiled(args, kwargs, grid, num_warps, dot_warps, dot_backend)
            return
        if num_warps not in (1, 2, 4, 8, 16, 32):
            raise ValueError(_num_warps_msg(num_warps))
        bound = self.bind(args, kwargs)
        g = _resolve_grid(grid, bound)
        if _env_flag("ENCELADUS_DUMP") or ir.verify_enabled():
            self._interp_ir(bound, num_warps)
        if any(interop.framework_of(v) is not None for v in bound.values()):
            # The interpreter works on NumPy views of the frameworks' shared memory.
            bound = {k: self._host_view(k, v) for k, v in bound.items()}
        interp.run_grid(self.fn, g, self._interp_args(bound))

    # ---- compiled execution ----

    def _binder(self) -> Callable[..., tuple[tuple, tuple]]:
        """Builds (once) a function that splits call arguments into runtime and constexpr
        values, applying defaults, faster than `inspect.Signature.bind`."""
        b = self.__dict__.get("_binder_fn")
        if b is None:
            ns: dict[str, Any] = {}
            params = []
            for p in self.signature.parameters.values():
                if p.default is p.empty:
                    params.append(p.name)
                else:
                    ns[f"_d_{p.name}"] = p.default
                    params.append(f"{p.name}=_d_{p.name}")
            rt = [p.name for p in self.params if not p.is_constexpr]
            ce = [p.name for p in self.params if p.is_constexpr]
            src = (f"def binder({', '.join(params)}):\n"
                   f"    return ({''.join(n + ', ' for n in rt)}), "
                   f"({''.join(n + ', ' for n in ce)})\n")  # fmt: skip
            exec(src, ns)  # noqa: S102 - the source is built from parameter names only
            b = self.__dict__["_binder_fn"] = ns["binder"]
            self._rt_names = rt
            self._ce_names = ce
            self._dns = tuple(p.do_not_specialize for p in self.params if not p.is_constexpr)
            self._compiled: dict[tuple, Any] = {}
            self._spec_history: list[tuple] = []
        return b

    def _lookup(self, args: tuple, kwargs: dict, num_warps: int,
                dot_warps: tuple[int, int] | None, dot_backend: str):  # fmt: skip
        """Splits launch arguments into runtime and constexpr values, and returns them with
        the specialization key and the compiled kernel for that key, or `None`."""
        binder = self._binder()
        if self._deps:
            self._check_globals()
        try:
            runtime, consts = binder(*args, **kwargs)
            # List comprehensions: this runs on every launch, and generators cost more.
            key = (tuple([_spec_key(v, d) for v, d in zip(runtime, self._dns, strict=True)]),
                   tuple([_const_key(c) for c in consts]), num_warps, dot_warps,
                   dot_backend, _debug_key(b"ENCELADUS_DEBUG"))  # fmt: skip
            return runtime, consts, key, self._compiled.get(key)
        except TypeError as e:
            raise self._argument_error(args, kwargs, e) from None

    def _argument_error(self, args: tuple, kwargs: dict, e: TypeError) -> TypeError:
        """Returns an error that names the kernel and the argument that the fast launch path
        rejected with `e`."""
        try:
            bound = self.bind(args, kwargs)
            self.specialize(bound)
        except TypeError as named:
            return named
        for p in self.params:
            if not p.is_constexpr:
                continue
            v = core.unwrap(bound[p.name])
            try:
                hash(_const_key(v))
            except TypeError:
                return TypeError(
                    f"{self.__name__}: the tl.constexpr parameter `{p.name}` needs a hashable "
                    "value, such as a number, a string, a dtype, or a tuple, but got a "
                    f"{type(v).__name__}"
                )
        return TypeError(f"{self.__name__}: {e}")

    def _run_compiled(self, args: tuple, kwargs: dict, grid: Any, num_warps: int,
                      dot_warps: tuple[int, int] | None = None,
                      dot_backend: str = "auto") -> None:  # fmt: skip
        runtime, consts, key, ck = self._lookup(args, kwargs, num_warps, dot_warps, dot_backend)
        if ck is None:
            ck = self._compile_for(runtime, consts, num_warps, key, dot_warps,
                                   dot_backend=dot_backend)
        if callable(grid):
            meta = dict(zip(self._rt_names, runtime, strict=True))
            meta.update(zip(self._ce_names, (core.unwrap(c) for c in consts), strict=True))
            grid = grid(meta)
        if type(grid) is not tuple or len(grid) != 3:
            g0 = grid[0] if type(grid) is tuple and len(grid) == 1 else None
            # A 1-tuple of a non-negative int is the common case; skip the general path.
            grid = (g0, 1, 1) if type(g0) is int and g0 >= 0 else _normalize_grid(grid)
        ck.launch(grid, runtime)

    def _compile_for(self, runtime: tuple, consts: tuple, num_warps: int, key: tuple,
                     dot_warps: tuple[int, int] | None = None, record: bool = True,
                     dot_backend: str = "auto"):  # fmt: skip
        if num_warps not in (1, 2, 4, 8, 16, 32):
            raise ValueError(_num_warps_msg(num_warps))
        from enceladus.runtime import dot_backend as backends
        from enceladus.runtime.compile import compile_specialization

        backends.check(dot_backend)

        bound = dict(zip(self._rt_names, runtime, strict=True))
        bound.update(zip(self._ce_names, consts, strict=True))
        spec = self.specialize(bound)
        ck = compile_specialization(self, spec, num_warps, dot_warps, dot_backend)
        self._compiled[key] = ck
        if not record:  # autotuning compiles many configs on purpose
            return ck
        self._spec_history.append(key)
        if len(self._spec_history) == RECOMPILE_WARNING + 1:
            _warn_recompiles(self)
        return ck

    def warmup(self, *args: Any, grid: Any = None, num_warps: int = 4,
               dot_warps: tuple[int, int] | None = None, dot_backend: str = "auto",
               _record: bool = True, **kwargs: Any):  # fmt: skip
        """Compiles the kernel for these arguments without launching it.

        Returns:
            A `CompiledKernel` with `msl`, `ir`, `threadgroup_memory_bytes`, and `num_warps`.
        """
        runtime, consts, key, ck = self._lookup(args, kwargs, num_warps, dot_warps, dot_backend)
        return ck or self._compile_for(runtime, consts, num_warps, key, dot_warps, _record,
                                       dot_backend)  # fmt: skip

    def explain(self, *args: Any, grid: Any = None, num_warps: int = 4,
                dot_warps: tuple[int, int] | None = None, dot_backend: str = "auto",
                **kwargs: Any) -> str:  # fmt: skip
        """Prints and returns what the compiler decided for these arguments.

        The report lists each tile's layout and register estimate, the layout
        conversions and how codegen performs them, threadgroup memory use, and the
        `tl.dot` backend, each with its source line. It compiles the kernel's IR and MSL
        but doesn't launch it or create a Metal pipeline.

        Args:
            *args: The launch arguments, as for `kernel[grid](...)`.
            grid: The launch grid, shown in the report's header.
            num_warps: SIMD groups per program.
            dot_warps: The `(WM, WN)` SIMD-group grid for `tl.dot`.
            dot_backend: The `tl.dot` backend: "auto", "simdgroup", or "mpp".
            **kwargs: Keyword launch arguments, including constexprs.

        Returns:
            The report text.
        """
        from enceladus.compiler.explain import explain_kernel
        from enceladus.runtime import dot_backend as backends
        from enceladus.runtime.compile import build_module
        from enceladus.runtime.device import get_device

        if num_warps not in (1, 2, 4, 8, 16, 32):
            raise ValueError(_num_warps_msg(num_warps))
        bound = self.bind(args, kwargs)
        g = _resolve_grid(grid, bound) if grid is not None else None
        module = build_module(self, self.specialize(bound), num_warps, dot_warps,
                              backend=backends.resolve(dot_backend))  # fmt: skip
        text = explain_kernel(module, get_device().caps.max_threadgroup_memory, g)
        print(text)
        return text

    def _interp_ir(self, bound: Mapping[str, Any], num_warps: int) -> ir.Module:
        """Builds and verifies the IR once per specialization, printing it for ENCELADUS_DUMP."""
        spec = self.specialize(bound)
        key = spec.key(num_warps, self.math_mode)
        if self._deps:
            self._check_globals()
        module = self._ir_cache.get(key)
        if module is None:
            module = build_ir(self, spec.arg_types, spec.arg_facts, spec.constexprs, num_warps,
                              self.math_mode)  # fmt: skip
            self._ir_cache[key] = module
            if _env_flag("ENCELADUS_DUMP"):
                print(f"// Enceladus IR for {self.__name__}\n{module}", flush=True)
        return module

    def _host_view(self, name: str, v: Any) -> Any:
        try:
            return interop.host_view(v)
        except (TypeError, ValueError) as e:
            raise type(e)(f"{self.__name__}: argument `{name}`: {e}") from None

    def _interp_args(self, bound: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for p in self.params:
            v = bound[p.name]
            if p.is_constexpr:
                out[p.name] = core.unwrap(v)
                continue
            try:
                t = arg_type(v)
                if isinstance(t, ir.PointerType):
                    out[p.name] = interp.pointer_from_array(v)
                else:
                    out[p.name] = interp.scalar_arg(v, t.dtype)
            except (TypeError, ValueError) as e:
                raise type(e)(f"{self.__name__}: argument `{p.name}`: {e}") from None
        return out


def _resolve_grid(grid: Any, bound: Mapping[str, Any]) -> tuple[int, int, int]:
    g = grid(dict(bound)) if callable(grid) else grid
    if isinstance(g, (int, np.integer)):
        g = (g,)
    g = tuple(int(x) for x in g)
    if not 1 <= len(g) <= 3:
        raise ValueError(_GRID_RANK_MSG.format(g))
    if any(x < 0 for x in g):
        raise ValueError(f"grid dimensions can't be negative, but got {g}")
    return (*g, *(1,) * (3 - len(g)))  # type: ignore[return-value]


def _normalize_grid(g: Any) -> tuple[int, int, int]:
    if isinstance(g, (int, np.integer)):
        g = (g,)
    g = tuple(int(x) for x in g)
    if not 1 <= len(g) <= 3:
        raise ValueError(_GRID_RANK_MSG.format(g))
    if any(x < 0 for x in g):
        raise ValueError(f"grid dimensions can't be negative, but got {g}")
    return (*g, *(1,) * (3 - len(g)))  # type: ignore[return-value]


def _num_warps_msg(num_warps: Any) -> str:
    return f"num_warps must be a power of two from 1 to 32, but got {num_warps}"


_GRID_RANK_MSG = (
    "the grid needs 1 to 3 dimensions, but got {}. Pass a tuple such as `(n,)` or `(m, n)`."
)


def _const_key(v: Any) -> Any:
    """Returns the part of the specialization key contributed by one constexpr value.

    The key includes the type, because `1`, `1.0`, and `True` compare equal but compile to
    different code.
    """
    v = core.unwrap(v)
    t = type(v)
    if t is tuple or t is list:
        return (t, tuple([_const_key(x) for x in v]))
    return (t, v)


def _spec_key(v: Any, no_facts: bool) -> Any:
    """Returns the part of the specialization key contributed by one runtime argument."""
    t = type(v)
    if t is bool:
        return "i1"
    if t is int:
        if -(1 << 31) <= v < (1 << 31):
            return "i32" if no_facts else ("i32", v % 16 == 0, v == 1)
        return "i64" if no_facts else ("i64", v % 16 == 0, v == 1)
    if t is float:
        return "f32"
    if t.__name__ == "Tensor" and t.__module__ == "torch":
        return interop.torch_spec_key(v, no_facts)
    from enceladus.runtime.tensor import Tensor

    if t is Tensor:
        return (v.np_dtype, True if no_facts else v.data_ptr % 16 == 0)
    if t is np.ndarray:
        return (v.dtype, True if no_facts else v.__array_interface__["data"][0] % 16 == 0)
    if interop.framework_of(v) == interop.KIND_TORCH:  # a torch.Tensor subclass
        return interop.torch_spec_key(v, no_facts)
    ty = arg_type(v)
    facts = {} if no_facts else arg_facts(v)
    return (str(ty), tuple(sorted(facts.items())))


def _warn_recompiles(fn: JITFunction) -> None:
    import warnings

    hist = fn._spec_history
    changes: dict[str, int] = {}
    for prev, cur in zip(hist, hist[1:], strict=False):
        for name, a, b in zip(fn._rt_names, prev[0], cur[0], strict=True):
            if a != b:
                changes[name] = changes.get(name, 0) + 1
        for name, a, b in zip(fn._ce_names, prev[1], cur[1], strict=True):
            if a != b:
                changes[name] = changes.get(name, 0) + 1
    worst = max(changes, key=changes.get) if changes else "?"
    hint = ""
    if worst in fn._rt_names:
        i = fn._rt_names.index(worst)
        if all(isinstance(k[0][i], tuple) and k[0][i][0] in ("i32", "i64") for k in hist):
            hint = " If it's a size that varies, add it to do_not_specialize."
    warnings.warn(
        f"{fn.__name__} has been compiled {len(hist)} times; argument `{worst}` changed its "
        f"specialization most often.{hint}",
        stacklevel=4,
    )


def _dependency_hash(fn: JITFunction, seen: set[int], deps: list) -> str:
    """Hashes `fn`'s source and dependencies, and appends each global that the hash
    covers to `deps` as a `(globals dict, name, value)` triple."""
    seen.add(id(fn))
    h = hashlib.sha256(inspect.getsource(fn.fn).encode())
    g = fn.fn.__globals__
    names = sorted({n.id for n in ast.walk(fn.source_info().tree) if isinstance(n, ast.Name)})
    for n in names:
        v = g.get(n)
        v = core.unwrap(v)
        if is_jit_function(v):
            deps.append((g, n, v))
            if id(v) not in seen:
                h.update(f"{n}:{_dependency_hash(v, seen, deps)}".encode())
        elif isinstance(v, (int, float, bool, str, core.dtype)):
            deps.append((g, n, v))
            h.update(f"{n}={v!r}".encode())
    return h.hexdigest()


def jit(
    fn: Callable[..., Any] | None = None,
    *,
    interpret: bool | None = None,
    do_not_specialize: Iterable[str | int] = (),
    math_mode: str = "relaxed",
) -> Any:
    """Decorates a Enceladus kernel or device function.

    Use it bare (`@enceladus.jit`) or with options:

    Args:
        interpret: Run launches in the NumPy interpreter. When it's `None`, the
            `ENCELADUS_INTERPRET` environment variable decides.
        do_not_specialize: Parameter names or indices whose values don't produce
            specialization facts such as divisibility by 16.
        math_mode: `"relaxed"` (the default, keeps infinities and NaNs) or `"fast"`.
    """

    def deco(f: Callable[..., Any]) -> JITFunction:
        return JITFunction(f, interpret=interpret, do_not_specialize=do_not_specialize,
                           math_mode=math_mode)  # fmt: skip

    return deco(fn) if fn is not None else deco
