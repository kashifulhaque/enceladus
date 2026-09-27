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
import enum
import functools
import hashlib
import inspect
import os
import struct
import sys
import sysconfig
import textwrap
import types
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from enceladus.compiler import ir
from enceladus.compiler.errors import CompilationError
from enceladus.compiler.frontend import SourceInfo, build_ir, is_jit_function, parse_function
from enceladus.compiler.pipeline import compile_module
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
    fd = interop.framework_dtype(v)  # PyTorch and MLX dtypes aren't NumPy dtypes
    return core.dtype_from_numpy(fd if fd is not None else np.dtype(d))


def arg_type(value: Any) -> ir.Type:
    """Returns the IR type of a runtime kernel argument.

    An array-like becomes a pointer to its element type. A Python `int` becomes `i32` if it
    fits in signed 32 bits, `i64` if it fits in signed 64 bits, and `u64` if it's at least
    2**63 and fits in unsigned 64 bits. A `np.uint64` becomes `u64`. A `bool` becomes `i1`,
    and a `float` becomes `f32`. An `ir.Type` passes through, which lets callers build IR
    from types.

    Raises:
        TypeError: The value can't be a runtime kernel argument.
    """
    if isinstance(value, ir.Type):
        return value
    if isinstance(value, (bool, np.bool_)):
        return ir.i1
    if isinstance(value, np.uint64):
        return ir.u64
    if isinstance(value, (int, np.integer)):
        v = int(value)
        if -(1 << 31) <= v < (1 << 31):
            return ir.i32
        if -(1 << 63) <= v < (1 << 63):
            return ir.i64
        if 0 <= v < (1 << 64):
            return ir.u64
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
        # (getter, name, value) for each binding that `cache_key` covers, where
        # `getter(name, default)` reads the binding's current value.
        self._deps: tuple[tuple[Callable[[str, Any], Any], str, Any], ...] = ()
        # (value, token) for each mutable container that `cache_key` covers by content.
        self._volatile: tuple[tuple[Any, str], ...] = ()
        # The @enceladus.jit functions that this one reaches, in a deterministic order.
        self._jit_deps: tuple[JITFunction, ...] = ()
        self._ir_cache: dict[str, ir.Module] = {}
        # Specialization keys that interpreted launches checked with the compiler: the ones
        # that compile, and the ones that don't, or whose source can't be read.
        self._verified: set[str] = set()
        self._unverified: set[str] = set()
        self._verify_warned = False
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
        """A SHA-256 over the source and every value that the frontend can read.

        The hash covers referenced @enceladus.jit functions, global constants and tuples,
        module attributes such as `consts.SCALE`, and the source and globals of plain
        Python helpers that the kernel calls at compile time. Changing any of them
        changes the key.
        """
        if self._cache_key is None:
            finder = _DependencyFinder()
            key = finder.jit_hash(self)
            self._deps = tuple(finder.deps)
            self._volatile = tuple(finder.volatile)
            self._jit_deps = tuple(finder.jits)
            self._cache_key = key
        return self._cache_key

    def _check_globals(self) -> None:
        """Drops compiled kernels if a value that they depend on changed.

        Compiled code bakes in global constants and helper functions, so a kernel
        recompiles after, for example, `SCALE = 5` replaces `SCALE = 3`, as the
        interpreter would see the new value. A rebinding that leaves the dependency
        hash unchanged keeps the compiled kernels.
        """
        for get, n, v in self._deps:
            if get(n, _MISSING) is not v:
                break
        else:
            for v, tok in self._volatile:
                cur = _plain_token(v)
                if cur is None or cur[0] != tok:
                    break
            else:
                return
        old = self._cache_key
        self._cache_key = None
        if self.cache_key == old:
            return
        self._ir_cache.clear()
        self._verified.clear()
        self._unverified.clear()
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
            # Like `_env_flag`, but through the bound C method: `os.environ.get` costs
            # about 0.3 µs, a tenth of a launch.
            interpret = core.env_lookup(b"ENCELADUS_INTERPRET", b"0") not in (b"", b"0")
        if not interpret:
            self._run_compiled(args, kwargs, grid, num_warps, dot_warps, dot_backend)
            return
        if num_warps not in (1, 2, 4, 8, 16, 32):
            raise ValueError(_num_warps_msg(num_warps))
        bound = self.bind(args, kwargs)
        g = _resolve_grid(grid, bound)
        if _env_flag("ENCELADUS_DUMP"):
            self._interp_ir(bound, num_warps)
        verify = os.environ.get("ENCELADUS_VERIFY", "")
        if verify != "0":
            self._check_compiles(bound, num_warps, strict=verify != "")
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

    def _check_compiles(self, bound: Mapping[str, Any], num_warps: int, strict: bool) -> None:
        """Runs the compiler's checks for an interpreted launch, once per specialization.

        The interpreter runs kernels as Python, so it accepts some kernels that compiled
        mode refuses. By default, a refused kernel gets one warning and still runs.
        `ENCELADUS_VERIFY=1` raises the error instead, and `ENCELADUS_VERIFY=0` skips the
        check. A kernel whose source can't be read, such as one defined in an interactive
        prompt, isn't checked unless the check is strict.

        Raises:
            CompilationError: `strict` is true and compiled mode refuses the kernel.
        """
        spec = self.specialize(bound)
        key = spec.key(num_warps, self.math_mode)
        if self._deps:
            self._check_globals()
        if key in self._verified or (not strict and key in self._unverified):
            return
        try:
            try:
                self.source_info()
            except CompilationError as e:
                if not strict and isinstance(e.__cause__, (OSError, TypeError)):
                    self._unverified.add(key)
                    return
                raise
            module = build_ir(self, spec.arg_types, spec.arg_facts, spec.constexprs,
                              num_warps, self.math_mode)  # fmt: skip
            compile_module(module)
        except CompilationError as e:
            if strict:
                raise
            self._unverified.add(key)
            if not self._verify_warned:
                self._verify_warned = True
                self._warn_refused(e)
            return
        self._verified.add(key)

    def _warn_refused(self, e: CompilationError) -> None:
        import warnings

        loc = e.loc
        if loc is None:
            src = self.source_info()
            file, line = src.file, src.first_line
        else:
            file, line = loc.file, loc.line
        warnings.warn_explicit(
            f"compiled mode refuses the kernel `{self.__name__}`, which the interpreter runs "
            f"anyway: {e.message} To raise this error in the interpreter too, set "
            "ENCELADUS_VERIFY=1. To skip this check, set ENCELADUS_VERIFY=0.",
            UserWarning, file, line,
        )  # fmt: skip

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
    if t is int or t is bool or t is str:
        return (t, v)
    if t is float:
        # The bit pattern keeps -0.0 apart from 0.0 and lets a NaN hit the cache.
        return (t, _pack_double(v))
    if t is tuple or t is list:
        return (t, tuple([_const_key(x) for x in v]))
    if isinstance(v, (float, complex, np.floating, np.complexfloating)):
        return (t, np.asarray(v).tobytes())
    return (t, v)


def _spec_key(v: Any, no_facts: bool) -> Any:
    """Returns the part of the specialization key contributed by one runtime argument."""
    t = type(v)
    if t is bool:
        return "i1"
    if t is int:
        if -(1 << 31) <= v < (1 << 31):
            return "i32" if no_facts else ("i32", v % 16 == 0, v == 1)
        if -(1 << 63) <= v < (1 << 63):
            return "i64" if no_facts else ("i64", v % 16 == 0, v == 1)
        if 0 <= v < (1 << 64):
            return "u64" if no_facts else ("u64", v % 16 == 0, False)
        raise TypeError(f"integer argument {v} doesn't fit in 64 bits")
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


# ---------------------------------------------------------------------------
# Dependency hashing
# ---------------------------------------------------------------------------

_MISSING = object()
_pack_double = struct.Struct("<d").pack
# Mixed into the token of a value that has no deterministic content hash, so that another
# process never reuses a disk cache entry built from it.
_PROCESS_NONCE = os.urandom(8).hex()
_MAX_DEPTH = 32


def _type_name(t: type) -> str:
    return f"{t.__module__}.{t.__qualname__}"


def _plain_token(v: Any, depth: int = 0) -> tuple[str, bool] | None:
    """Returns a deterministic token for a constant data value, and whether the value
    contains a mutable container, or `None` for anything else.

    Floats are keyed by their bit pattern, so `-0.0` differs from `0.0` and a NaN equals
    itself.
    """
    if depth > _MAX_DEPTH:
        return None
    v = core.unwrap(v)
    t = type(v)
    if v is None or v is Ellipsis:
        return repr(v), False
    if isinstance(v, enum.Enum):
        inner = _plain_token(v.value, depth + 1)
        return None if inner is None else (f"{_type_name(t)}.{v.name}={inner[0]}", inner[1])
    if isinstance(v, (bool, int, str, bytes)):
        return f"{_type_name(t)}:{v!r}", False
    if isinstance(v, float):
        return f"{_type_name(t)}:{_pack_double(v).hex()}", False
    if isinstance(v, complex):
        return f"{_type_name(t)}:{_pack_double(v.real).hex()}{_pack_double(v.imag).hex()}", False
    if isinstance(v, np.generic):
        return f"{_type_name(t)}:{v.tobytes().hex()}", False
    if isinstance(v, core.dtype):
        return f"dtype:{v!r}", False
    if isinstance(v, slice):
        inner = _plain_token((v.start, v.stop, v.step), depth + 1)
        return None if inner is None else (f"slice{inner[0]}", False)
    if isinstance(v, (tuple, list, set, frozenset, dict)):
        mutable = isinstance(v, (list, set, dict))
        items = [item for kv in v.items() for item in kv] if isinstance(v, dict) else list(v)
        parts = []
        for x in items:
            r = _plain_token(x, depth + 1)
            if r is None:
                return None
            parts.append(r[0])
            mutable |= r[1]
        if isinstance(v, dict):
            parts = sorted(f"{k}:{x}" for k, x in zip(parts[::2], parts[1::2], strict=True))
        elif isinstance(v, (set, frozenset)):
            parts.sort()
        return f"{_type_name(t)}({','.join(parts)})", mutable
    return None


def is_tracked(v: Any) -> bool:
    """Returns whether `_DependencyFinder.token` gives `v` a deterministic token.

    The frontend refuses globals that fail this check, because the kernel cache can't
    detect a change to them. It mirrors `token` without hashing anything.
    """
    v = core.unwrap(v)
    if _plain_token(v) is not None or is_jit_function(v):
        return True
    if isinstance(v, (core.Builtin, types.ModuleType, types.FunctionType)):
        return True
    if isinstance(v, tuple):
        return all(is_tracked(x) for x in v)
    if isinstance(v, functools.partial):
        return (is_tracked(v.func) and is_tracked(v.args)
                and all(is_tracked(x) for x in v.keywords.values()))  # fmt: skip
    if isinstance(v, (types.BuiltinFunctionType, type)):
        if _library_token(v) is not None:
            return True
        if isinstance(v, type):
            try:
                inspect.getsource(v)
            except (OSError, TypeError):
                pass
            else:
                return True
    return isinstance(getattr(v, "__wrapped__", None), types.FunctionType)


def is_content_hashed(v: Any) -> bool:
    """Returns whether the dependency hash covers `v` and every attribute of `v`.

    The finder follows attribute chains through modules and classes. On any other object,
    it hashes the object itself, which covers the attributes only of data values such as
    numbers, strings, dtypes, enums, and tuples, including named tuples.
    """
    v = core.unwrap(v)
    if isinstance(v, tuple):
        return all(is_tracked(x) for x in v)
    return _plain_token(v) is not None or isinstance(v, core.dtype)


@functools.cache
def _library_roots() -> tuple[str, ...]:
    """Returns the directories of installed code: the standard library, site-packages, and
    Enceladus itself, whose source `cache.compiler_hash` covers."""
    paths = sysconfig.get_paths()
    roots = {paths[k] for k in ("stdlib", "platstdlib", "purelib", "platlib") if k in paths}
    roots.add(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return tuple(os.path.realpath(r) + os.sep for r in roots)


def _library_token(obj: Any) -> str | None:
    """Returns a name-and-version token if `obj` is defined in installed code, or `None` if
    it's user code whose source the hash must cover."""
    module = getattr(obj, "__module__", None) or ""
    code = getattr(obj, "__code__", None)
    if isinstance(code, types.CodeType):
        file: str | None = code.co_filename
    else:
        file = getattr(sys.modules.get(module), "__file__", None)
    # A file of None means a builtin or C extension type or function.
    if file is not None and not os.path.realpath(file).startswith(_library_roots()):
        return None
    name = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", "?")
    top = sys.modules.get(module.split(".")[0])
    return f"lib:{module}.{name}:{getattr(top, '__version__', '')}"


def _code_token(code: types.CodeType) -> str:
    """Returns a deterministic token for a code object, for functions without source."""
    parts = [code.co_code.hex(), repr(code.co_names), repr(code.co_varnames)]
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            parts.append(_code_token(c))
        else:
            r = _plain_token(c)
            parts.append(r[0] if r is not None else type(c).__name__)
    return "|".join(parts)


def _code_names(code: types.CodeType) -> set[str]:
    names = set(code.co_names)
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            names |= _code_names(c)
    return names


def _chains(nodes: Iterable[ast.AST], local: Iterable[str] = ()) -> list[tuple[str, ...]]:
    """Returns every name, and every attribute chain on a name, that `nodes` read, such as
    `("m",)` and `("m", "SCALE")` for `m.SCALE`, except chains on the names in `local`."""
    out: set[tuple[str, ...]] = set()
    for root in nodes:
        for n in ast.walk(root):
            if isinstance(n, ast.Name):
                out.add((n.id,))
            elif isinstance(n, ast.Attribute):
                attrs = []
                cur: ast.AST = n
                while isinstance(cur, ast.Attribute):
                    attrs.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    out.add((cur.id, *reversed(attrs)))
    local = set(local)
    return sorted(c for c in out if c[0] not in local)


def _is_enceladus_module(obj: Any) -> bool:
    name = obj.__name__ if isinstance(obj, types.ModuleType) else ""
    return name == "enceladus" or name.startswith("enceladus.")


def _cell_getter(cell: types.CellType) -> Callable[[str, Any], Any]:
    def get(_name: str, default: Any) -> Any:
        try:
            return cell.cell_contents
        except ValueError:  # the cell is empty
            return default

    return get


class _DependencyFinder:
    """Hashes a kernel's source and every value that the frontend can read from it.

    Triton's `DependenciesFinder` hashes referenced globals. Enceladus kernels can also read
    module attributes, tuples, and plain Python functions that run at compile time, so the
    finder resolves attribute chains on modules and classes, hashes tuples by content,
    and hashes the source, defaults, closure, and globals of plain Python helpers,
    transitively. `deps` collects each binding that the hash read, so that
    `JITFunction._check_globals` can notice a rebinding without rehashing.
    """

    def __init__(self) -> None:
        self.seen: set[int] = set()
        self.deps: list[tuple[Callable[[str, Any], Any], str, Any]] = []
        self.volatile: list[tuple[Any, str]] = []
        self.jits: list[JITFunction] = []
        self._dep_keys: set[tuple[int, str]] = set()

    def jit_hash(self, fn: JITFunction) -> str:
        self.seen.add(id(fn))
        h = hashlib.sha256(inspect.getsource(fn.fn).encode())
        # Only the body: decorators such as @enceladus.autotune(CONFIGS, ...) don't affect
        # the generated code.
        # Parameters shadow globals of the same name.
        chains = _chains(fn.source_info().tree.body, fn.arg_names)
        self._hash_refs(h, fn.fn.__globals__, chains)
        return h.hexdigest()

    def _record(self, get: Callable[[str, Any], Any], owner: Any, name: str, value: Any) -> None:
        k = (id(owner), name)
        if k not in self._dep_keys:
            self._dep_keys.add(k)
            self.deps.append((get, name, value))

    def _hash_refs(self, h: Any, g: dict[str, Any], chains: Iterable[tuple[str, ...]]) -> None:
        lines = set()
        for chain in chains:
            if chain[0] not in g:
                continue  # a local, a parameter, or a builtin
            obj = g[chain[0]]
            if not _is_enceladus_module(obj):
                # Skipping `tl` and `enceladus` keeps the launch path free of checks for
                # typical kernels, which depend on nothing else.
                self._record(g.get, g, chain[0], obj)
            path = chain[0]
            for attr in chain[1:]:
                owner = core.unwrap(obj)
                if not isinstance(owner, (types.ModuleType, type)):
                    break  # an attribute of a value, which the value's token covers
                try:
                    obj = getattr(owner, attr)
                except Exception:  # noqa: BLE001 - the frontend reports a missing attribute
                    obj = _MISSING
                    break
                if not _is_enceladus_module(owner):
                    d = vars(owner)
                    get = d.get if attr in d else functools.partial(getattr, owner)
                    self._record(get, owner, attr, obj)
                path += "." + attr
            lines.add(f"{path}={'<missing>' if obj is _MISSING else self.token(obj)}")
        h.update("\n".join(sorted(lines)).encode())

    def token(self, v: Any) -> str:
        """Returns a deterministic token for a value that a kernel reads."""
        v = core.unwrap(v)
        plain = _plain_token(v)
        if plain is not None:
            if plain[1]:
                self.volatile.append((v, plain[0]))
            return plain[0]
        if isinstance(v, core.Builtin):
            return f"builtin:{v.name}"
        if is_jit_function(v):
            if id(v) in self.seen:
                return f"jit:{v.__qualname__}:seen"
            self.jits.append(v)
            return f"jit:{self.jit_hash(v)}"
        if isinstance(v, tuple):
            return f"{_type_name(type(v))}({','.join(self.token(x) for x in v)})"
        if isinstance(v, types.ModuleType):
            return f"module:{v.__name__}"
        if isinstance(v, types.FunctionType):
            return f"fn:{self._function_token(v)}"
        if isinstance(v, functools.partial):
            kw = ",".join(f"{k}={self.token(x)}" for k, x in sorted(v.keywords.items()))
            return f"partial:{self.token(v.func)}:{self.token(v.args)}:{kw}"
        if isinstance(v, (types.BuiltinFunctionType, type)):
            lib = _library_token(v)
            if lib is not None:
                return lib
            if isinstance(v, type):
                try:
                    src = inspect.getsource(v)
                except (OSError, TypeError):
                    pass
                else:
                    return f"type:{_type_name(v)}:{hashlib.sha256(src.encode()).hexdigest()}"
        wrapped = getattr(v, "__wrapped__", None)
        if isinstance(wrapped, types.FunctionType):  # for example, a functools.cache wrapper
            return f"wrapped:{_type_name(type(v))}:{self.token(wrapped)}"
        return f"opaque:{_type_name(type(v))}:{id(v)}:{_PROCESS_NONCE}"

    def _function_token(self, f: types.FunctionType) -> str:
        name = f"{f.__module__}.{f.__qualname__}"
        if id(f) in self.seen:
            return f"{name}:seen"
        self.seen.add(id(f))
        lib = _library_token(f)
        if lib is not None:
            return lib
        h = hashlib.sha256(name.encode())
        tree = None
        try:
            src = textwrap.dedent(inspect.getsource(f))
        except (OSError, TypeError):
            h.update(_code_token(f.__code__).encode())
        else:
            h.update(src.encode())
            try:
                tree = ast.parse(src)
            except SyntaxError:  # for example, a lambda in the middle of an expression
                h.update(_code_token(f.__code__).encode())
        if tree is not None:
            body: list[ast.stmt] = tree.body
            if len(body) == 1 and isinstance(body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = body[0].body
            chains = _chains(body, f.__code__.co_varnames + f.__code__.co_cellvars)
        else:
            names = sorted(_code_names(f.__code__))
            chains = [(n,) for n in names] + [(n, a) for n in names for a in names]
        h.update(f"defaults={self.token(f.__defaults__ or ())}".encode())
        for k, x in sorted((f.__kwdefaults__ or {}).items()):
            h.update(f"kwdefault {k}={self.token(x)}".encode())
        for var, cell in zip(f.__code__.co_freevars, f.__closure__ or (), strict=True):
            get = _cell_getter(cell)
            val = get(var, _MISSING)
            self._record(get, cell, var, val)
            h.update(f"closure {var}={'<empty>' if val is _MISSING else self.token(val)}".encode())
        self._hash_refs(h, f.__globals__, chains)
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
        math_mode: `"relaxed"` (the default) or `"fast"`. Relaxed mode keeps infinities
            and NaNs, and comparisons with a NaN follow IEEE 754: they're false, except
            `!=`. Fast mode lets the Metal compiler assume that no value is infinite or
            NaN. In both modes, `tl.tanh` uses an accurate implementation, and `tl.sin`
            and `tl.cos` use Metal's precise variants.
    """

    def deco(f: Callable[..., Any]) -> JITFunction:
        return JITFunction(f, interpret=interpret, do_not_specialize=do_not_specialize,
                           math_mode=math_mode)  # fmt: skip

    return deco(fn) if fn is not None else deco
