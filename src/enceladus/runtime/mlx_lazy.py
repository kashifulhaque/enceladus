"""Lazy launches on MLX arrays through `mx.fast.metal_kernel`.

By default, a launch on MLX arrays evaluates them, runs on Enceladus's queue, and waits
for the GPU, which costs about 100 µs. With `enceladus.lazy_mlx(True)`, an eligible
launch instead adds the kernel to MLX's lazy graph and returns right away; MLX runs it
when something evaluates the results.

MLX kernels are functional: each output is a new array. A launch is lazy only when every
array the kernel writes is a *fresh output*, an array from `enceladus.new_empty` or
`enceladus.new_zeros` that no launch has written yet. The kernel then writes a new array
that starts zero-filled, like the fresh output, and the argument object takes the new
array's value, as `out[...] = result` would. Any other launch on MLX arrays keeps the
synchronized path, which writes in place.

The adapter reuses the generated MSL: the prelude becomes the `header` of
`mx.fast.metal_kernel`, wrapped in a namespace, and the kernel's body becomes its
`source`, after a prologue that binds the kernel's thread-position parameters.
"""

from __future__ import annotations

import hashlib
import re
import weakref
from dataclasses import dataclass
from typing import Any

import numpy as np

# Kernel parameter attributes that `mx.fast.metal_kernel` provides under the same name.
_MLX_ATTRIBUTES = frozenset({
    "simdgroup_index_in_threadgroup", "thread_index_in_simdgroup", "thread_index_in_threadgroup",
    "thread_position_in_grid", "thread_position_in_threadgroup", "threadgroup_position_in_grid",
    "threadgroups_per_grid", "threads_per_threadgroup", "threads_per_simdgroup",
})  # fmt: skip

# The float-only math functions of `metal::precise`. MLX compiles with fast math
# functions, and Enceladus's other paths with precise ones (see
# `raw.MATH_FP32_FUNCTIONS`); using-declarations select the precise variants for calls in
# the kernel, so results match bit for bit. `abs`, `min`, `max`, `clamp`, and the other
# functions that also take integers aren't listed, because a using-declaration would hide
# their integer overloads.
PRECISE_FUNCTIONS = (
    "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh", "ceil", "copysign", "cos",
    "cosh", "cospi", "divide", "exp", "exp10", "exp2", "fabs", "fdim", "floor", "fma", "fmax",
    "fmax3", "fmedian3", "fmin", "fmin3", "fmod", "fract", "frexp", "ilogb", "ldexp", "log",
    "log10", "log2", "modf", "nextafter", "pow", "powr", "rint", "round", "rsqrt", "sin",
    "sincos", "sinh", "sinpi", "sqrt", "tan", "tanh", "tanpi", "trunc",
)  # fmt: skip

_KERNEL = re.compile(r"^\[\[kernel\]\] void (\w+)\((.*?)\) \{\n", re.S | re.M)
_PARAM = re.compile(r"^\s*(.+?)\s+(\w+)\s*\[\[(\w+)(?:\((\d+)\))?\]\]\s*$")
_OUTSIDE = re.compile(r"^(#include\b|using namespace \w+;)")
_MAX_DIM = (1 << 31) - 1  # MLX takes the grid as C++ ints
# Generated code computes element offsets in 32 bits (`launcher.MAX_ELEMENT_INDEX`).
_MAX_ELEMENTS = 1 << 31

_enabled = False


def lazy_mlx(enabled: bool = True) -> None:
    """Sets whether launches on MLX arrays join MLX's lazy graph when they can.

    A launch is lazy when every array argument is an MLX array, and every argument that
    the kernel writes is a fresh output from `enceladus.new_empty` or
    `enceladus.new_zeros` that no launch has written yet. The kernel then writes a
    zero-filled new array that replaces the argument's value, and the launch returns
    without waiting. Other launches on MLX arrays evaluate, write in place, and wait.

    Lazy launches don't check that a strided view spans fewer than 2^31 elements,
    because MLX reports strides only for evaluated arrays.
    """
    global _enabled
    _enabled = bool(enabled)


def enabled() -> bool:
    """Returns whether `lazy_mlx(True)` is in effect."""
    return _enabled


# ---- Fresh outputs ----

# id(array) -> weak reference, for arrays from `new_output` that no launch has written.
_fresh: dict[int, weakref.ref] = {}


def new_output(shape: tuple[int, ...], dtype: Any) -> Any:
    """Returns a lazy zero-filled MLX array registered as a fresh output."""
    import mlx.core as mx

    a = mx.zeros(shape, dtype=dtype)
    key = id(a)
    _fresh[key] = weakref.ref(a, lambda _r, key=key: _fresh.pop(key, None))
    return a


def is_fresh(a: Any) -> bool:
    """Returns whether `a` is a fresh output that no launch has written."""
    r = _fresh.get(id(a))
    return r is not None and r() is a


def consume(a: Any) -> None:
    """Marks `a` as written, so it's no longer a fresh output."""
    if is_fresh(a):
        del _fresh[id(a)]


# ---- Adapting generated MSL ----


@dataclass
class MlxKernel:
    """A compiled kernel adapted to `mx.fast.metal_kernel`.

    Attributes:
        kernel: The callable from `mx.fast.metal_kernel`.
        inputs: The argument indices of the kernel's inputs, in MLX order.
        outputs: The argument indices of the arrays the kernel writes, in MLX order.
        scalars: For each argument index that's a scalar, a function that converts a
            launch value to an MLX scalar input.
        group: Threads per threadgroup.
        verified: Whether MLX compiled and ran the kernel once.
    """

    kernel: Any
    inputs: list[int]
    outputs: list[int]
    scalars: dict[int, Any]
    group: int
    verified: bool = False


def _scalar_converter(dtype: str) -> Any:
    import mlx.core as mx

    if dtype == "i1":
        return bool
    if dtype == "i32":
        return int
    if dtype == "f32":
        return float
    if dtype in ("i64", "u64"):
        t = np.int64 if dtype == "i64" else np.uint64
        return lambda v: mx.array(t(v))
    raise TypeError(f"a {dtype} scalar can't be an MLX kernel input")


def adapt(ck: Any) -> MlxKernel | str:
    """Adapts compiled kernel `ck` to `mx.fast.metal_kernel`.

    Returns:
        The adapted kernel, or the reason that `ck` can't run through MLX.
    """
    import mlx.core as mx

    if ck.enable_logging or ck.assert_buffer is not None:
        return "it prints or asserts, which needs Enceladus's own queue"
    m = _KERNEL.search(ck.msl)
    body = ck.msl[m.end():].rstrip() if m else ""
    if not m or m.group(1) != ck.name or not body.endswith("}"):
        return "its MSL doesn't have the structure that Enceladus generates"
    prologue, buffers = [], []
    for p in m.group(2).split(","):
        pm = _PARAM.match(p)
        if pm is None:
            return f"its MSL has a parameter that the adapter can't parse: {p.strip()}"
        ty, name, attr, _ = pm.groups()
        if attr == "buffer":
            buffers.append(name)
        elif attr in _MLX_ATTRIBUTES:
            prologue.append(f"  const {ty} {name} = {attr};\n")
        else:
            return f"it uses [[{attr}]], which MLX kernels don't provide"
    if buffers != [a.name for a in ck.args]:
        return "its MSL parameters don't match its arguments"
    if not any(a.written for a in ck.args):
        return "it writes no arrays, and an MLX kernel needs an output"
    try:
        scalars = {i: _scalar_converter(a.dtype)
                   for i, a in enumerate(ck.args) if not a.is_pointer}  # fmt: skip
    except TypeError as e:
        return str(e)
    # Includes and namespace directives stay at file scope; the prelude goes in a
    # namespace whose using-declarations select the precise math functions.
    outside, prelude = [], []
    for line in ck.msl[: m.start()].splitlines(keepends=True):
        (outside if _OUTSIDE.match(line) else prelude).append(line)
    using = "".join(f"  using metal::precise::{f};\n" for f in PRECISE_FUNCTIONS)
    header = (f"{''.join(outside)}namespace enc_lazy {{\n{using}{''.join(prelude)}\n"
              "}  // namespace enc_lazy\n")  # fmt: skip
    source = f"  using namespace enc_lazy;\n{using}{''.join(prologue)}{body[:-1]}"
    outputs = [i for i, a in enumerate(ck.args) if a.written]
    inputs = [i for i, a in enumerate(ck.args) if not a.written]
    # MLX caches compiled kernels by name, so the name must identify the source.
    digest = hashlib.sha256(ck.msl.encode()).hexdigest()[:16]
    kernel = mx.fast.metal_kernel(
        name=f"enceladus_{ck.name}_{digest}",
        input_names=[ck.args[i].name for i in inputs],
        output_names=[ck.args[i].name for i in outputs],
        source=source, header=header, ensure_row_contiguous=False,
        compile_options={"math_mode": ck.math_mode},
    )  # fmt: skip
    return MlxKernel(kernel, inputs, outputs, scalars, ck.num_warps * 32)


def ineligible_reason(ck: Any, grid: tuple[int, int, int], values: Any) -> str | None:
    """Returns why a launch on MLX arrays can't be lazy, or `None` if it can.

    The caller guarantees that every array argument is an MLX array.
    """
    if grid[0] * ck.num_warps * 32 > _MAX_DIM or grid[1] > _MAX_DIM or grid[2] > _MAX_DIM:
        return "its grid is too large for mx.fast.metal_kernel"
    written = [values[i] for i, a in enumerate(ck.args) if a.written]
    for i, a in enumerate(ck.args):
        if not a.is_pointer:
            continue
        v = values[i]
        if v.size == 0:
            return f"argument `{a.name}` is empty"
        if v.size > _MAX_ELEMENTS:
            return f"argument `{a.name}` has 2^31 elements or more"
        if a.written and not is_fresh(v):
            return (f"it writes argument `{a.name}`, which isn't a fresh output from "
                    "enceladus.new_empty or enceladus.new_zeros")  # fmt: skip
        if a.written and sum(w is v for w in written) > 1:
            return f"argument `{a.name}` is passed more than once"
        if not a.written and any(w is v for w in written):
            return f"argument `{a.name}` is also an output"
    return None


def launch(mk: MlxKernel, grid: tuple[int, int, int], values: Any) -> str | None:
    """Adds one launch to MLX's graph and rebinds the written arguments to the results.

    The first launch of each kernel evaluates its results, so that an MSL compile error
    shows up here instead of at a later `mx.eval`.

    Returns:
        `None`, or the reason that MLX couldn't compile or run the kernel, in which case
        nothing changed.
    """
    inputs = [mk.scalars[i](values[i]) if i in mk.scalars else values[i] for i in mk.inputs]
    try:
        outs = mk.kernel(
            inputs=inputs,
            output_shapes=[values[i].shape for i in mk.outputs],
            output_dtypes=[values[i].dtype for i in mk.outputs],
            grid=(grid[0] * mk.group, grid[1], grid[2]),
            threadgroup=(mk.group, 1, 1),
            init_value=0,
        )  # fmt: skip
        if not mk.verified:
            import mlx.core as mx

            mx.eval(*outs)
    except (RuntimeError, ValueError, TypeError) as e:
        if mk.verified:
            raise
        first = str(e).strip().splitlines()
        return f"MLX couldn't run it: {first[0] if first else type(e).__name__}"
    mk.verified = True
    for i, out in zip(mk.outputs, outs, strict=True):
        values[i][...] = out  # shares `out`'s buffer; no copy
        consume(values[i])
    return None
