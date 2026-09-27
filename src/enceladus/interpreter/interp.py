"""The NumPy interpreter.

The interpreter doesn't interpret the IR. It runs the kernel's Python function directly,
once per program, with `tl` builtins computing on NumPy arrays:

- `ITile` wraps a NumPy array and a `tl` dtype. Arithmetic on `float16` and `bfloat16`
  computes in `float32` and rounds to the tile dtype after each op.
- `IPointer` is a flat view over an array's storage plus a tile of element offsets.
- Runtime scalars (program IDs, scalar kernel arguments, and loop variables) are 0-d
  tiles, so integer division truncates and `int32` wraps, as on the GPU.

The per-builtin NumPy rules live next to each builtin in `enceladus.language.ops`. This
module holds the value types, the shared elementwise semantics, and the grid runner.
`enceladus.interpreter.rewrite` recompiles each kernel with the few source rewrites that
Python semantics need, such as typed loop variables.
"""

from __future__ import annotations

import math
import operator
import sys
import types
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np

from enceladus.compiler import ir, semantic
from enceladus.compiler.errors import CompilationError, Loc
from enceladus.language import core

_KERNEL_CODES: set[types.CodeType] = set()
"""Code objects of every `@enceladus.jit` function, used to find kernel source lines."""


def register_kernel_code(code: types.CodeType) -> None:
    _KERNEL_CODES.add(code)


def _np(dt: core.dtype) -> np.dtype:
    return dt.to_numpy()


def _compute_np(dt: core.dtype) -> np.dtype:
    """Returns the NumPy dtype that arithmetic on `dt` runs in."""
    return np.dtype(np.float32) if dt.is_floating() else _np(dt)


# ---------------------------------------------------------------------------
# Argument conversion
# ---------------------------------------------------------------------------


def _as_numpy(obj: Any) -> np.ndarray:
    """Returns a zero-copy NumPy view of an array-like kernel argument.

    This is the single place to add support for new array types. It accepts NumPy arrays,
    objects with a `.numpy()` method (such as `enceladus.Tensor`), and objects that implement
    `__array__`.
    """
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "numpy"):
        return np.asarray(obj.numpy())
    return np.asarray(obj)


def is_array_like(obj: Any) -> bool:
    return isinstance(obj, np.ndarray) or hasattr(obj, "numpy") or hasattr(obj, "__array__")


def pointer_from_array(obj: Any) -> IPointer:
    """Builds a scalar `IPointer` to the first element of an array-like argument."""
    arr = _as_numpy(obj)
    dt = obj.dtype if isinstance(getattr(obj, "dtype", None), core.dtype) else None
    dt = dt or core.dtype_from_numpy(arr.dtype)
    item = arr.itemsize
    if any(s < 0 for s in arr.strides):
        raise ValueError(
            f"array strides {arr.strides} include a negative stride. Enceladus kernels need "
            "non-negative strides; pass a copy made with `np.ascontiguousarray`."
        )
    if any(s % item for s in arr.strides):
        raise ValueError(
            f"array strides {arr.strides} aren't multiples of the item size {item}. Pass a "
            "copy made with `np.ascontiguousarray`."
        )
    if arr.size == 0:
        length = 0
    else:
        length = 1 + sum((n - 1) * (s // item) for n, s in zip(arr.shape, arr.strides, strict=True))
    flat = np.lib.stride_tricks.as_strided(arr, shape=(length,), strides=(item,))
    return IPointer(flat, np.int64(0), dt)


def scalar_arg(value: Any, dt: core.dtype) -> ITile:
    return ITile(np.asarray(value, dtype=_np(dt)), dt)


# ---------------------------------------------------------------------------
# Kernel source locations
# ---------------------------------------------------------------------------


def _loc(code: types.CodeType, line: int, lasti: int) -> Loc:
    """Returns the location of instruction `lasti`, with its column when Python knows it."""
    col = None
    try:
        pos = list(code.co_positions())[lasti // 2]
        if pos[0] == line and pos[2] is not None:
            col = pos[2] + 1
    except (IndexError, ValueError):
        pass
    loc = Loc(code.co_filename, line, 1)
    if col is None:
        text = loc.source_line() or ""
        col = len(text) - len(text.lstrip()) + 1
    return Loc(code.co_filename, line, col)


def kernel_loc() -> Loc | None:
    """Returns the location of the innermost kernel frame on the call stack."""
    f = sys._getframe(1)
    while f is not None:
        if f.f_code in _KERNEL_CODES:
            return _loc(f.f_code, f.f_lineno, f.f_lasti)
        f = f.f_back
    return None


def _kernel_loc_from_tb(tb: types.TracebackType | None) -> Loc | None:
    best = None
    while tb is not None:
        if tb.tb_frame.f_code in _KERNEL_CODES:
            best = _loc(tb.tb_frame.f_code, tb.tb_lineno, tb.tb_lasti)
        tb = tb.tb_next
    return best


def _where() -> str:
    loc = kernel_loc()
    pid = core.INTERP.program_id
    src = ""
    if loc is not None:
        text = loc.source_line()
        src = f" at {loc.file}:{loc.line}" + (f" (`{text.strip()}`)" if text else "")
    return f"{src}, program_id={pid}"


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


class ITile:
    """An interpreter tile: a NumPy array plus its `tl` dtype."""

    __slots__ = ("data", "dtype")
    __array_priority__ = 1000  # Make NumPy defer to ITile operators.

    def __init__(self, data: Any, dtype: core.dtype):
        self.data = np.asarray(data)
        self.dtype = dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def numel(self) -> int:
        return int(self.data.size)

    @property
    def T(self) -> ITile:  # noqa: N802 - matches Triton
        return ITile(self.data.T, self.dtype)

    def __repr__(self) -> str:
        return f"ITile({self.data!r}, {self.dtype})"

    def __getattr__(self, name: str) -> Any:
        m = core.TILE_METHODS.get(name)
        if m is None:
            raise AttributeError(
                f"tiles have no attribute {name!r}. Check the spelling, or call the tl "
                "function of that name, for example `tl.sum(x)` for `x.sum()`."
            )
        return lambda *a, **k: m.interp(self, *a, **k)

    def __getitem__(self, idx: Any) -> ITile:
        return ITile(self.data[_check_index(idx)], self.dtype)

    # Python constructs that the compiler refuses. Without these methods, Python would
    # fall back to `__getitem__` or raise a TypeError that names the interpreter's classes.
    def __setitem__(self, idx: Any, value: Any) -> None:
        raise CompilationError(
            "tiles are immutable, so you can't assign to an element or attribute. Build a new "
            "tile instead, for example with tl.where."
        )

    def __iter__(self):
        raise CompilationError(
            "tiles can't be iterated or unpacked. Loop with `for i in range(...)`, and select "
            "elements with masks, or use tl.static_range with compile-time tuples."
        )

    def __pow__(self, o):
        raise CompilationError(
            "`**` isn't supported on runtime values. Multiply explicitly, as in `x * x`, or "
            "use tl.exp2 and tl.log2."
        )

    __rpow__ = __pow__

    def __matmul__(self, o):
        raise CompilationError("`@` isn't supported on runtime values. Use tl.dot(a, b).")

    __rmatmul__ = __matmul__

    def _scalar(self) -> Any:
        if self.data.ndim != 0:
            raise CompilationError(
                f"a tile of shape {self.shape} has no single truth value. To choose values "
                "elementwise, use tl.where(cond, x, y)."
            )
        return self.data.item()

    def __bool__(self) -> bool:
        if self.data.ndim != 0:
            # Python calls this for `if`, `not`, `and`, and `or`.
            raise CompilationError(
                f"a tile of shape {self.shape} has no single truth value. To choose values "
                "elementwise, use tl.where(cond, x, y). To combine masks elementwise, use `&` "
                "and `|` instead of `and` and `or`."
            )
        return bool(self._scalar())

    def __index__(self) -> int:
        if not self.dtype.is_int():
            raise TypeError(
                f"a {self.dtype} value can't be used as an integer. Convert it with "
                "`.to(tl.int32)` first."
            )
        return int(self._scalar())

    def __int__(self) -> int:
        return int(self._scalar())

    def __float__(self) -> float:
        return float(self._scalar())

    # Operators.
    def __add__(self, o):
        return binary("add", self, o)

    def __radd__(self, o):
        return binary("add", o, self)

    def __sub__(self, o):
        return binary("sub", self, o)

    def __rsub__(self, o):
        return binary("sub", o, self)

    def __mul__(self, o):
        return binary("mul", self, o)

    def __rmul__(self, o):
        return binary("mul", o, self)

    def __truediv__(self, o):
        return binary("div", self, o)

    def __rtruediv__(self, o):
        return binary("div", o, self)

    def __floordiv__(self, o):
        return binary("floordiv", self, o)

    def __rfloordiv__(self, o):
        return binary("floordiv", o, self)

    def __mod__(self, o):
        return binary("mod", self, o)

    def __rmod__(self, o):
        return binary("mod", o, self)

    def __and__(self, o):
        return binary("and", self, o)

    def __rand__(self, o):
        return binary("and", o, self)

    def __or__(self, o):
        return binary("or", self, o)

    def __ror__(self, o):
        return binary("or", o, self)

    def __xor__(self, o):
        return binary("xor", self, o)

    def __rxor__(self, o):
        return binary("xor", o, self)

    def __lshift__(self, o):
        return binary("shl", self, o)

    def __rlshift__(self, o):
        return binary("shl", o, self)

    def __rshift__(self, o):
        return binary("shr", self, o)

    def __rrshift__(self, o):
        return binary("shr", o, self)

    def __eq__(self, o):  # type: ignore[override]
        return binary("eq", self, o)

    def __ne__(self, o):  # type: ignore[override]
        return binary("ne", self, o)

    def __lt__(self, o):
        return binary("lt", self, o)

    def __le__(self, o):
        return binary("le", self, o)

    def __gt__(self, o):
        return binary("gt", self, o)

    def __ge__(self, o):
        return binary("ge", self, o)

    def __neg__(self):
        return unary("neg", self)

    def __pos__(self):
        return self

    def __invert__(self):
        return unary("not", self)

    __hash__ = None  # type: ignore[assignment]


def _check_index(idx: Any) -> Any:
    items = idx if isinstance(idx, tuple) else (idx,)
    for it in items:
        if it is None or (isinstance(it, slice) and it == slice(None)):
            continue
        raise CompilationError(
            "tiles support only `None` and `:` in subscripts, as in `x[:, None]`, but got "
            f"{it!r}. To select elements, use a mask or tl.where, or load the element from "
            "memory with tl.load."
        )
    return idx


class IPointer:
    """An interpreter pointer or pointer tile: a flat storage view plus element offsets."""

    __slots__ = ("flat", "offsets", "elem")

    def __init__(self, flat: np.ndarray, offsets: Any, elem: core.dtype):
        self.flat = flat
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.elem = elem

    @property
    def dtype(self) -> core.dtype:
        """The pointee dtype, so `x_ptr.dtype` and `x_ptr.dtype.element_ty` both work."""
        return self.elem

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.offsets.shape)

    def with_offsets(self, offsets: Any) -> IPointer:
        return IPointer(self.flat, offsets, self.elem)

    def __add__(self, o):
        return _pointer_add(self, o, 1)

    __radd__ = __add__

    def __sub__(self, o):
        return _pointer_add(self, o, -1)

    def __getitem__(self, idx):
        return self.with_offsets(self.offsets[_check_index(idx)])

    def __getattr__(self, name: str) -> Any:
        m = core.TILE_METHODS.get(name)
        if m is None:
            raise AttributeError(
                f"pointers have no attribute {name!r}. Check the spelling, or call the tl "
                "function of that name."
            )
        return lambda *a, **k: m.interp(self, *a, **k)

    def __repr__(self) -> str:
        return f"IPointer(*{self.elem}, offsets={self.offsets!r})"

    def check_bounds(self, offsets: np.ndarray, mask: np.ndarray | None, what: str) -> None:
        n = self.flat.shape[0]
        bad = (offsets < 0) | (offsets >= n)
        if mask is not None:
            bad &= mask
        if bad.any():
            first = int(offsets[bad].flat[0])
            hint = (" Pass a mask that excludes it." if mask is None else
                    " The mask doesn't exclude it; check the mask's bounds.")
            raise IndexError(
                f"out-of-bounds {what}{_where()}: element offset {first} is outside the "
                f"buffer of {n} elements.{hint}"
            )


def _pointer_add(p: IPointer, o: Any, sign: int) -> IPointer:
    if isinstance(o, IPointer):
        raise CompilationError(
            "pointers support only `+` and `-` with an integer offset. Compute the offset "
            "first, as in `ptr + i * stride`."
        )
    if isinstance(o, ITile):
        if not o.dtype.is_int():
            raise CompilationError(
                f"a pointer offset must be an integer, but got {o.dtype}. Convert it with "
                "`.to(tl.int32)`."
            )
        off = o.data.astype(np.int64)
    elif isinstance(o, (int, np.integer)) and not isinstance(o, bool):
        off = np.int64(o)
    else:
        raise CompilationError(
            f"a pointer offset must be an integer, but got {o!r}. Convert it with "
            "`.to(tl.int32)`."
        )
    return p.with_offsets(p.offsets + sign * off)


class IDesc:
    """An interpreter tensor descriptor."""

    __slots__ = ("base", "shape", "strides", "block_shape")

    def __init__(self, base: IPointer, shape, strides, block_shape):
        self.base = base
        self.shape = tuple(int(s) for s in shape)
        self.strides = tuple(int(s) for s in strides)
        self.block_shape = tuple(block_shape)

    @property
    def dtype(self) -> core.dtype:
        return self.base.elem

    def __getattr__(self, name: str) -> Any:
        m = core.DESC_METHODS.get(name)
        if m is None:
            raise AttributeError(f"tensor descriptors have no attribute {name!r}")
        return lambda *a, **k: m.interp(self, *a, **k)

    def block_offsets(self, offsets: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
        """Returns the element offsets and the in-bounds mask of the block at `offsets`."""
        if len(offsets) != len(self.shape):
            raise CompilationError(
                f"the descriptor has rank {len(self.shape)}, but got {len(offsets)} offsets"
            )
        rank = len(self.shape)
        off = np.zeros(self.block_shape, dtype=np.int64)
        mask = np.ones(self.block_shape, dtype=bool)
        for d in range(rank):
            idx = int(offsets[d]) + np.arange(self.block_shape[d], dtype=np.int64)
            view = [1] * rank
            view[d] = self.block_shape[d]
            idx = idx.reshape(view)
            off = off + idx * self.strides[d]
            mask = mask & (idx >= 0) & (idx < self.shape[d])
        return off + self.base.offsets, mask


# ---------------------------------------------------------------------------
# Elementwise semantics
# ---------------------------------------------------------------------------


def _operand(x: Any) -> core.dtype | semantic.Literal:
    if isinstance(x, ITile):
        return x.dtype
    x = core.unwrap(x)
    if isinstance(x, np.generic):
        x = x.item()
    if semantic.is_literal(x):
        return x
    raise CompilationError(f"expected a number or a tile, but got {type(x).__name__} {x!r}")


def as_compute(x: Any, dt: core.dtype) -> np.ndarray:
    """Returns `x` as an array in the compute dtype of `dt`."""
    if isinstance(x, ITile):
        if x.dtype is dt:
            return x.data.astype(_compute_np(dt), copy=False)
        return convert(x, dt).data.astype(_compute_np(dt), copy=False)
    return literal_array(core.unwrap(x), dt).astype(_compute_np(dt))


def literal_array(x: semantic.Literal, dt: core.dtype) -> np.ndarray:
    """Returns the literal `x` as a 0-d array of `dt`, converted like a C cast.

    An integer that fits in the width of `dt` as a signed or an unsigned number wraps to
    two's complement, so `-1` becomes the largest `uint32`, as it does on the GPU. A float
    that converts to an integer type must be finite and fit the type after truncation
    toward zero.

    Raises:
        CompilationError: The literal doesn't fit in `dt`.
    """
    if isinstance(x, float) and dt.is_int() and not dt.is_bool():
        lo, hi = semantic.int_range(dt)
        if not (math.isfinite(x) and lo <= math.trunc(x) <= hi):
            raise CompilationError(
                f"the value {x!r} doesn't fit in {dt}, whose range is {lo} to {hi}. Use a value "
                "in that range, or a floating-point dtype."
            )
    v = semantic.coerce_literal(x, dt)
    try:
        return np.asarray(v, dtype=_np(dt))
    except OverflowError:
        # NumPy refuses out-of-range Python ints. Wrap the ones that fit in the width.
        bits = dt.primitive_bitwidth
        if not -(1 << (bits - 1)) <= v < (1 << bits):
            lo, hi = semantic.int_range(dt)
            raise CompilationError(
                f"the value {v} doesn't fit in {dt}, whose range is {lo} to {hi}. Use a value "
                "in that range, or a wider dtype."
            ) from None
        v &= (1 << bits) - 1
        if dt.is_signed() and v >> (bits - 1):
            v -= 1 << bits
        return np.asarray(v, dtype=_np(dt))


def wrap(data: np.ndarray, dt: core.dtype) -> ITile:
    """Rounds a computed array to `dt` and wraps it."""
    return ITile(np.asarray(data).astype(_np(dt), copy=False), dt)


def convert(x: ITile, dt: core.dtype) -> ITile:
    """Converts a tile to `dt` as compiled kernels do.

    Conversion to `int1` is `x != 0`. Float-to-integer conversion truncates toward zero
    and follows Metal for values out of range; see `float_to_int`.
    """
    if x.dtype is dt:
        return x
    data = x.data
    if dt.is_bool():
        return ITile(data != 0, dt)
    if x.dtype.is_floating() and dt.is_int():
        return ITile(float_to_int(data, dt), dt)
    if x.dtype is core.bfloat16 or dt is core.bfloat16:
        data = data.astype(np.float32)
    return ITile(data.astype(_np(dt)), dt)


_TWO_64 = float(1 << 64)
_TWO_63 = float(1 << 63)


def float_to_int(data: np.ndarray, dt: core.dtype) -> np.ndarray:
    """Converts floats to the integer type `dt` the way Metal does on the GPU.

    Values truncate toward zero. For `int64`, the result is the truncated value modulo
    2**64, as a signed number, and infinities become 0. For every other integer type,
    values out of range saturate to the type's minimum or maximum, including
    infinities. NaN becomes 0 for every type. Codegen converts `float16` and `bfloat16`
    through `float32`, which is exact, so this function does too.
    """
    f = np.asarray(data).astype(np.float32).astype(np.float64)  # exact
    nan = np.isnan(f)
    t = np.trunc(np.where(nan, 0.0, f))
    if dt is core.int64:
        # fmod is exact, so `r` is the exact remainder of trunc(x) by 2**64.
        with np.errstate(invalid="ignore"):
            r = np.fmod(t, _TWO_64)
        r = np.where(r >= _TWO_63, r - _TWO_64, np.where(r < -_TWO_63, r + _TWO_64, r))
        return np.where(np.isfinite(r), r, 0.0).astype(np.int64)
    lo, hi = semantic.int_range(dt)
    npdt = _np(dt)
    if dt.primitive_bitwidth < 64:
        return np.clip(t, lo, hi).astype(npdt)
    # float64 can't hold 2**64 - 1, so fill the saturated values after converting.
    out = np.clip(t, 0.0, np.nextafter(_TWO_64, 0.0)).astype(npdt)
    return np.where(t >= _TWO_64, npdt.type(hi), out)


def _c_div(a: np.ndarray, b: np.ndarray, signed: bool) -> np.ndarray:
    b = np.where(b == 0, 1, b).astype(b.dtype)  # Division by zero is undefined; avoid traps.
    q = np.floor_divide(a, b)
    if signed:
        q = q + ((q < 0) & (q * b != a)).astype(q.dtype)
    return q


def _erf(x: np.ndarray) -> np.ndarray:
    return np.vectorize(math.erf, otypes=[np.float64])(x.astype(np.float64)).astype(x.dtype)


_BINARY: dict[str, Callable[[np.ndarray, np.ndarray, core.dtype], np.ndarray]] = {
    "add": lambda a, b, dt: a + b,
    "sub": lambda a, b, dt: a - b,
    "mul": lambda a, b, dt: a * b,
    "div": lambda a, b, dt: a / b,
    "floordiv": lambda a, b, dt: _c_div(a, b, dt.is_signed()),
    "mod": lambda a, b, dt: np.fmod(a, b if dt.is_floating() else np.where(b == 0, 1, b)),
    "and": lambda a, b, dt: a & b,
    "or": lambda a, b, dt: a | b,
    "xor": lambda a, b, dt: a ^ b,
    "shl": lambda a, b, dt: np.left_shift(a, b.astype(a.dtype)),
    "shr": lambda a, b, dt: np.right_shift(a, b.astype(a.dtype)),
    "min": lambda a, b, dt: np.fmin(a, b) if dt.is_floating() else np.minimum(a, b),
    "max": lambda a, b, dt: np.fmax(a, b) if dt.is_floating() else np.maximum(a, b),
    "eq": lambda a, b, dt: a == b,
    "ne": lambda a, b, dt: a != b,
    "lt": lambda a, b, dt: a < b,
    "le": lambda a, b, dt: a <= b,
    "gt": lambda a, b, dt: a > b,
    "ge": lambda a, b, dt: a >= b,
}


def binary(op: str, x: Any, y: Any) -> ITile:
    """Computes a binary op or comparison with Enceladus's promotion rules."""
    if isinstance(y, IPointer) and op == "add":
        return _pointer_add(y, x, 1)
    if isinstance(x, IPointer) and op in ("add", "sub"):
        return _pointer_add(x, y, 1 if op == "add" else -1)
    if isinstance(x, IPointer) or isinstance(y, IPointer):
        raise CompilationError(
            "pointers support only `+` and `-` with an integer offset. Compute the offset "
            "first, as in `ptr + i * stride`."
        )
    dt = semantic.computation_dtype(op, _operand(x), _operand(y))
    a, b = as_compute(x, dt), as_compute(y, dt)
    semantic.broadcast_shapes(a.shape, b.shape)
    out = _BINARY[op](a, b, dt)
    if op in semantic.CMP_OPS:
        return ITile(np.asarray(out, dtype=bool), core.int1)
    return wrap(out, dt)


_UNARY: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "neg": lambda a: -a,
    "abs": np.abs,
    "exp": np.exp,
    "exp2": np.exp2,
    "log": np.log,
    "log2": np.log2,
    "sqrt": np.sqrt,
    "rsqrt": lambda a: 1.0 / np.sqrt(a),
    "sin": np.sin,
    "cos": np.cos,
    "tanh": np.tanh,
    "sigmoid": lambda a: 1.0 / (1.0 + np.exp(-a)),
    "erf": _erf,
    "floor": np.floor,
    "ceil": np.ceil,
}


def unary(op: str, x: Any, name: str | None = None) -> ITile:
    """Computes a unary op. Math functions need floating-point input."""
    if not isinstance(x, ITile):
        x = wrap(np.asarray(core.unwrap(x)), semantic.literal_dtype(core.unwrap(x)))
    dt = x.dtype
    if op in ir.MATH_UNARY and not dt.is_floating():
        raise CompilationError(
            f"tl.{name or op} needs a floating-point input, but got {dt}. Convert it first, "
            "for example with `x.to(tl.float32)`."
        )
    if op == "not":
        if not dt.is_int():
            raise CompilationError(f"`~` needs an integer or boolean operand, but got {dt}")
        return ITile(np.logical_not(x.data) if dt.is_bool() else np.invert(x.data), dt)
    if op == "neg" and dt.is_bool():
        dt = core.int32
    return wrap(_UNARY[op](as_compute(x, dt)), dt)


def to_tile(x: Any, dt: core.dtype | None = None) -> ITile:
    """Converts a literal or tile to a tile, converting to `dt` when it's given."""
    if isinstance(x, ITile):
        return convert(x, dt) if dt is not None else x
    x = core.unwrap(x)
    if isinstance(x, np.generic):
        x = x.item()
    if not semantic.is_literal(x):
        raise CompilationError(f"expected a number or a tile, but got {type(x).__name__} {x!r}")
    dt = dt or semantic.literal_dtype(x)
    return ITile(literal_array(x, dt), dt)


# ---------------------------------------------------------------------------
# Loops and comparisons
# ---------------------------------------------------------------------------


def typed_range(*args: Any) -> Iterator[ITile]:
    """Iterates over `range(*args)`, yielding the loop variable as a typed scalar.

    As in compiled code, the loop variable is an `int32` scalar when every bound fits in
    `int32`, an `int64` scalar when a bound is `uint32`, `int64`, or a literal outside the
    `int32` range, and a `uint64` scalar when a bound is `uint64` and none is negative.
    Arithmetic on it wraps like arithmetic on any other runtime scalar.
    """
    if not 1 <= len(args) <= 3:
        raise CompilationError(f"range() takes 1 to 3 arguments, but got {len(args)}")
    bounds = [core.unwrap(a) for a in ((0, args[0], 1) if len(args) == 1 else (*args, 1)[:3])]
    ranges = []  # The range of values each bound can take.
    for v in bounds:
        if isinstance(v, np.integer):
            v = int(v)
        if isinstance(v, ITile) and v.shape == () and v.dtype.is_int() and not v.dtype.is_bool():
            ranges.append(semantic.int_range(v.dtype))
        elif isinstance(v, int) and not isinstance(v, bool):
            ranges.append((v, v))
        else:
            what = f"a {v.dtype} tile of shape {v.shape}" if isinstance(v, ITile) else repr(v)
            raise CompilationError(
                f"range() bounds must be integer scalars, not {what}. Convert a float bound "
                "with `.to(tl.int32)`."
            )
    for dt in (core.int32, core.int64, core.uint64):
        lo, hi = semantic.int_range(dt)
        if all(lo <= a and b <= hi for a, b in ranges):
            break
    else:
        raise CompilationError(
            "range() has a tl.uint64 bound and a signed bound or negative step, and no counter "
            "type holds the values of both. Convert the bounds with `.to(tl.int64)`, or make "
            "them all unsigned."
        )
    lb, ub, step = (int(to_tile(v, dt).data) for v in bounds)
    if step == 0:
        raise CompilationError("the step of range() can't be zero. Use a nonzero step.")
    return _typed_range(lb, ub, step, dt)


def _typed_range(lb: int, ub: int, step: int, dt: core.dtype) -> Iterator[ITile]:
    npdt = _np(dt)
    new = object.__new__
    for v in range(lb, ub, step):
        t = new(ITile)
        t.data = np.array(v, npdt)
        t.dtype = dt
        yield t


_CMP_FUNCS: dict[str, Callable[[Any, Any], Any]] = {
    "lt": operator.lt, "le": operator.le, "gt": operator.gt, "ge": operator.ge,
    "eq": operator.eq, "ne": operator.ne, "is": operator.is_, "is_not": operator.is_not,
    "in": lambda a, b: a in b, "not_in": lambda a, b: a not in b,
}  # fmt: skip


def chain_compare(left: Any, op: str, right: Any, *rest: Any) -> Any:
    """Evaluates the chained comparison `left op right op2 right2 ...`.

    `rest` alternates operator names and zero-argument functions that evaluate the later
    operands, so each operand is evaluated at most once and the chain stops at the first
    false comparison, as in Python. Chains of scalars, including runtime scalars, work;
    as in compiled code, a chain that compares tiles is refused.
    """
    result = _chain_link(_CMP_FUNCS[op](left, right))
    for i in range(0, len(rest), 2):
        if not result:
            return result
        left, right = right, rest[i + 1]()
        result = _chain_link(_CMP_FUNCS[rest[i]](left, right))
    return result


def _chain_link(r: Any) -> Any:
    if isinstance(r, ITile) and r.data.ndim != 0:
        raise CompilationError(
            f"chained comparisons such as `a < b < c` need scalars, but one comparison gives a "
            f"tile of shape {r.shape}. Combine the comparisons elementwise instead, as in "
            "`(a < b) & (b < c)`."
        )
    return r


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------


def run_grid(fn: Callable[..., Any], grid: tuple[int, int, int], kwargs: dict[str, Any]) -> None:
    """Runs `fn(**kwargs)` once per program in `grid`, sequentially.

    Errors raised inside the kernel that don't carry a location get the kernel source
    line of the innermost kernel frame.
    """
    from enceladus.interpreter.rewrite import interpretable

    fn = interpretable(fn)
    state = core.INTERP
    saved = (state.program_id, state.grid)
    state.depth += 1
    state.grid = grid
    try:
        with np.errstate(all="ignore"):
            for z in range(grid[2]):
                for y in range(grid[1]):
                    for x in range(grid[0]):
                        state.program_id = (x, y, z)
                        fn(**kwargs)
    except CompilationError as e:
        raise e.with_loc(_kernel_loc_from_tb(e.__traceback__)) from None
    finally:
        state.depth -= 1
        state.program_id, state.grid = saved
