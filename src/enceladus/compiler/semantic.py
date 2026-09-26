"""Type promotion, broadcasting, and the IR helpers that the builtins share.

The dtype rules at the top of this module are pure functions over `tl` dtypes and Python
literals. The frontend and the interpreter both call them, so the two execution paths
agree on result types.

The rules follow Triton:

- A Python literal adopts the dtype of the other operand when the kinds match (an `int`
  literal with an integer tile, a `float` literal with a float tile). An `int` literal with
  a float operand also adopts the float type. A `float` literal with an integer operand
  gives `float32`.
- Two float types promote to the wider one. `float16` with `bfloat16` gives `float32`.
- An integer with a float gives the float type, so `int32 op float32` gives `float32`.
- Two integer types of the same signedness promote to the wider one. With mixed
  signedness, the unsigned type wins if it's at least as wide as the signed one.
- Arithmetic on `int1` operands computes in `int32`. Bitwise ops and comparisons keep
  `int1`.
- `/` always computes in floating point: two integers divide in `float32`.
- `//` and `%` on integers truncate toward zero, as in C. This differs from Python, where
  `//` rounds toward negative infinity. `%` on floats is `fmod`, whose result has the sign
  of the dividend. `//` on floats is an error.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from enceladus.compiler import ir
from enceladus.compiler.errors import CompilationError
from enceladus.language import core

Literal = bool | int | float
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1

ARITH_OPS = frozenset(["add", "sub", "mul", "div", "floordiv", "mod", "min", "max"])
BITWISE_OPS = frozenset(["and", "or", "xor"])
SHIFT_OPS = frozenset(["shl", "shr"])
CMP_OPS = frozenset(ir.CMP_PREDS)

_OP_SYMBOL = {
    "add": "+", "sub": "-", "mul": "*", "div": "/", "floordiv": "//", "mod": "%",
    "and": "&", "or": "|", "xor": "^", "shl": "<<", "shr": ">>", "eq": "==", "ne": "!=",
    "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "min": "minimum", "max": "maximum",
}  # fmt: skip


def is_literal(x: Any) -> bool:
    return isinstance(x, (bool, int, float))


def int_range(dt: core.dtype) -> tuple[int, int]:
    bits = dt.primitive_bitwidth
    if dt.is_bool():
        return 0, 1
    if dt.is_signed():
        return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return 0, (1 << bits) - 1


def literal_dtype(v: Literal) -> core.dtype:
    """Returns the dtype of a Python literal on its own: `i1`, `i32`, `i64`, or `fp32`."""
    if isinstance(v, bool):
        return core.int1
    if isinstance(v, int):
        if INT32_MIN <= v <= INT32_MAX:
            return core.int32
        if -(1 << 63) <= v < (1 << 63):
            return core.int64
        if 0 <= v < (1 << 64):
            return core.uint64
        raise CompilationError(f"integer literal {v} doesn't fit in 64 bits")
    return core.float32


def adopt_literal(v: Literal, dt: core.dtype) -> core.dtype:
    """Returns the dtype a literal takes when it's combined with an operand of dtype `dt`."""
    if isinstance(v, bool) and dt.is_bool():
        return core.int1
    if isinstance(v, float):
        return dt if dt.is_floating() else core.float32
    if dt.is_floating():
        return dt
    if dt.is_bool():
        return literal_dtype(int(v))
    lo, hi = int_range(dt)
    return dt if lo <= int(v) <= hi else literal_dtype(int(v))


def integer_promote(a: core.dtype, b: core.dtype) -> core.dtype:
    if a is b:
        return a
    if a.is_bool():
        return b
    if b.is_bool():
        return a
    if a.is_signed() == b.is_signed():
        return a if a.primitive_bitwidth >= b.primitive_bitwidth else b
    s, u = (a, b) if a.is_signed() else (b, a)
    return u if u.primitive_bitwidth >= s.primitive_bitwidth else s


def float_promote(a: core.dtype, b: core.dtype) -> core.dtype:
    if a is b:
        return a
    if {a, b} == {core.float16, core.bfloat16}:
        return core.float32
    return a if a.primitive_bitwidth >= b.primitive_bitwidth else b


def computation_dtype(op: str, a: core.dtype | Literal, b: core.dtype | Literal) -> core.dtype:
    """Returns the dtype in which a binary op or comparison computes.

    Args:
        op: A binary op name (`"add"`, `"div"`, ...), a comparison (`"lt"`, ...), or
            `"select"` for the two value operands of `where`.
        a: The left operand's dtype, or the Python literal itself.
        b: The right operand's dtype, or the Python literal itself.

    Raises:
        CompilationError: The operand types don't support the op.
    """
    if is_literal(a) and is_literal(b):
        a, b = literal_dtype(a), literal_dtype(b)
    elif is_literal(a):
        a = adopt_literal(a, b)
    elif is_literal(b):
        b = adopt_literal(b, a)
    sym = _OP_SYMBOL.get(op, op)
    if op in BITWISE_OPS or op in SHIFT_OPS:
        if not (a.is_int() and b.is_int()):
            raise CompilationError(
                f"`{sym}` needs integer operands, but got {a} and {b}. Cast the operands "
                "with `.to(tl.int32)` first."
            )
    if op == "floordiv" and (a.is_floating() or b.is_floating()):
        raise CompilationError(
            f"`//` needs integer operands, but got {a} and {b}. For floats, use "
            "`tl.floor(x / y)`."
        )
    if a.is_floating() or b.is_floating():
        if a.is_floating() and b.is_floating():
            return float_promote(a, b)
        return a if a.is_floating() else b
    if op == "div":
        return core.float32
    t = integer_promote(a, b)
    if t.is_bool() and op in ARITH_OPS | SHIFT_OPS:
        return core.int32
    return t


def broadcast_shapes(a: Sequence[int], b: Sequence[int]) -> tuple[int, ...]:
    """Returns the broadcast of two shapes, aligning trailing dimensions like NumPy."""
    a, b = tuple(a), tuple(b)
    n = max(len(a), len(b))
    a = (1,) * (n - len(a)) + a
    b = (1,) * (n - len(b)) + b
    out = []
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        if x != y and x != 1 and y != 1:
            raise CompilationError(
                f"can't broadcast shapes {a} and {b}: dimension {i} is {x} in one and {y} in "
                "the other."
            )
        out.append(max(x, y))
    return tuple(out)


def check_shape(shape: Any, what: str) -> tuple[int, ...]:
    """Checks that `shape` is a valid tile shape and returns it as a tuple.

    Raises:
        CompilationError: A dimension isn't a compile-time power of two from 1 to 65536.
    """
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(core.unwrap(d) for d in shape)
    for i, d in enumerate(shape):
        if isinstance(d, bool) or not isinstance(d, int):
            raise CompilationError(
                f"dimension {i} of the {what} shape must be a compile-time integer, but got "
                f"{describe(d)}. Annotate the kernel parameter that sets it as tl.constexpr."
            )
        if not ir.is_pow2_dim(d):
            raise CompilationError(
                f"dimension {i} of the {what} shape {shape} is {d}, but every tile dimension "
                "must be a power of two from 1 to 65536. Round the block size up to a power "
                "of two and mask the extra elements."
            )
    return shape


def describe(x: Any) -> str:
    """Describes a frontend value for an error message."""
    if isinstance(x, ir.Value):
        name = f" `{x.name_hint}`" if x.name_hint else ""
        return f"a runtime value{name} of type {x.type}"
    return f"{type(x).__name__} {x!r}"


# ---------------------------------------------------------------------------
# IR helpers
# ---------------------------------------------------------------------------


def dtype_of(v: ir.Value) -> core.dtype:
    """Returns the element dtype of a numeric value."""
    e = ir.elem_of(v.type)
    if not isinstance(e, ir.ScalarType):
        raise CompilationError(f"expected a numeric value, but got {describe(v)}")
    return e.dtype


def is_tile(x: Any) -> bool:
    return isinstance(x, ir.Value) and isinstance(x.type, ir.TileType)


def is_pointer(x: Any) -> bool:
    return isinstance(x, ir.Value) and isinstance(ir.elem_of(x.type), ir.PointerType)


def operand_dtype(x: Any) -> core.dtype | Literal:
    """Returns the dtype of a runtime value, or the literal itself."""
    if isinstance(x, ir.Value):
        return dtype_of(x)
    x = core.unwrap(x)
    if is_literal(x):
        return x
    raise CompilationError(f"expected a number or a tile, but got {describe(x)}")


def coerce_literal(value: Literal, dt: core.dtype) -> Literal:
    if dt.is_bool():
        return bool(value)
    if dt.is_int():
        return int(value)  # Truncates a float toward zero, like a C cast.
    return float(value)


def const(b: ir.Builder, value: Literal, dt: core.dtype) -> ir.Value:
    return b.create("const", [], [ir.scalar(dt)], {"value": coerce_literal(value, dt)}).result


def to_value(b: ir.Builder, x: Any, dt: core.dtype | None = None) -> ir.Value:
    """Materializes a literal as a `const`, and casts to `dt` when it's given."""
    x = core.unwrap(x)
    if isinstance(x, ir.Value):
        return cast(b, x, dt) if dt is not None else x
    if is_literal(x):
        return const(b, x, dt or literal_dtype(x))
    raise CompilationError(f"expected a number or a tile, but got {describe(x)}")


def cast(b: ir.Builder, v: ir.Value, dt: core.dtype, bitcast: bool = False) -> ir.Value:
    """Converts `v` to element dtype `dt`. A bitcast reinterprets the bits instead."""
    if is_pointer(v):
        raise CompilationError(f"can't convert a pointer ({v.type}) to {dt}")
    src = dtype_of(v)
    if src is dt:
        return v
    t = ir.with_elem(v.type, ir.scalar(dt))
    if bitcast:
        if src.primitive_bitwidth != dt.primitive_bitwidth or src.is_bool() or dt.is_bool():
            raise CompilationError(
                f"a bitcast needs types of the same width, but {src} has "
                f"{src.primitive_bitwidth} bits and {dt} has {dt.primitive_bitwidth}"
            )
        return b.create("bitcast", [v], [t]).result
    return b.create("cast", [v], [t]).result


def splat(b: ir.Builder, v: ir.Value, shape: tuple[int, ...]) -> ir.Value:
    return b.create("splat", [v], [ir.TileType(shape, v.type)]).result


def expand_dims(b: ir.Builder, v: ir.Value, axis: int) -> ir.Value:
    s = list(ir.shape_of(v.type))
    if not -len(s) - 1 <= axis <= len(s):
        raise CompilationError(f"axis {axis} is out of range for a tile of rank {len(s)}")
    axis = axis % (len(s) + 1)
    if not s:
        return splat(b, v, (1,))
    s.insert(axis, 1)
    t = ir.TileType(tuple(s), ir.elem_of(v.type))
    return b.create("expand_dims", [v], [t], {"axis": axis}).result


def broadcast_to(b: ir.Builder, v: ir.Value, shape: tuple[int, ...]) -> ir.Value:
    """Broadcasts `v` to `shape`, splatting scalars and prepending size-1 dimensions."""
    shape = tuple(shape)
    if not shape:
        if is_tile(v):
            raise CompilationError(f"can't broadcast a tile of shape {v.type.shape} to a scalar")
        return v
    if not is_tile(v):
        return splat(b, v, shape)
    src = v.type.shape
    if src == shape:
        return v
    if len(src) > len(shape):
        raise CompilationError(f"can't broadcast shape {src} to the smaller rank of {shape}")
    while len(ir.shape_of(v.type)) < len(shape):
        v = expand_dims(b, v, 0)
    src = v.type.shape
    for i, (x, y) in enumerate(zip(src, shape, strict=True)):
        if x != y and x != 1:
            raise CompilationError(
                f"can't broadcast shape {src} to {shape}: dimension {i} is {x}, not 1 or {y}"
            )
    if src == shape:
        return v
    return b.create("broadcast", [v], [ir.TileType(shape, v.type.elem)]).result


def _shape(x: Any) -> tuple[int, ...]:
    return ir.shape_of(x.type) if isinstance(x, ir.Value) else ()


def binary(b: ir.Builder, op: str, x: Any, y: Any) -> ir.Value:
    """Emits a binary op or comparison with promotion and broadcasting."""
    x, y = core.unwrap(x), core.unwrap(y)
    if is_pointer(x) or is_pointer(y):
        return _pointer_binary(b, op, x, y)
    dt = computation_dtype(op, operand_dtype(x), operand_dtype(y))
    shape = broadcast_shapes(_shape(x), _shape(y))
    xv = broadcast_to(b, to_value(b, x, dt), shape)
    yv = broadcast_to(b, to_value(b, y, dt), shape)
    if op in CMP_OPS:
        t = ir.with_elem(xv.type, ir.i1)
        return b.create("cmp", [xv, yv], [t], {"pred": op}).result
    return b.create("binary", [xv, yv], [xv.type], {"op": op}).result


def _pointer_binary(b: ir.Builder, op: str, x: Any, y: Any) -> ir.Value:
    if op == "add" and is_pointer(y) and not is_pointer(x):
        x, y = y, x
    if op not in ("add", "sub") or is_pointer(y):
        raise CompilationError(
            f"pointers support only `+` and `-` with an integer offset, not `{_OP_SYMBOL[op]}`"
        )
    yd = operand_dtype(y)
    if isinstance(yd, float) or (isinstance(yd, core.dtype) and not yd.is_int()):
        raise CompilationError(f"a pointer offset must be an integer, but got {describe(y)}")
    off = to_value(b, y)
    if op == "sub":
        off = b.create("unary", [off], [off.type], {"op": "neg"}).result
    if dtype_of(off).is_bool():
        off = cast(b, off, core.int32)
    shape = broadcast_shapes(_shape(x), _shape(off))
    xp = broadcast_to(b, x, shape)
    off = broadcast_to(b, off, shape)
    return b.create("addptr", [xp, off], [xp.type]).result


def unary(b: ir.Builder, op: str, x: Any, name: str | None = None) -> ir.Value:
    """Emits a unary op. Math functions need floating-point input."""
    v = to_value(b, x)
    dt = dtype_of(v)
    if op in ir.MATH_UNARY and not dt.is_floating():
        raise CompilationError(
            f"tl.{name or op} needs a floating-point input, but got {dt}. Convert it first, "
            "for example with `x.to(tl.float32)`."
        )
    if op == "not" and not dt.is_int():
        raise CompilationError(f"`~` needs an integer or boolean operand, but got {dt}")
    return b.create("unary", [v], [v.type], {"op": op}).result


def to_bool(b: ir.Builder, x: Any) -> ir.Value:
    """Converts a value to `int1` by comparing it with zero."""
    v = to_value(b, x)
    if dtype_of(v).is_bool():
        return v
    return binary(b, "ne", v, 0)


def where(b: ir.Builder, cond: Any, x: Any, y: Any) -> ir.Value:
    """Emits `select(cond, x, y)` with promotion and broadcasting of all three operands."""
    c = to_bool(b, cond)
    x, y = core.unwrap(x), core.unwrap(y)
    if is_pointer(x) or is_pointer(y):
        if not (is_pointer(x) and is_pointer(y) and ir.elem_of(x.type) == ir.elem_of(y.type)):
            raise CompilationError("tl.where needs two pointers of the same type, or no pointers")
        xv, yv = x, y
    else:
        dt = computation_dtype("select", operand_dtype(x), operand_dtype(y))
        xv, yv = to_value(b, x, dt), to_value(b, y, dt)
    shape = broadcast_shapes(broadcast_shapes(_shape(c), _shape(xv)), _shape(yv))
    c, xv, yv = (broadcast_to(b, v, shape) for v in (c, xv, yv))
    return b.create("select", [c, xv, yv], [xv.type]).result
