"""Constant folding, algebraic identities, common-subexpression elimination, and DCE.

`fold` rewrites ops whose result is known at compile time:

- Arithmetic, comparisons, and casts on constants become constants. Constants are
  `const` scalars and the tiles that `full` makes or that shape ops and `splat` build
  from a constant. Results come from the interpreter's own functions, so they match
  interpreted runs bit for bit: integers wrap around, and floats round as IEEE 754
  does, with 16-bit floats computed in `float32` and rounded after each op.
- Operations whose result the language leaves undefined stay unfolded, so the GPU
  computes them as before: integer division or remainder by zero, `INT_MIN / -1`,
  shifts by a negative amount or by the bit width or more, negation and `abs` of the
  most negative integer, and float-to-integer conversions of NaN, infinity, or values
  out of range.
- Algebraic identities hold for every operand value: `x + 0`, `x * 1`, `x * 0`,
  `x - x`, and the bitwise identities for integers; `x * 1.0`, `x / 1.0`,
  `x + (-0.0)`, and `x - 0.0` for floats. `x * 0.0` and `x + 0.0` stay, because NaN,
  infinity, and `-0.0` break them. `tl.where` on a constant condition, or with two
  equal values, selects one of them.
- An `if` on a constant condition keeps only the taken branch.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

from enceladus.compiler import ir
from enceladus.compiler.errors import CompilationError
from enceladus.language import core

# Ops with no side effects whose results depend only on operands and attributes.
PURE_OPS = frozenset(
    """const splat arange full program_id num_programs binary cmp unary fma select cast
    bitcast broadcast expand_dims reshape trans join split addptr make_desc""".split()
)
# Ops that are safe to delete when their results are unused.
REMOVABLE_OPS = PURE_OPS | {"load", "dot", "reduce", "scan", "desc_load", "local_load"}

# Shape ops that keep a constant tile constant.
_CONST_VIEWS = frozenset(["splat", "broadcast", "expand_dims", "reshape", "trans"])


def _freeze(v: Any) -> Any:
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    if isinstance(v, float):
        # By bits, so that 0.0 and -0.0 (equal in Python) stay distinct, and NaN
        # matches itself.
        return ("float", v.hex())
    return v


# ---------------------------------------------------------------------------
# Constant folding and algebraic identities
# ---------------------------------------------------------------------------


def _dtype(v: ir.Value) -> core.dtype | None:
    e = ir.elem_of(v.type)
    return e.dtype if isinstance(e, ir.ScalarType) else None


def const_value(v: ir.Value) -> bool | int | float | None:
    """Returns the value of every element of `v` when it's a compile-time constant."""
    op = v.defining_op
    while op is not None and op.name in _CONST_VIEWS:
        op = op.operands[0].defining_op
    if op is None or op.name not in ("const", "full"):
        return None
    return op.attrs["value"]


class _Const:
    """A constant operand: its Python value, its dtype, and a 0-d interpreter tile."""

    def __init__(self, value: bool | int | float, dt: core.dtype, tile: Any) -> None:
        self.value, self.dt, self.tile = value, dt, tile


def _operand(v: ir.Value) -> _Const | None:
    """Returns `v` as a constant, or None if it isn't one or its value is ambiguous.

    A 16-bit float constant is emitted as a rounded `float` literal, so a value that
    `float32` can't hold exactly would round twice on the GPU and once in the
    interpreter. Such constants stay unfolded.
    """
    value, dt = const_value(v), _dtype(v)
    if value is None or dt is None:
        return None
    if dt in (core.float16, core.bfloat16) and not (
        math.isnan(value) or float(np.float32(value)) == value
    ):
        return None
    from enceladus.interpreter import interp as I  # noqa: N812

    try:
        tile = I.ITile(I.literal_array(value, dt), dt)
    except CompilationError:
        return None
    return _Const(_py(tile), dt, tile)


def _py(tile: Any) -> bool | int | float:
    """Returns the value of a 0-d interpreter tile as a Python number."""
    dt = tile.dtype
    if dt.is_bool():
        return bool(tile.data)
    if dt.is_int():
        return int(tile.data)
    return float(tile.data)


def _int_min(dt: core.dtype) -> int | None:
    return -(1 << (dt.primitive_bitwidth - 1)) if dt.is_signed() else None


def _all_ones(dt: core.dtype) -> int:
    return -1 if dt.is_signed() else (1 << dt.primitive_bitwidth) - 1


def _undefined(kind: str, a: _Const, b: _Const) -> bool:
    """Returns whether integer op `kind` on `a` and `b` has no defined result."""
    dt = a.dt
    if not dt.is_int() or dt.is_bool():
        return False
    if kind in ("floordiv", "mod"):
        return b.value == 0 or (a.value == _int_min(dt) and b.value == -1)
    if kind in ("shl", "shr"):
        return not 0 <= b.value < dt.primitive_bitwidth
    return False


def _fold_binary(op: ir.Op, a: _Const, b: _Const) -> Any:
    from enceladus.interpreter import interp as I  # noqa: N812

    kind = op.attrs["op"] if op.name == "binary" else op.attrs["pred"]
    if _undefined(kind, a, b):
        return None
    if kind in ("min", "max") and a.dt.is_floating() and a.value == 0 and b.value == 0:
        return None  # which zero `fmin` returns for 0.0 and -0.0 is unspecified
    out = I.binary(kind, a.tile, b.tile)
    return _py(out) if out.dtype is _dtype(op.result) else None


def _fold_unary(op: ir.Op, a: _Const) -> Any:
    from enceladus.interpreter import interp as I  # noqa: N812

    kind = op.attrs["op"]
    if kind not in ("neg", "not", "abs", "floor", "ceil"):
        return None  # math functions round differently on the GPU
    if kind in ("neg", "abs") and a.dt.is_int() and a.value == _int_min(a.dt):
        return None
    out = I.unary(kind, a.tile)
    return _py(out) if out.dtype is _dtype(op.result) else None


def _fold_cast(op: ir.Op, a: _Const) -> Any:
    from enceladus.interpreter import interp as I  # noqa: N812

    dst = _dtype(op.result)
    if op.name == "bitcast" or dst is None:
        return None
    if a.dt.is_floating() and dst.is_int() and not dst.is_bool():
        lo, hi = (-(1 << (dst.primitive_bitwidth - 1)), (1 << (dst.primitive_bitwidth - 1)) - 1) \
            if dst.is_signed() else (0, (1 << dst.primitive_bitwidth) - 1)  # fmt: skip
        if not (math.isfinite(a.value) and lo <= math.trunc(a.value) <= hi):
            return None
    return _py(I.convert(a.tile, dst))


def _lossless(src: core.dtype, dst: core.dtype) -> bool:
    """Returns whether converting `src` to `dst` keeps every value exactly."""
    if src.is_floating():
        return dst is core.float32 or dst is src
    if src.is_bool():
        return True
    if dst.is_floating():
        return dst is core.float32 and src.primitive_bitwidth <= 16
    if dst.is_bool():
        return False
    sb, db = src.primitive_bitwidth, dst.primitive_bitwidth
    if src.is_signed() == dst.is_signed():
        return db >= sb
    return dst.is_signed() and db > sb


def _int_value(v: ir.Value) -> int | None:
    """Returns the value of an integer constant, wrapped to its dtype's range."""
    c, dt = const_value(v), _dtype(v)
    if c is None or dt is None or not dt.is_int():
        return None
    if dt.is_bool():
        return int(bool(c))
    bits = dt.primitive_bitwidth
    c = int(c) & ((1 << bits) - 1)
    return c - (1 << bits) if dt.is_signed() and c >> (bits - 1) else c


def _is_neg_zero(v: Any) -> bool:
    return isinstance(v, float) and v == 0 and math.copysign(1.0, v) < 0


def _is_pos_zero(v: Any) -> bool:
    return isinstance(v, float) and v == 0 and math.copysign(1.0, v) > 0


def _binary_identity(op: ir.Op) -> ir.Value | bool | int | float | None:
    """Returns the value or constant that `op` always computes, from an identity."""
    kind = op.attrs["op"]
    x, y = op.operands
    dt = _dtype(op.result)
    cx, cy = const_value(x), const_value(y)
    if dt.is_floating():
        if kind == "mul" and cy == 1:
            return x
        if kind == "mul" and cx == 1:
            return y
        if kind == "div" and cy == 1:
            return x
        if kind == "add" and _is_neg_zero(cy):
            return x
        if kind == "add" and _is_neg_zero(cx):
            return y
        if kind == "sub" and _is_pos_zero(cy):
            return x
        if kind in ("min", "max") and x is y:
            return x
        return None
    ones = _all_ones(dt) if not dt.is_bool() else 1
    cx, cy = _int_value(x), _int_value(y)
    if x is y:
        if kind in ("and", "or", "min", "max"):
            return x
        if kind in ("sub", "xor"):
            return False if dt.is_bool() else 0
    # Identities with the constant on either side.
    for a, c in ((x, cy), (y, cx)):
        if c is None:
            continue
        if kind in ("add", "or", "xor") and c == 0:
            return a
        if kind == "mul" and c == 1:
            return a
        if kind in ("mul", "and") and c == 0:
            return False if dt.is_bool() else 0
        if kind == "and" and c == ones:
            return a
        if kind == "or" and c == ones:
            return True if dt.is_bool() else ones
    if cy is not None:
        if kind in ("sub", "shl", "shr") and cy == 0:
            return x
        if kind == "floordiv" and cy == 1:
            return x
        if kind == "mod" and cy == 1:
            return 0
    return None


def _identity(op: ir.Op) -> ir.Value | bool | int | float | None:
    """Returns what an op that isn't all constant always computes, or None."""
    name = op.name
    if name == "binary":
        return _binary_identity(op)
    if name == "select":
        c, x, y = op.operands
        cv = const_value(c)
        if cv is not None:
            return x if cv else y
        return x if x is y else None
    if name == "cmp":
        x, y = op.operands
        if x is y and _dtype(x).is_int():
            return op.attrs["pred"] in ("eq", "le", "ge")
        return None
    if name == "unary" and op.attrs["op"] in ("neg", "not"):
        inner = op.operands[0].defining_op
        if inner is not None and inner.name == "unary" and inner.attrs["op"] == op.attrs["op"]:
            return inner.operands[0]
        return None
    if name == "cast":
        inner = op.operands[0].defining_op
        if inner is not None and inner.name == "cast":
            src = inner.operands[0]
            if src.type == op.result.type and _lossless(_dtype(src), _dtype(inner.result)):
                return src
        return None
    if name == "addptr":
        return op.operands[0] if const_value(op.operands[1]) == 0 else None
    return None


def _fold(op: ir.Op) -> ir.Value | bool | int | float | None:
    """Returns the value or constant that replaces `op`'s result, or None to keep it."""
    name = op.name
    if name not in ("binary", "cmp", "unary", "cast", "bitcast", "select", "addptr"):
        return None
    if len(op.results) != 1 or _dtype(op.result) is None and name != "addptr":
        return None
    consts = [_operand(v) for v in op.operands]
    with np.errstate(all="ignore"):
        if all(c is not None for c in consts):
            if name in ("binary", "cmp"):
                out = _fold_binary(op, *consts)
            elif name == "unary":
                out = _fold_unary(op, *consts)
            elif name in ("cast", "bitcast"):
                out = _fold_cast(op, *consts)
            else:
                out = None
            if out is not None:
                return out
        return _identity(op)


def _materialize(value: bool | int | float, t: ir.Type, loc) -> ir.Op:
    """Returns a new op that produces `value` with type `t`."""
    if isinstance(t, ir.TileType):
        return ir.Op("full", [], [t], {"value": value}, loc=loc)
    return ir.Op("const", [], [t], {"value": value}, loc=loc)


class _Inline:
    """Maps the results of a pruned `if` to the operands of its taken branch's `yield`."""

    def __init__(self, op: ir.Op, yield_op: ir.Op) -> None:
        self.op, self.yield_op = op, yield_op


def fold(module: ir.Module) -> int:
    """Folds constants, applies algebraic identities, and prunes constant `if`s.

    Returns:
        The number of ops rewritten.
    """
    replace: dict[int, ir.Value] = {}
    changed = 0

    def remap(v: ir.Value) -> ir.Value:
        return replace.get(id(v), v)

    def run(block: ir.Block) -> None:
        nonlocal changed
        kept: list[ir.Op] = []
        queue: deque[ir.Op | _Inline] = deque(block.ops)
        while queue:
            item = queue.popleft()
            if isinstance(item, _Inline):
                for r, v in zip(item.op.results, item.yield_op.operands, strict=True):
                    replace[id(r)] = remap(v)
                continue
            op = item
            op.operands = [remap(v) for v in op.operands]
            if op.name == "if" and const_value(op.operands[0]) is not None:
                taken = op.regions[0 if const_value(op.operands[0]) else 1].block.ops
                queue.extendleft(reversed([*taken[:-1], _Inline(op, taken[-1])]))
                changed += 1
                continue
            for r in op.regions:
                run(r.block)
            out = _fold(op)
            if out is None:
                kept.append(op)
                continue
            changed += 1
            if not isinstance(out, ir.Value):
                new = _materialize(out, op.result.type, op.loc)
                kept.append(new)
                out = new.result
            if out.name_hint is None:
                out.name_hint = op.result.name_hint
            replace[id(op.result)] = out
        for op in kept:
            op.parent = block
        block.ops = kept

    run(module.body)
    return changed


# ---------------------------------------------------------------------------
# CSE and DCE
# ---------------------------------------------------------------------------


def cse(module: ir.Module) -> int:
    """Merges identical pure ops. Returns the number of ops removed."""
    replace: dict[int, ir.Value] = {}
    removed = 0

    def run(block: ir.Block, scope: list[dict]) -> None:
        nonlocal removed
        table: dict = {}
        scope.append(table)
        kept = []
        for op in block.ops:
            op.operands = [replace.get(id(v), v) for v in op.operands]
            for r in op.regions:
                run(r.block, scope)
            if op.name in PURE_OPS and not op.regions:
                key = (
                    op.name,
                    tuple(id(v) for v in op.operands),
                    _freeze(op.attrs),
                    tuple(str(r.type) for r in op.results),
                )
                prev = next((t[key] for t in reversed(scope) if key in t), None)
                if prev is not None:
                    for a, b in zip(op.results, prev.results, strict=True):
                        replace[id(a)] = b
                        if b.name_hint is None:
                            b.name_hint = a.name_hint
                    removed += 1
                    continue
                table[key] = op
            kept.append(op)
        block.ops = kept
        scope.pop()

    run(module.body, [])
    return removed


def dce(module: ir.Module) -> int:
    """Deletes removable ops whose results are unused. Returns the number removed."""
    total = 0
    while True:
        used: set[int] = set()
        for op in module.walk():
            used.update(id(v) for v in op.operands)
        removed = 0

        def run(block: ir.Block, used: set[int] = used) -> None:
            nonlocal removed
            kept = []
            for op in block.ops:
                for r in op.regions:
                    run(r.block)
                if op.name in REMOVABLE_OPS and not any(id(v) in used for v in op.results):
                    removed += 1
                    continue
                kept.append(op)
            block.ops = kept

        run(module.body)
        total += removed
        if not removed:
            return total


def simplify(module: ir.Module) -> None:
    """Folds constants and identities, then runs CSE and DCE."""
    fold(module)
    cse(module)
    dce(module)
