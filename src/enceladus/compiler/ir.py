"""Enceladus IR: types, values, ops, blocks, a printer, and a verifier.

The IR is a small SSA form modeled on MLIR. A `Module` holds one `func` op whose region
holds the kernel body. Each op has a name, operands, results, attributes, regions, and a
source location. Ops that carry regions (`for`, `if`, `reduce`, `scan`) end each region
with a `yield` terminator. The function body ends with `return`.

The op set and the operand rules for each op are listed in `VERIFIERS`.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from enceladus.compiler.errors import INTERNAL_ERROR_HINT, CompilationError, Loc
from enceladus.language import core

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

MAX_TILE_DIM = 1 << 16


class Type:
    """Base class of IR types."""


@dataclass(frozen=True)
class ScalarType(Type):
    """A scalar type: `i1`, `i8`-`i64`, `u8`-`u64`, `f16`, `bf16`, or `f32`."""

    name: str

    @property
    def dtype(self) -> core.dtype:
        return core.dtype_from_name(self.name)

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class PointerType(Type):
    """A pointer to global memory with element type `elem`."""

    elem: ScalarType

    def __str__(self) -> str:
        return f"ptr<{self.elem}>"


@dataclass(frozen=True)
class TileType(Type):
    """A statically shaped tile. `layout` is `None` until the layout pass assigns one."""

    shape: tuple[int, ...]
    elem: ScalarType | PointerType
    layout: Any = None

    def __str__(self) -> str:
        dims = "".join(f"{d}x" for d in self.shape)
        lay = "" if self.layout is None else f", {self.layout}"
        return f"tile<{dims}{self.elem}{lay}>"

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class DescType(Type):
    """A tensor descriptor: base pointer, shape, and strides, loaded in `block_shape` blocks."""

    elem: ScalarType
    shape_rank: int
    block_shape: tuple[int, ...]

    def __str__(self) -> str:
        dims = "".join(f"{d}x" for d in self.block_shape)
        return f"desc<{dims}{self.elem}>"


_SCALARS: dict[str, ScalarType] = {d.ir_name: ScalarType(d.ir_name) for d in core.ALL_DTYPES}
i1, i8, i16, i32, i64 = (_SCALARS[n] for n in ("i1", "i8", "i16", "i32", "i64"))
u8, u16, u32, u64 = (_SCALARS[n] for n in ("u8", "u16", "u32", "u64"))
f16, bf16, f32 = _SCALARS["f16"], _SCALARS["bf16"], _SCALARS["f32"]


def scalar(dt: core.dtype) -> ScalarType:
    """Returns the IR scalar type for a `tl` dtype."""
    return _SCALARS[dt.ir_name]


def shape_of(t: Type) -> tuple[int, ...]:
    """Returns the tile shape of `t`, or `()` for a scalar or pointer."""
    return t.shape if isinstance(t, TileType) else ()


def elem_of(t: Type) -> Type:
    """Returns the element type of a tile, or `t` itself for scalars and pointers."""
    return t.elem if isinstance(t, TileType) else t


def with_elem(t: Type, elem: ScalarType | PointerType) -> Type:
    """Returns a type with the shape of `t` and element type `elem`."""
    return TileType(t.shape, elem) if isinstance(t, TileType) else elem


def with_shape(shape: Sequence[int], elem: ScalarType | PointerType) -> Type:
    """Returns `elem` for an empty shape, or a tile type otherwise."""
    return TileType(tuple(shape), elem) if len(shape) else elem


def is_pow2_dim(d: int) -> bool:
    return 1 <= d <= MAX_TILE_DIM and (d & (d - 1)) == 0


# ---------------------------------------------------------------------------
# Values, ops, blocks
# ---------------------------------------------------------------------------


class Value:
    """An SSA value, defined by exactly one op result or block argument."""

    __slots__ = ("type", "name_hint", "owner", "index")

    def __init__(self, type: Type, name_hint: str | None = None, owner: Any = None, index=0):
        self.type = type
        self.name_hint = name_hint
        self.owner = owner  # Op or Block
        self.index = index

    @property
    def defining_op(self) -> Op | None:
        return self.owner if isinstance(self.owner, Op) else None

    def __repr__(self) -> str:
        return f"<Value {self.name_hint or '?'}: {self.type}>"


class Block:
    """A list of ops with block arguments."""

    def __init__(self, arg_types: Sequence[Type] = (), arg_names: Sequence[str | None] = ()):
        names = list(arg_names) + [None] * (len(arg_types) - len(arg_names))
        self.args: list[Value] = [
            Value(t, n, self, i) for i, (t, n) in enumerate(zip(arg_types, names, strict=True))
        ]
        self.ops: list[Op] = []
        self.parent: Region | None = None

    def append(self, op: Op) -> Op:
        op.parent = self
        self.ops.append(op)
        return op


class Region:
    """A region holding a single block."""

    def __init__(self, block: Block | None = None):
        self.block = block if block is not None else Block()
        self.block.parent = self
        self.parent: Op | None = None


class Op:
    """An operation.

    Attributes:
        name: The op name, for example `"binary"` or `"for"`.
        operands: The input values.
        results: The values this op defines.
        attrs: Compile-time attributes, for example `{"op": "add"}`.
        regions: Nested regions, for ops such as `for`, `if`, and `reduce`.
        loc: The source location that produced this op.
    """

    def __init__(
        self,
        name: str,
        operands: Sequence[Value] = (),
        result_types: Sequence[Type] = (),
        attrs: dict[str, Any] | None = None,
        regions: Sequence[Region] = (),
        loc: Loc | None = None,
    ):
        self.name = name
        self.operands: list[Value] = list(operands)
        self.results: list[Value] = [Value(t, None, self, i) for i, t in enumerate(result_types)]
        self.attrs: dict[str, Any] = dict(attrs or {})
        self.regions: list[Region] = list(regions)
        for r in self.regions:
            r.parent = self
        self.loc = loc
        self.parent: Block | None = None

    @property
    def result(self) -> Value:
        if len(self.results) != 1:
            raise ValueError(f"op {self.name} has {len(self.results)} results, not 1")
        return self.results[0]

    def walk(self) -> Iterator[Op]:
        """Yields this op and every op nested in its regions, in program order."""
        yield self
        for r in self.regions:
            for op in r.block.ops:
                yield from op.walk()

    def __repr__(self) -> str:
        return f"<Op {self.name} at {self.loc}>"


class Module:
    """A compiled kernel: one `func` op plus module attributes.

    Attributes:
        func: The `func` op. Its attributes hold `sym_name`, `arg_names`, and `arg_attrs`
            (one dict of specialization facts per runtime argument).
        attrs: Module attributes, such as `num_warps` and `constexprs`.
    """

    def __init__(self, func: Op, attrs: dict[str, Any] | None = None):
        self.func = func
        self.attrs = dict(attrs or {})

    @property
    def name(self) -> str:
        return self.func.attrs["sym_name"]

    @property
    def body(self) -> Block:
        return self.func.regions[0].block

    def walk(self) -> Iterator[Op]:
        """Yields every op in the kernel body, in program order."""
        for op in self.body.ops:
            yield from op.walk()

    def format(self, locs: bool = False) -> str:
        """Returns the textual IR. With `locs=True`, each op shows its source location."""
        return _Printer(locs).module(self)

    def __str__(self) -> str:
        return self.format()


class Builder:
    """Creates ops at an insertion point and stamps them with the current location."""

    def __init__(self) -> None:
        self.block: Block | None = None
        self.loc: Loc | None = None

    def create(
        self,
        name: str,
        operands: Sequence[Value] = (),
        result_types: Sequence[Type] = (),
        attrs: dict[str, Any] | None = None,
        regions: Sequence[Region] = (),
    ) -> Op:
        op = Op(name, operands, result_types, attrs, regions, self.loc)
        assert self.block is not None, "builder has no insertion block"
        return self.block.append(op)

    @contextmanager
    def at(self, block: Block) -> Iterator[Block]:
        """Temporarily moves the insertion point to the end of `block`."""
        saved = self.block
        self.block = block
        try:
            yield block
        finally:
            self.block = saved


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------

_IDENT = re.compile(r"[^A-Za-z0-9_]")


def format_attr(v: Any) -> str:
    """Formats an attribute value deterministically."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(format_attr(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k} = {format_attr(v[k])}" for k in sorted(v)) + "}"
    if v is None:
        return "none"
    return str(v)


class _Printer:
    def __init__(self, locs: bool) -> None:
        self.locs = locs
        self.names: dict[int, str] = {}
        self.used: set[str] = set()
        self.counter = 0
        self.lines: list[str] = []

    def name(self, v: Value) -> str:
        n = self.names.get(id(v))
        if n is None:
            if v.name_hint:
                base = _IDENT.sub("_", v.name_hint)
                n, k = base, 0
                while n in self.used:
                    k += 1
                    n = f"{base}_{k}"
            else:
                n = str(self.counter)
                self.counter += 1
            self.used.add(n)
            self.names[id(v)] = n
        return "%" + n

    def module(self, m: Module) -> str:
        attrs = format_attr(m.attrs)
        self.lines.append(f"module @{m.name} attributes {attrs} {{")
        f = m.func
        args = []
        arg_attrs = f.attrs.get("arg_attrs", [{}] * len(m.body.args))
        for v, a in zip(m.body.args, arg_attrs, strict=True):
            s = f"{self.name(v)}: {v.type}"
            if a:
                s += " " + format_attr(a)
            args.append(s)
        self.lines.append(f"  func @{m.name}({', '.join(args)}) {{")
        for op in m.body.ops:
            self.op(op, 2)
        self.lines.append("  }")
        self.lines.append("}")
        return "\n".join(self.lines) + "\n"

    def op(self, op: Op, depth: int) -> None:
        ind = "  " * depth
        lhs = ", ".join(self.name(r) for r in op.results)
        lhs = f"{lhs} = " if lhs else ""
        ops = ", ".join(self.name(v) for v in op.operands)
        attrs = f" {format_attr(op.attrs)}" if op.attrs else ""
        sig_in = ", ".join(str(v.type) for v in op.operands)
        sig_out = ", ".join(str(r.type) for r in op.results)
        if len(op.results) != 1:
            sig_out = f"({sig_out})"
        loc = f"  loc({op.loc})" if self.locs and op.loc is not None else ""
        if not op.regions:
            self.lines.append(f'{ind}{lhs}"{op.name}"({ops}){attrs} : ({sig_in}) -> {sig_out}{loc}')
            return
        self.lines.append(f'{ind}{lhs}"{op.name}"({ops}) ({{')
        for i, r in enumerate(op.regions):
            if i:
                self.lines.append(f"{ind}}}, {{")
            b = r.block
            bargs = ", ".join(f"{self.name(a)}: {a.type}" for a in b.args)
            self.lines.append(f"{ind}^bb0({bargs}):")
            for inner in b.ops:
                self.op(inner, depth + 1)
        self.lines.append(f"{ind}}}){attrs} : ({sig_in}) -> {sig_out}{loc}")


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

BINARY_OPS = frozenset(
    ["add", "sub", "mul", "div", "floordiv", "mod", "and", "or", "xor", "shl", "shr", "min", "max"]
)
INT_ONLY_BINARY = frozenset(["floordiv", "and", "or", "xor", "shl", "shr"])
CMP_PREDS = frozenset(["eq", "ne", "lt", "le", "gt", "ge"])
MATH_UNARY = frozenset(
    ["exp", "exp2", "log", "log2", "sqrt", "rsqrt", "sin", "cos", "tanh", "sigmoid", "erf",
     "floor", "ceil"]
)  # fmt: skip
UNARY_OPS = MATH_UNARY | {"neg", "not", "abs"}
REDUCE_KINDS = frozenset(["sum", "max", "min", "argmax", "argmin"])
SCAN_KINDS = frozenset(["sum", "max", "min"])
ATOMIC_OPS = frozenset(["add", "max", "min", "xchg", "and", "or", "xor"])
TERMINATORS = frozenset(["yield", "return"])


class _VerifyError(Exception):
    pass


def _fail(msg: str) -> None:
    raise _VerifyError(msg)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise _VerifyError(msg)


def _arity(op: Op, operands: int | tuple[int, ...] | None, results: int) -> None:
    if operands is not None:
        allowed = (operands,) if isinstance(operands, int) else operands
        _check(len(op.operands) in allowed, f"expects {operands} operands, got {len(op.operands)}")
    _check(len(op.results) == results, f"expects {results} results, got {len(op.results)}")


def _is_num(t: Type) -> bool:
    return isinstance(elem_of(t), ScalarType)


def _is_float(t: Type) -> bool:
    e = elem_of(t)
    return isinstance(e, ScalarType) and e.dtype.is_floating()


def _is_int(t: Type) -> bool:
    e = elem_of(t)
    return isinstance(e, ScalarType) and e.dtype.is_int()


def _is_ptr(t: Type) -> bool:
    return isinstance(elem_of(t), PointerType)


def _is_int_scalar(t: Type) -> bool:
    return isinstance(t, ScalarType) and t.dtype.is_int()


def _yield_types(region: Region) -> list[Type]:
    ops = region.block.ops
    _check(bool(ops) and ops[-1].name == "yield", "region must end with `yield`")
    return [v.type for v in ops[-1].operands]


def _v_const(op: Op) -> None:
    _arity(op, 0, 1)
    t = op.result.type
    _check(isinstance(t, ScalarType), "result must be a scalar")
    v = op.attrs.get("value")
    _check(isinstance(v, (bool, int, float)), "needs a numeric `value` attribute")
    if t.dtype.is_int():
        _check(isinstance(v, (bool, int)), f"integer constant has non-integer value {v!r}")


def _v_splat(op: Op) -> None:
    _arity(op, 1, 1)
    src, t = op.operands[0].type, op.result.type
    _check(isinstance(src, (ScalarType, PointerType)), "operand must be a scalar or pointer")
    _check(isinstance(t, TileType) and t.elem == src, "result must be a tile of the operand type")


def _v_arange(op: Op) -> None:
    _arity(op, 0, 1)
    s, e = op.attrs.get("start"), op.attrs.get("end")
    _check(isinstance(s, int) and isinstance(e, int) and e > s, "needs int `start` < `end`")
    _check(op.result.type == TileType((e - s,), i32), f"result must be tile<{e - s}xi32>")


def _v_full(op: Op) -> None:
    _arity(op, 0, 1)
    t = op.result.type
    _check(isinstance(t, TileType) and isinstance(t.elem, ScalarType), "result must be a tile")
    _check(isinstance(op.attrs.get("value"), (bool, int, float)), "needs a numeric `value`")


def _v_program_id(op: Op) -> None:
    _arity(op, 0, 1)
    _check(op.attrs.get("axis") in (0, 1, 2), "`axis` must be 0, 1, or 2")
    _check(op.result.type == i32, "result must be i32")


def _v_binary(op: Op) -> None:
    _arity(op, 2, 1)
    kind = op.attrs.get("op")
    _check(kind in BINARY_OPS, f"unknown binary op {kind!r}")
    a, b, r = op.operands[0].type, op.operands[1].type, op.result.type
    _check(a == b == r, f"operand and result types must match, got {a}, {b} -> {r}")
    _check(_is_num(a), "operands must be numeric")
    if kind in INT_ONLY_BINARY:
        _check(_is_int(a), f"`{kind}` needs integer operands")
    if kind == "div":
        _check(_is_float(a), "`div` needs floating-point operands")


def _v_cmp(op: Op) -> None:
    _arity(op, 2, 1)
    _check(op.attrs.get("pred") in CMP_PREDS, f"unknown predicate {op.attrs.get('pred')!r}")
    a, b, r = op.operands[0].type, op.operands[1].type, op.result.type
    _check(a == b, f"operand types must match, got {a} and {b}")
    _check(r == with_elem(a, i1), f"result must be {with_elem(a, i1)}")


def _v_unary(op: Op) -> None:
    _arity(op, 1, 1)
    kind = op.attrs.get("op")
    _check(kind in UNARY_OPS, f"unknown unary op {kind!r}")
    a, r = op.operands[0].type, op.result.type
    _check(a == r and _is_num(a), "operand and result must have the same numeric type")
    if kind in MATH_UNARY:
        _check(_is_float(a), f"`{kind}` needs a floating-point operand")
    if kind == "not":
        _check(_is_int(a), "`not` needs an integer operand")


def _v_fma(op: Op) -> None:
    _arity(op, 3, 1)
    ts = {v.type for v in op.operands} | {op.result.type}
    _check(len(ts) == 1 and _is_float(op.result.type), "operands and result need one float type")


def _v_select(op: Op) -> None:
    _arity(op, 3, 1)
    c, a, b = (v.type for v in op.operands)
    _check(a == b == op.result.type, "value operands and result must have the same type")
    _check(c == with_elem(a, i1), f"condition must be {with_elem(a, i1)}")


def _v_cast(op: Op) -> None:
    _arity(op, 1, 1)
    a, r = op.operands[0].type, op.result.type
    _check(_is_num(a) and _is_num(r), "cast operands must be numeric")
    _check(shape_of(a) == shape_of(r), "cast must preserve the shape")


def _v_bitcast(op: Op) -> None:
    _v_cast(op)
    a, r = elem_of(op.operands[0].type), elem_of(op.result.type)
    _check(a.dtype.primitive_bitwidth == r.dtype.primitive_bitwidth, "bitcast must keep bitwidth")


def _v_broadcast(op: Op) -> None:
    _arity(op, 1, 1)
    a, r = op.operands[0].type, op.result.type
    _check(isinstance(a, TileType) and isinstance(r, TileType), "operand and result must be tiles")
    _check(a.elem == r.elem and len(a.shape) == len(r.shape), "rank and element type must match")
    for i, (x, y) in enumerate(zip(a.shape, r.shape, strict=True)):
        _check(x == y or x == 1, f"dimension {i} can't broadcast from {x} to {y}")


def _v_expand_dims(op: Op) -> None:
    _arity(op, 1, 1)
    a, r = op.operands[0].type, op.result.type
    ax = op.attrs.get("axis")
    s = list(shape_of(a))
    _check(isinstance(ax, int) and 0 <= ax <= len(s), "`axis` is out of range")
    s.insert(ax, 1)
    _check(r == TileType(tuple(s), elem_of(a)), f"result must be {TileType(tuple(s), elem_of(a))}")


def _v_reshape(op: Op) -> None:
    _arity(op, 1, 1)
    a, r = op.operands[0].type, op.result.type
    _check(isinstance(a, TileType) and isinstance(r, TileType), "operand and result must be tiles")
    _check(a.elem == r.elem and a.numel == r.numel, "reshape must keep element type and count")


def _v_trans(op: Op) -> None:
    _arity(op, 1, 1)
    a, r = op.operands[0].type, op.result.type
    perm = op.attrs.get("perm")
    _check(isinstance(a, TileType), "operand must be a tile")
    _check(sorted(perm or ()) == list(range(len(a.shape))), f"`perm` {perm} isn't a permutation")
    _check(r == TileType(tuple(a.shape[p] for p in perm), a.elem), "result shape must be permuted")


def _v_addptr(op: Op) -> None:
    _arity(op, 2, 1)
    p, o = op.operands[0].type, op.operands[1].type
    _check(_is_ptr(p), "operand 0 must be a pointer")
    _check(_is_int(o) and shape_of(o) == shape_of(p), "offset must be an integer of the same shape")
    _check(op.result.type == p, "result type must match the pointer operand")


def _v_load(op: Op) -> None:
    _arity(op, (1, 3), 1)
    p = op.operands[0].type
    _check(_is_ptr(p), "operand 0 must be a pointer")
    val = with_elem(p, elem_of(p).elem)
    _check(op.result.type == val, f"result must be {val}")
    if len(op.operands) == 3:
        _check(op.operands[1].type == with_elem(p, i1), "mask must be i1 with the pointer shape")
        _check(op.operands[2].type == val, f"`other` must be {val}")


def _v_store(op: Op) -> None:
    _arity(op, (2, 3), 0)
    p = op.operands[0].type
    _check(_is_ptr(p), "operand 0 must be a pointer")
    val = with_elem(p, elem_of(p).elem)
    _check(op.operands[1].type == val, f"stored value must be {val}")
    if len(op.operands) == 3:
        _check(op.operands[2].type == with_elem(p, i1), "mask must be i1 with the pointer shape")


def _v_make_desc(op: Op) -> None:
    _check(len(op.results) == 1, "expects 1 result")
    t = op.result.type
    _check(isinstance(t, DescType), "result must be a descriptor")
    _check(len(op.operands) == 1 + 2 * t.shape_rank, "expects base, shape, and strides operands")
    base = op.operands[0].type
    _check(isinstance(base, PointerType) and base.elem == t.elem, "base must be ptr<elem>")
    _check(all(_is_int_scalar(v.type) for v in op.operands[1:]), "shape and strides must be ints")
    _check(len(t.block_shape) == t.shape_rank, "block rank must match the shape rank")
    _check(all(is_pow2_dim(d) for d in t.block_shape), "block dims must be powers of two")


def _v_desc_load(op: Op) -> None:
    _arity(op, None, 1)
    d = op.operands[0].type
    _check(isinstance(d, DescType), "operand 0 must be a descriptor")
    _check(len(op.operands) == 1 + d.shape_rank, "expects one offset per dimension")
    _check(all(_is_int_scalar(v.type) for v in op.operands[1:]), "offsets must be int scalars")
    _check(op.result.type == TileType(d.block_shape, d.elem), "result must be the block tile")


def _v_desc_store(op: Op) -> None:
    _arity(op, None, 0)
    d = op.operands[0].type
    _check(isinstance(d, DescType), "operand 0 must be a descriptor")
    _check(len(op.operands) == 2 + d.shape_rank, "expects offsets and a value")
    _check(all(_is_int_scalar(v.type) for v in op.operands[1:-1]), "offsets must be int scalars")
    _check(op.operands[-1].type == TileType(d.block_shape, d.elem), "value must be the block tile")


def _v_atomic_rmw(op: Op) -> None:
    _arity(op, (2, 3), 1)
    _check(op.attrs.get("op") in ATOMIC_OPS, f"unknown atomic op {op.attrs.get('op')!r}")
    p = op.operands[0].type
    _check(_is_ptr(p), "operand 0 must be a pointer")
    val = with_elem(p, elem_of(p).elem)
    _check(op.operands[1].type == val and op.result.type == val, f"value and result must be {val}")
    if len(op.operands) == 3:
        _check(op.operands[2].type == with_elem(p, i1), "mask must be i1 with the pointer shape")


def _v_atomic_cas(op: Op) -> None:
    _arity(op, 3, 1)
    p = op.operands[0].type
    _check(_is_ptr(p), "operand 0 must be a pointer")
    val = with_elem(p, elem_of(p).elem)
    _check(all(v.type == val for v in op.operands[1:]), f"compare and value must be {val}")
    _check(op.result.type == val, f"result must be {val}")


def _v_dot(op: Op) -> None:
    _arity(op, 3, 1)
    a, b, c = (v.type for v in op.operands)
    _check(all(isinstance(t, TileType) and len(t.shape) == 2 for t in (a, b, c)), "needs 2D tiles")
    _check(a.shape[1] == b.shape[0], f"inner dimensions differ: {a.shape} and {b.shape}")
    _check(c.shape == (a.shape[0], b.shape[1]), f"accumulator must be {a.shape[0]}x{b.shape[1]}")
    _check(a.elem == b.elem and _is_float(a), "operands must share a floating-point type")
    _check(_is_float(c) and op.result.type == c, "result must match the float accumulator")


def _reduced(t: Type, axis: int) -> tuple[int, ...]:
    s = list(shape_of(t))
    del s[axis]
    return tuple(s)


def _v_reduce(op: Op, scan: bool = False) -> None:
    n = len(op.operands)
    _check(n >= 1 and len(op.results) == n, "expects matching operand and result counts")
    ins = [v.type for v in op.operands]
    _check(all(isinstance(t, TileType) for t in ins), "operands must be tiles")
    _check(len({t.shape for t in ins}) == 1, "operands must have the same shape")
    axis = op.attrs.get("axis")
    _check(isinstance(axis, int) and 0 <= axis < len(ins[0].shape), "`axis` is out of range")
    kind = op.attrs.get("kind")
    kinds = SCAN_KINDS if scan else REDUCE_KINDS
    _check((kind is None) == (len(op.regions) == 1), "needs exactly one of `kind` or a region")
    if kind is not None:
        _check(kind in kinds and n == 1, f"unknown kind {kind!r} or too many operands")
        elem = i32 if kind in ("argmax", "argmin") else ins[0].elem
        outs = [ins[0] if scan else with_shape(_reduced(ins[0], axis), elem)]
    else:
        elems = [t.elem for t in ins]
        blk = op.regions[0].block
        _check([a.type for a in blk.args] == elems + elems, "combine region args must be 2x elems")
        _check(_yield_types(op.regions[0]) == elems, "combine region must yield the element types")
        outs = ins if scan else [with_shape(_reduced(t, axis), t.elem) for t in ins]
    _check([r.type for r in op.results] == outs, f"results must be {[str(t) for t in outs]}")


def _v_for(op: Op) -> None:
    _check(len(op.operands) >= 3 and len(op.regions) == 1, "needs bounds, step, and a body")
    lb, ub, st = (v.type for v in op.operands[:3])
    _check(lb == ub == st and _is_int_scalar(lb), "bounds and step must share an int type")
    inits = [v.type for v in op.operands[3:]]
    blk = op.regions[0].block
    _check([a.type for a in blk.args] == [lb, *inits], "body args must be counter and iter args")
    _check([r.type for r in op.results] == inits, "results must match the iteration arguments")
    _check(_yield_types(op.regions[0]) == inits, "body must yield the iteration argument types")


def _v_if(op: Op) -> None:
    _check(len(op.operands) == 1 and len(op.regions) == 2, "needs a condition and two regions")
    _check(op.operands[0].type == i1, "condition must be an i1 scalar")
    outs = [r.type for r in op.results]
    for r in op.regions:
        _check(not r.block.args, "branches take no arguments")
        _check(_yield_types(r) == outs, "each branch must yield the result types")


def _v_while(op: Op) -> None:
    _check(len(op.regions) == 2, "needs a condition region and a body region")


def _v_terminator(op: Op) -> None:
    _check(not op.results, "terminators have no results")
    if op.name == "return":
        _check(not op.operands, "kernels return no values")


def _v_no_results(op: Op) -> None:
    _check(not op.results, "has no results")


def _v_convert_layout(op: Op) -> None:
    _arity(op, 1, 1)
    a, r = op.operands[0].type, op.result.type
    _check(isinstance(a, TileType) and isinstance(r, TileType), "operand and result must be tiles")
    _check(a.shape == r.shape and a.elem == r.elem, "must keep the shape and element type")


def _v_local_alloc(op: Op) -> None:
    _arity(op, (0, 1), 1)


def _v_local_store(op: Op) -> None:
    _arity(op, 2, 0)


def _v_local_load(op: Op) -> None:
    _arity(op, 1, 1)


def _v_barrier(op: Op) -> None:
    _arity(op, 0, 0)


def _v_hint(op: Op) -> None:
    _arity(op, 1, 1)
    t = op.operands[0].type
    _check(_is_ptr(t) or _is_int(t), "operand must be an integer or a pointer")
    _check(op.result.type == t, "result type must match the operand")
    _check(op.attrs.get("kind") in ("multiple_of", "max_contiguous"), "unknown hint `kind`")
    vals = op.attrs.get("values")
    _check(isinstance(vals, tuple) and len(vals) == max(1, len(shape_of(t)))
           and all(isinstance(v, int) and v > 0 for v in vals),
           "`values` must hold one positive int per dimension")  # fmt: skip


VERIFIERS: dict[str, Callable[[Op], None]] = {
    "const": _v_const,
    "splat": _v_splat,
    "arange": _v_arange,
    "full": _v_full,
    "program_id": _v_program_id,
    "num_programs": _v_program_id,
    "binary": _v_binary,
    "cmp": _v_cmp,
    "unary": _v_unary,
    "fma": _v_fma,
    "select": _v_select,
    "cast": _v_cast,
    "bitcast": _v_bitcast,
    "broadcast": _v_broadcast,
    "expand_dims": _v_expand_dims,
    "reshape": _v_reshape,
    "trans": _v_trans,
    "addptr": _v_addptr,
    "load": _v_load,
    "store": _v_store,
    "make_desc": _v_make_desc,
    "desc_load": _v_desc_load,
    "desc_store": _v_desc_store,
    "atomic_rmw": _v_atomic_rmw,
    "atomic_cas": _v_atomic_cas,
    "dot": _v_dot,
    "reduce": _v_reduce,
    "scan": lambda op: _v_reduce(op, scan=True),
    "for": _v_for,
    "if": _v_if,
    "while": _v_while,
    "yield": _v_terminator,
    "return": _v_terminator,
    "print": _v_no_results,
    "assert": _v_no_results,
    "convert_layout": _v_convert_layout,
    "local_alloc": _v_local_alloc,
    "local_store": _v_local_store,
    "local_load": _v_local_load,
    "barrier": _v_barrier,
    "hint": _v_hint,
}
"""The verifier rule for every op in the IR."""


def verify_enabled() -> bool:
    """Returns whether `ENCELADUS_VERIFY` asks for verification after every pass."""
    return os.environ.get("ENCELADUS_VERIFY", "0") not in ("", "0")


def verify(module: Module) -> None:
    """Checks the module's structure and types.

    The verifier checks operand and result types and counts for every op, that each value
    is defined once, that each use is dominated by its definition, and that each region
    ends with the right terminator.

    Raises:
        CompilationError: The IR is malformed. The error carries the op's location.
    """
    _Verifier().run(module)


class _Verifier:
    def __init__(self) -> None:
        self.defined: set[int] = set()

    def define(self, v: Value, op: Op | None) -> None:
        if id(v) in self.defined:
            raise self.error(op, f"value {v!r} is defined more than once")
        self.defined.add(id(v))
        t = v.type
        if isinstance(t, TileType):
            for i, d in enumerate(t.shape):
                if not is_pow2_dim(d):
                    raise self.error(op, f"tile dimension {i} is {d}, not a power of two")

    def error(self, op: Op | None, msg: str) -> CompilationError:
        name = f"`{op.name}` " if op is not None else ""
        return CompilationError(
            f"IR verification failed: {name}op: {msg}. {INTERNAL_ERROR_HINT}",
            op.loc if op else None,
        )

    def run(self, module: Module) -> None:
        f = module.func
        if f.name != "func" or len(f.regions) != 1:
            raise self.error(f, "the module must hold a `func` op with one region")
        n = len(module.body.args)
        if len(f.attrs.get("arg_attrs", [])) != n or len(f.attrs.get("arg_names", [])) != n:
            raise self.error(f, "`arg_attrs` and `arg_names` need one entry per argument")
        self.block(module.body, set(), "return", f)

    def block(self, block: Block, visible: set[int], terminator: str, parent: Op) -> None:
        visible = set(visible)
        for a in block.args:
            self.define(a, parent)
            visible.add(id(a))
        if not block.ops or block.ops[-1].name != terminator:
            raise self.error(parent, f"region must end with `{terminator}`")
        last = len(block.ops) - 1
        for i, op in enumerate(block.ops):
            if op.parent is not block:
                raise self.error(op, "op's parent link doesn't match its block")
            for k, v in enumerate(op.operands):
                if id(v) not in visible:
                    raise self.error(op, f"operand {k} ({v!r}) isn't defined before this use")
            if op.name in TERMINATORS and (i != last or op.name != terminator):
                raise self.error(op, f"`{op.name}` can only end a region that expects it")
            rule = VERIFIERS.get(op.name)
            if rule is None:
                raise self.error(op, "unknown op")
            try:
                rule(op)
            except _VerifyError as e:
                raise self.error(op, str(e)) from None
            for r in op.regions:
                self.block(r.block, visible, "yield", op)
            for r in op.results:
                self.define(r, op)
                visible.add(id(r))
