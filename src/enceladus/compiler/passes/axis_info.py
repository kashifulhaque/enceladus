"""AxisInfo: contiguity, divisibility, and constancy of integer and pointer values.

For each dimension `d` of an integer or pointer tile, the analysis tracks three facts,
with the same meaning as Triton's `AxisInfo`:

- *contiguity* `c`: along `d`, every aligned block of `c` elements (the elements at
  indices `[k*c, (k+1)*c)`) holds consecutive values. Pointers step by one element.
- *divisibility*: the largest power of two that divides the first value of every such
  block. Pointer divisibility counts bytes, so `addptr` scales offsets by the element
  size. With a contiguity of 1, it divides every value.
- *constancy* `k`: along `d`, every aligned block of `k` elements holds equal values.

Scalars keep their divisibility in `scalar_div`. Every fact is a power of two, and
every fact is a lower bound: a smaller value is always correct. Codegen relies on the
facts for vector loads and stores, so a rule that can't prove a fact must return a
smaller one.

The facts come from `arange` (contiguous), `splat` and constants (constant),
specialization facts on arguments (a divisibility of 16), and the `tl.multiple_of` and
`tl.max_contiguous` hints. They propagate through arithmetic, shape ops, comparisons, and
loops. Loop-carried values reach a fixed point over the loop body.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

from enceladus.compiler import ir

MAX_DIV = 1 << 30
# A loop body is analyzed at most this many times before its carried values fall back to
# no facts. Each pass can only lower facts, so real loops settle in two or three passes.
MAX_LOOP_PASSES = 16


@dataclass(frozen=True)
class AxisInfo:
    contiguity: tuple[int, ...]
    divisibility: tuple[int, ...]
    constancy: tuple[int, ...]
    const_value: int | None = None  # set when every element is this integer
    scalar_div: int = 1  # divisibility of a rank-0 value

    @property
    def rank(self) -> int:
        return len(self.contiguity)


def _unknown(shape: tuple[int, ...]) -> AxisInfo:
    n = len(shape)
    return AxisInfo((1,) * n, (1,) * n, (1,) * n)


def _pow2_div(v: int) -> int:
    """Returns the largest power of two that divides `v`, capped at `MAX_DIV`."""
    if v == 0:
        return MAX_DIV
    return min(v & -v, MAX_DIV)


def _uniform(shape: tuple[int, ...], div: int, value: int | None) -> AxisInfo:
    return AxisInfo((1,) * len(shape), (div,) * len(shape), tuple(shape), value, div)


def elem_step(t: ir.Type) -> int:
    """Returns the step between consecutive values of `t` in divisibility units.

    Integer values step by 1. Pointer values step by the element size, because their
    divisibility counts bytes while their contiguity counts elements.
    """
    e = ir.elem_of(t)
    if isinstance(e, ir.PointerType):
        return max(1, e.elem.dtype.itemsize)
    return 1


def div_at(info: AxisInfo, d: int, c: int, step: int = 1) -> int:
    """Returns the divisibility of the first value of every aligned block of `c` along `d`.

    When `c` is at least the contiguity, every such block starts a contiguous block, so
    the divisibility holds as is. Otherwise a block can start inside a contiguous block,
    `m * c` elements after its start, which only keeps a factor of `c * step`.
    """
    if info.contiguity[d] <= c:
        return info.divisibility[d]
    return min(info.divisibility[d], c * step)


def every_div(info: AxisInfo, step: int = 1) -> int:
    """Returns a power of two that divides every value of `info`'s tile."""
    if info.const_value is not None:
        return _pow2_div(info.const_value)
    if not info.rank:
        return info.scalar_div
    return min(div_at(info, d, 1, step) for d in range(info.rank))


def meet(a: AxisInfo, b: AxisInfo, step: int = 1) -> AxisInfo:
    """Returns facts that hold for a value that is either `a` or `b`."""
    cv = a.const_value if a.const_value == b.const_value else None
    if not a.rank:
        return AxisInfo((), (), (), cv, gcd(a.scalar_div, b.scalar_div))
    contig = tuple(min(x, y) for x, y in zip(a.contiguity, b.contiguity, strict=True))
    div = tuple(min(div_at(a, d, c, step), div_at(b, d, c, step)) for d, c in enumerate(contig))
    const = tuple(min(x, y) for x, y in zip(a.constancy, b.constancy, strict=True))
    return AxisInfo(contig, div, const, cv)


# Integer casts that keep every value, so they keep every fact.
def _value_preserving_cast(src: ir.Type, dst: ir.Type) -> bool:
    s, t = ir.elem_of(src), ir.elem_of(dst)
    if not (isinstance(s, ir.ScalarType) and isinstance(t, ir.ScalarType)):
        return False
    if s.name == "i1" or t.name == "i1" or s.name[0] not in "iu" or t.name[0] not in "iu":
        return False
    sb, tb = s.dtype.primitive_bitwidth, t.dtype.primitive_bitwidth
    if s.name[0] == t.name[0]:
        return tb >= sb
    return s.name[0] == "u" and tb > sb  # unsigned to a wider signed type


class AxisAnalysis:
    """Computes `AxisInfo` for every integer and pointer value of a module."""

    def __init__(self, module: ir.Module) -> None:
        self.info: dict[int, AxisInfo] = {}
        func = module.func
        arg_attrs = func.attrs.get("arg_attrs", [{}] * len(module.body.args))
        for v, attrs in zip(module.body.args, arg_attrs, strict=False):
            div = attrs.get("divisibility", 1)
            val = 1 if attrs.get("equal_to_1") else None
            self.info[id(v)] = _uniform((), div, val)
        self._block(module.body)

    def get(self, v: ir.Value) -> AxisInfo:
        info = self.info.get(id(v))
        return info if info is not None else _unknown(ir.shape_of(v.type))

    def _block(self, block: ir.Block) -> None:
        for op in block.ops:
            if op.name == "for":
                self._for(op)
                continue
            for r in op.regions:
                self._block(r.block)
            if len(op.results) == 1:
                self.info[id(op.result)] = self._op(op)

    def _for(self, op: ir.Op) -> None:
        """Analyzes a loop until the facts of its carried values stop changing."""
        body = op.regions[0].block
        lb, _, step = (self.get(v) for v in op.operands[:3])
        # The counter takes the values lb + i * step.
        self.info[id(body.args[0])] = _uniform((), gcd(every_div(lb), every_div(step)), None)
        args, inits = body.args[1:], op.operands[3:]
        steps = [elem_step(a.type) for a in args]
        cur = [self.get(v) for v in inits]
        for _ in range(MAX_LOOP_PASSES):
            for a, c in zip(args, cur, strict=True):
                self.info[id(a)] = c
            self._block(body)
            yielded = body.ops[-1].operands
            new = [meet(c, self.get(y), s) for c, y, s in zip(cur, yielded, steps, strict=True)]
            if new == cur:
                break
            cur = new
        else:
            cur = [_unknown(ir.shape_of(a.type)) for a in args]
            for a, c in zip(args, cur, strict=True):
                self.info[id(a)] = c
            self._block(body)
        for r, c in zip(op.results, cur, strict=True):
            self.info[id(r)] = c

    def _op(self, op: ir.Op) -> AxisInfo:
        shape = ir.shape_of(op.result.type)
        n = len(shape)
        name = op.name
        if name == "const":
            v = op.attrs["value"]
            if isinstance(v, bool) or not isinstance(v, int):
                return _uniform(shape, 1, None) if n else _unknown(shape)
            return _uniform(shape, _pow2_div(v), v)
        if name == "full":
            v = op.attrs["value"]
            if isinstance(v, int) and not isinstance(v, bool):
                return _uniform(shape, _pow2_div(v), v)
            return AxisInfo((1,) * n, (1,) * n, tuple(shape))
        if name == "splat":
            s = self.get(op.operands[0])
            return _uniform(shape, s.scalar_div, s.const_value) if not s.rank else _unknown(shape)
        if name == "arange":
            start = op.attrs["start"]
            return AxisInfo((shape[0],), (_pow2_div(start),), (1,), None)
        if name == "hint":
            return self._hint(op)
        if name == "expand_dims":
            a = self.get(op.operands[0])
            ax = op.attrs["axis"]
            # Along the new size-1 dimension every value starts a block, so the
            # divisibility must divide every value.
            div = every_div(a, elem_step(op.result.type))
            return AxisInfo(
                a.contiguity[:ax] + (1,) + a.contiguity[ax:],
                a.divisibility[:ax] + (div,) + a.divisibility[ax:],
                a.constancy[:ax] + (1,) + a.constancy[ax:],
                a.const_value,
            )
        if name == "broadcast":
            a = self.get(op.operands[0])
            src = ir.shape_of(op.operands[0].type)
            if a.rank != n:
                return _unknown(shape)
            return AxisInfo(
                tuple(1 if s == 1 else c for s, c in zip(src, a.contiguity, strict=True)),
                a.divisibility,
                tuple(d if s == 1 else c for s, d, c in zip(src, shape, a.constancy, strict=True)),
                a.const_value,
            )
        if name == "binary" and op.attrs["op"] in ("add", "sub", "mul") or name == "addptr":
            a, b = (self.get(v) for v in op.operands)
            if a.rank != n or b.rank != n:
                return self._elementwise(op, shape)
            step = elem_step(op.result.type)
            if name == "addptr":
                # Offsets count elements; pointer divisibility counts bytes.
                b = AxisInfo(b.contiguity, tuple(min(d * step, MAX_DIV) for d in b.divisibility),
                             b.constancy, None, min(b.scalar_div * step, MAX_DIV))  # fmt: skip
                return self._arith("add", a, b, n, step)
            return self._arith(op.attrs["op"], a, b, n, step)
        if name == "cmp":
            return self._cmp(op, shape)
        if name == "cast" and _value_preserving_cast(op.operands[0].type, op.result.type):
            return self.get(op.operands[0])
        if name == "load" and n:
            # Equal addresses load equal values.
            k = self._elementwise(op, shape)
            return AxisInfo((1,) * n, (1,) * n, k.constancy)
        if name in ("binary", "unary", "select", "cast", "bitcast", "fma"):
            return self._elementwise(op, shape)
        return _unknown(shape)

    def _elementwise(self, op: ir.Op, shape: tuple[int, ...]) -> AxisInfo:
        """Returns the facts of an elementwise op that computes nothing else: constancy.

        Wherever every tile operand is constant, so is the result.
        """
        n = len(shape)
        const = tuple(shape)
        for v in op.operands:
            if isinstance(v.type, ir.TileType):
                i = self.get(v)
                if i.rank != n:
                    return _unknown(shape)
                const = tuple(gcd(x, y) for x, y in zip(const, i.constancy, strict=True))
        return AxisInfo((1,) * n, (1,) * n, const if n else ())

    def _cmp(self, op: ir.Op, shape: tuple[int, ...]) -> AxisInfo:
        """Returns the facts of a comparison.

        Beyond the constancy of its operands, a comparison of a contiguous `x` with a
        constant `r` is constant on aligned blocks of `g`, where `g` divides the first
        value of each block and `r`: in a block `s, s + 1, ..., s + g - 1`, `s + t < r`
        exactly when `s < r`. That holds for `x < r`, `x >= r`, `r > x`, and `r <= x`,
        but not for the other predicates: `s + t <= s` holds only for `t = 0`.
        """
        base = self._elementwise(op, shape)
        n = len(shape)
        a, b = (self.get(v) for v in op.operands)
        if not n or a.rank != n or b.rank != n or elem_step(op.operands[0].type) != 1:
            return base
        pred = op.attrs["pred"]
        const = list(base.constancy)
        pairs = []
        if pred in ("lt", "ge"):
            pairs.append((a, b))  # x < r, x >= r
        if pred in ("gt", "le"):
            pairs.append((b, a))  # r > x, r <= x
        for x, r in pairs:
            for d in range(n):
                g = min(x.contiguity[d], x.divisibility[d], r.constancy[d], div_at(r, d, 1))
                const[d] = max(const[d], g)
        return AxisInfo(base.contiguity, base.divisibility, tuple(const))

    def _hint(self, op: ir.Op) -> AxisInfo:
        """Applies a `tl.multiple_of` or `tl.max_contiguous` promise to its operand's facts."""
        a = self.get(op.operands[0])
        vals = tuple(op.attrs["values"])
        kind = op.attrs["kind"]
        shape = ir.shape_of(op.result.type)
        if not shape:
            if kind == "multiple_of":
                return AxisInfo((), (), (), a.const_value, max(a.scalar_div, _pow2_div(vals[0])))
            return a
        if kind == "multiple_of":
            div = tuple(max(x, _pow2_div(v)) for x, v in zip(a.divisibility, vals, strict=True))
            return AxisInfo(a.contiguity, div, a.constancy, a.const_value)
        contig = tuple(max(c, min(_pow2_div(v), s))
                       for c, v, s in zip(a.contiguity, vals, shape, strict=True))  # fmt: skip
        return AxisInfo(contig, a.divisibility, a.constancy, a.const_value)

    @staticmethod
    def _arith(kind: str, a: AxisInfo, b: AxisInfo, n: int, step: int = 1) -> AxisInfo:
        cv = None
        if a.const_value is not None and b.const_value is not None:
            cv = {"add": a.const_value + b.const_value, "sub": a.const_value - b.const_value,
                  "mul": a.const_value * b.const_value}[kind]  # fmt: skip
        if n == 0:
            sd = min(a.scalar_div * b.scalar_div, MAX_DIV) if kind == "mul" else \
                gcd(a.scalar_div, b.scalar_div)
            return AxisInfo((), (), (), cv, _pow2_div(cv) if cv is not None else sd)
        const = tuple(gcd(x, y) for x, y in zip(a.constancy, b.constancy, strict=True))
        if kind == "mul":
            if b.const_value == 1:
                return AxisInfo(a.contiguity, a.divisibility, a.constancy, cv)
            if a.const_value == 1:
                return AxisInfo(b.contiguity, b.divisibility, b.constancy, cv)
            # The product isn't contiguous, so its divisibility must divide every value.
            div = tuple(min(div_at(a, d, 1) * div_at(b, d, 1), MAX_DIV) for d in range(n))
            if cv is not None:
                div = (_pow2_div(cv),) * n
            return AxisInfo((1,) * n, div, const, cv)
        if kind == "sub":
            contig = tuple(gcd(ca, kb) for ca, kb in zip(a.contiguity, b.constancy, strict=True))
        else:
            contig = tuple(
                max(gcd(ca, kb), gcd(cb, ka))
                for ca, cb, ka, kb in zip(a.contiguity, b.contiguity, a.constancy, b.constancy,
                                          strict=True)  # fmt: skip
            )
        div = tuple(gcd(div_at(a, d, c, step), div_at(b, d, c, step)) for d, c in enumerate(contig))
        if cv is not None:
            div = (_pow2_div(cv),) * n
        return AxisInfo(contig, div, const, cv)


def contiguous_order(info: AxisInfo) -> tuple[int, ...]:
    """Returns dimensions from most to least contiguous, breaking ties toward the last dim."""
    dims = range(info.rank)
    return tuple(sorted(dims, key=lambda d: (-info.contiguity[d], -d)))


def vector_width(ptr: AxisInfo, d: int, elem_bytes: int, mask: AxisInfo | None = None) -> int:
    """Returns how many elements along `d` one vector access of a pointer tile can cover.

    A group of `w` elements that starts at an index that's a multiple of `w` along `d`
    is one vector access when its addresses are consecutive (a contiguity of at least
    `w`), its first address is aligned to `w * elem_bytes` bytes, and, for a masked
    access, its mask is constant (a mask constancy of at least `w`). The result is the
    largest power of two that meets all three, or 1.
    """
    w = min(ptr.contiguity[d], max(1, ptr.divisibility[d] // elem_bytes))
    if mask is not None:
        w = min(w, mask.constancy[d])
    return max(1, w)
