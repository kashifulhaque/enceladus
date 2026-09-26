"""AxisInfo: contiguity, divisibility, and constancy of integer and pointer values.

For each dimension of an integer or pointer tile, the analysis tracks:

- *contiguity*: the length of runs of consecutive values (pointers step by one element);
- *divisibility*: the largest power of two dividing the first value of each run;
- *constancy*: the length of runs of equal values.

Scalars keep their divisibility in `scalar_div`. Pointer divisibility counts bytes, as
in Triton, so `addptr` scales integer offsets by the element size.

The analysis follows Triton's `AxisInfo`, simplified.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

from tegula.compiler import ir

MAX_DIV = 1 << 30


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
    if v == 0:
        return MAX_DIV
    return min(v & -v, MAX_DIV)


def _uniform(shape: tuple[int, ...], div: int, value: int | None) -> AxisInfo:
    return AxisInfo((1,) * len(shape), (div,) * len(shape), tuple(shape), value, div)


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
                body = op.regions[0].block
                # Loop-carried values keep their initial contiguity but lose divisibility.
                self.info[id(body.args[0])] = _uniform((), 1, None)
                for arg, init in zip(body.args[1:], op.operands[3:], strict=True):
                    i = self.get(init)
                    n = i.rank
                    self.info[id(arg)] = AxisInfo(i.contiguity, (1,) * n, (1,) * n)
            for r in op.regions:
                self._block(r.block)
            if len(op.results) == 1:
                self.info[id(op.result)] = self._op(op)

    def _op(self, op: ir.Op) -> AxisInfo:
        shape = ir.shape_of(op.result.type)
        n = len(shape)
        name = op.name
        if name == "const":
            v = op.attrs["value"]
            if isinstance(v, bool) or not isinstance(v, int):
                return _unknown(shape)
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
        if name in ("program_id", "num_programs"):
            return _unknown(shape)
        if name == "expand_dims":
            a = self.get(op.operands[0])
            ax = op.attrs["axis"]
            return AxisInfo(
                a.contiguity[:ax] + (1,) + a.contiguity[ax:],
                a.divisibility[:ax] + (MAX_DIV,) + a.divisibility[ax:],
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
                return _unknown(shape)
            if name == "addptr":
                # Offsets count elements; pointer divisibility counts bytes.
                size = ir.elem_of(op.result.type).elem.dtype.itemsize
                b = AxisInfo(b.contiguity, tuple(min(d * size, MAX_DIV) for d in b.divisibility),
                             b.constancy, None, min(b.scalar_div * size, MAX_DIV))  # fmt: skip
                return self._arith("add", a, b, n)
            return self._arith(op.attrs["op"], a, b, n)
        return _unknown(shape)

    @staticmethod
    def _arith(kind: str, a: AxisInfo, b: AxisInfo, n: int) -> AxisInfo:
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
            div = tuple(min(x * y, MAX_DIV) for x, y in zip(a.divisibility, b.divisibility,
                                                            strict=True))  # fmt: skip
            return AxisInfo((1,) * n, div, const, cv)
        contig = tuple(
            max(gcd(ca, kb), gcd(cb, ka))
            for ca, cb, ka, kb in zip(a.contiguity, b.contiguity, a.constancy, b.constancy,
                                      strict=True)  # fmt: skip
        )
        if kind == "sub":
            contig = tuple(gcd(ca, kb) for ca, kb in zip(a.contiguity, b.constancy, strict=True))
        div = tuple(gcd(x, y) for x, y in zip(a.divisibility, b.divisibility, strict=True))
        return AxisInfo(contig, div, const, cv)


def contiguous_order(info: AxisInfo) -> tuple[int, ...]:
    """Returns dimensions from most to least contiguous, breaking ties toward the last dim."""
    dims = range(info.rank)
    return tuple(sorted(dims, key=lambda d: (-info.contiguity[d], -d)))
