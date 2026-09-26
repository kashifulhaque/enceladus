"""Scan lowering: `tl.cumsum` and `tl.associative_scan` over any tile layout.

The bits of the scan axis coordinate come from register, lane, and SIMD-group bases in
any interleaving. For example, a 1D blocked tile of 1,024 elements over 4 SIMD groups
has the coordinate `r_lo + 4 * lane + 128 * warp + 512 * r_hi`. The lowering walks the
axis bits from least to most significant, in *levels*: maximal runs of bits of one kind.

After the levels below bit `a` run, every element holds the inclusive scan of its
aligned block of `2^a` elements, so the block's last element holds the block total. A
level combines those totals:

1. Registers: each thread fetches the totals it needs and combines them in sequence.
2. Lanes: a Hillis-Steele scan with `simd_shuffle` over the lanes of the level, or
   `simd_prefix_exclusive_sum` for a sum over all 32 lanes.
3. SIMD groups: the block totals go through threadgroup memory, and each thread
   combines the totals of the SIMD groups before it.

Each element then combines the prefix of earlier blocks with its own value, as
`combine(prefix, value)`, so the combine function needs to be associative but not
commutative. A reverse scan runs the same steps with the digits of every level
reversed. `float16` and `bfloat16` sums accumulate in `float32`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from enceladus.compiler import ir
from enceladus.compiler import layout as L
from enceladus.compiler.codegen.msl import BARRIER, CTYPES, Tile, _add, _body_ops, is_half

if TYPE_CHECKING:
    from enceladus.compiler.codegen.msl import _Codegen

SIMD_PREFIX_TYPES = {"float", "half", "int", "uint", "short", "ushort", "char", "uchar"}

SHFL_HELPER = """\
// simd_shuffle from a lane index for every register type. bool, bfloat, and 64-bit
// integers aren't in the native type set, so they shuffle through bit-compatible types.
template <typename T> static inline T tg_shfl_idx(T x, ushort l) { return simd_shuffle(x, l); }
static inline bool tg_shfl_idx(bool x, ushort l) { return simd_shuffle(ushort(x), l) != 0; }
static inline bfloat tg_shfl_idx(bfloat x, ushort l) {
  return as_type<bfloat>(simd_shuffle(as_type<ushort>(x), l));
}
static inline long tg_shfl_idx(long x, ushort l) {
  return as_type<long>(simd_shuffle(as_type<uint2>(x), l));
}
static inline ulong tg_shfl_idx(ulong x, ushort l) {
  return as_type<ulong>(simd_shuffle(as_type<uint2>(x), l));
}"""

_SIZES = {"char": 1, "uchar": 1, "bool": 1, "short": 2, "ushort": 2, "half": 2, "bfloat": 2,
          "long": 8, "ulong": 8}  # fmt: skip


def shfl(cg: _Codegen, x: str, lane: str) -> str:
    """Returns an expression for `x` read from lane `lane` of the SIMD group."""
    cg.helper("tg_shfl_idx", SHFL_HELPER)
    return f"tg_shfl_idx({x}, ushort({lane}))"


def _gather(idx: str, pos: list[int]) -> str:
    """Returns a `uint` expression that packs bits `pos` of `idx` into a digit."""
    if pos == list(range(pos[0], pos[0] + len(pos))):
        m = (1 << len(pos)) - 1
        return f"(({idx} >> {pos[0]}) & {m}u)" if pos[0] else f"({idx} & {m}u)"
    return "(" + " | ".join(f"((({idx} >> {p}) & 1u) << {i})" for i, p in enumerate(pos)) + ")"


def _scatter(x: str, pos: list[int]) -> str:
    """Returns a `uint` expression that spreads the bits of digit `x` onto bits `pos`."""
    if pos == list(range(pos[0], pos[0] + len(pos))):
        return f"(({x}) << {pos[0]})" if pos[0] else f"({x})"
    return "(" + " | ".join(f"((({x}) >> {i} & 1u) << {p})" for i, p in enumerate(pos)) + ")"


def _bits(pos: list[int]) -> int:
    return sum(1 << p for p in pos)


def _spread(d: int, pos: list[int]) -> int:
    """Spreads the bits of the compile-time digit `d` onto bits `pos`."""
    return sum(1 << p for i, p in enumerate(pos) if d >> i & 1)


@dataclass
class _Level:
    kind: str  # "reg", "lane", or "warp"
    pos: list[int]  # hardware bit indices, from the least significant axis bit up


def _levels(lay: L.BitLayout, axis: int) -> list[_Level]:
    n = lay.shape[axis].bit_length() - 1
    owner: list[tuple[str, int] | None] = [None] * n
    for kind, bases in (("reg", lay.reg), ("lane", lay.lane), ("warp", lay.warp)):
        for i, b in enumerate(bases):
            if b[axis]:
                owner[b[axis].bit_length() - 1] = (kind, i)
    levels: list[_Level] = []
    for o in owner:
        assert o is not None
        if levels and levels[-1].kind == o[0]:
            levels[-1].pos.append(o[1])
        else:
            levels.append(_Level(o[0], [o[1]]))
    return levels


def _combiner(cg: _Codegen, region: ir.Block | None, acc_types: list[str]):
    """Returns `comb(dst, lhs, rhs)`, which emits `dst = combine(lhs, rhs)`.

    `dst` may alias `lhs` or `rhs`; both sides are read before `dst` is written.
    """

    def comb(dst: list[str], lhs: list[str], rhs: list[str]) -> None:
        if region is None:
            cg.e.line(f"{dst[0]} = {lhs[0]} + {rhs[0]};")
            return
        cg.e.line("{")
        with cg.e.indented():
            args = []
            for side in (lhs, rhs):
                for t, x in zip(acc_types, side, strict=True):
                    c = cg.fresh("ca")
                    cg.e.line(f"const {t} {c} = {x};")
                    args.append(c)
            for arg, x in zip(region.args, args, strict=True):
                cg.sv[id(arg)] = x
            cg.push_scope()
            cg.block(_body_ops(region))
            ys = [cg.s(y) for y in region.ops[-1].operands]
            cg.pop_scope()
            for d, y in zip(dst, ys, strict=True):
                cg.e.line(f"{d} = {y};")
        cg.e.line("}")

    return comb


class _Scan:
    def __init__(self, cg: _Codegen, lay: L.BitLayout, axis: int, rev: bool, work: list[str],
                 acc_types: list[str], region: ir.Block | None) -> None:  # fmt: skip
        self.cg, self.lay, self.axis, self.rev = cg, lay, axis, rev
        self.work, self.types = work, acc_types
        self.is_sum = region is None
        self.comb = _combiner(cg, region, acc_types)
        self.low_reg = 0  # register bits of the axis bits already scanned
        self.low_lane = 0
        self.low_warp = 0
        self.low_bits = 0

    # ---- helpers ----

    @property
    def hold(self) -> int:
        """The register bits of the element that holds each block's total."""
        return 0 if self.rev else self.low_reg

    def regs(self, r: int) -> list[str]:
        return [f"{w}[{r}]" for w in self.work]

    def locals(self, hint: str, exprs: list[str], const: bool = True) -> list[str]:
        out = []
        for t, x in zip(self.types, exprs, strict=True):
            n = self.cg.fresh(hint)
            self.cg.e.line(f"{'const ' if const else ''}{t} {n} = {x};")
            out.append(n)
        return out

    def block_regs(self, key: int) -> list[int]:
        """Returns the registers that `key` selects, over every lower register bit."""
        return [key | s for s in range(self.lay.num_regs) if s & ~self.low_reg == 0]

    def apply(self, key: int, prefix: list[str]) -> None:
        for r in self.block_regs(key):
            self.comb(self.regs(r), prefix, self.regs(r))

    def fetch(self, r: int) -> list[str]:
        """Copies the values of holder register `r` from the thread that holds them.

        Valid only when the scanned bits hold no SIMD-group bits.
        """
        if not self.low_lane:
            return self.locals("st", self.regs(r))
        pat = 0 if self.rev else self.low_lane
        src = self.cg.pro(f"scanhold|{self.low_lane}|{pat}", "sh", "uint",
                          f"((lane & ~{self.low_lane}u) | {pat}u)")  # fmt: skip
        return self.locals("st", [shfl(self.cg, x, src) for x in self.regs(r)])

    def exchange(self, holders: list[int]):
        """Writes the block totals of holder registers to threadgroup memory.

        Returns `(buffers, bid, stride)`: `bid(r)` is the buffer index of the block that
        register `r` belongs to, and `stride` is the index step between neighboring
        blocks along the axis.
        """
        cg, lay, axis, a = self.cg, self.lay, self.axis, self.low_bits
        shape = list(lay.shape)
        shape[axis] >>= a
        strides, acc = [0] * len(shape), 1
        for d in reversed(range(len(shape))):
            strides[d] = acc
            acc *= shape[d]
        parts = []
        for d, s in enumerate(strides):
            tt = cg.thread_coord(lay, d)
            if tt == "0":
                continue
            if d == axis:
                tt = f"({tt} >> {a})"
            parts.append(tt if s == 1 else f"{tt} * {s}")
        expr = " + ".join(parts)
        base = cg.pro(f"scanbid|{lay}|{axis}|{a}", "sb", "int", expr) if parts else "0"

        def bid(r: int) -> str:
            c = L.coords_of(lay, r)
            k = sum((x >> a if d == axis else x) * s
                    for d, (x, s) in enumerate(zip(c, strides, strict=True)))  # fmt: skip
            return _add(base, k)

        offsets, total = [], 0
        for t in self.types:
            offsets.append(total)
            total += -(-(acc * _SIZES.get(t, 4)) // 16) * 16
        cg.use_tg(total)
        conds = []
        for idx, m in (("lane", self.low_lane), ("warp", self.low_warp)):
            if m:
                conds.append(f"({idx} & {m}u) == {0 if self.rev else m}u")
        cg.e.line(BARRIER)
        bufs = []
        for t, off in zip(self.types, offsets, strict=True):
            b = cg.fresh("sbuf")
            cg.e.line(f"threadgroup {t}* {b} = (threadgroup {t}*)(tg_mem + {off});")
            bufs.append(b)
        with cg.e.block(f"if ({' && '.join(conds)})" if conds else ""):
            for r in holders:
                for b, x in zip(bufs, self.regs(r), strict=True):
                    cg.e.line(f"{b}[{bid(r)}] = {x};")
        cg.e.line(BARRIER)
        return bufs, bid, strides[axis]

    # ---- levels ----

    def run(self, lvl: _Level) -> None:
        if lvl.kind == "reg":
            self.reg_level(lvl)
        elif lvl.kind == "lane" and not self.low_warp:
            self.lane_level(lvl)
        else:
            self.tg_level(lvl)
        bits = _bits(lvl.pos)
        if lvl.kind == "reg":
            self.low_reg |= bits
        elif lvl.kind == "lane":
            self.low_lane |= bits
        else:
            self.low_warp |= bits
        self.low_bits += len(lvl.pos)

    def reg_level(self, lvl: _Level) -> None:
        pos, w = lvl.pos, len(lvl.pos)
        top = (1 << w) - 1
        mine = _bits(pos)
        keys = [r for r in range(self.lay.num_regs) if r & (mine | self.low_reg) == 0]

        def raw(e: int) -> int:
            return top - e if self.rev else e

        read = None
        if self.low_warp:
            holders = [k | _spread(d, pos) | self.hold for k in keys for d in range(top + 1)]
            bufs, bid, _ = self.exchange(holders)

            def read(r: int) -> list[str]:
                return self.locals("st", [f"{b}[{bid(r)}]" for b in bufs])

        for key in keys:
            totals = []
            for e in range(top):  # the last digit's total isn't needed
                hr = key | _spread(raw(e), pos) | self.hold
                totals.append(read(hr) if read is not None else self.fetch(hr))
            prefix = self.locals("sp", totals[0], const=False)
            for e in range(1, top + 1):
                self.apply(key | _spread(raw(e), pos), prefix)
                if e < top:
                    self.comb(prefix, prefix, totals[e])

    def lane_level(self, lvl: _Level) -> None:
        cg, pos, w = self.cg, lvl.pos, len(lvl.pos)
        top = (1 << w) - 1
        keys = [r for r in range(self.lay.num_regs) if r & self.low_reg == 0]
        if (self.is_sum and not self.rev and pos == [0, 1, 2, 3, 4] and not self.low_lane
                and self.types[0] in SIMD_PREFIX_TYPES):  # fmt: skip
            for key in keys:
                x = self.regs(key | self.hold)[0]
                p = self.locals("sp", [f"simd_prefix_exclusive_sum({x})"])
                self.apply(key, p)
            return
        mask = _bits(pos)
        dg = cg.pro(f"scandg|lane|{pos}", "dg", "uint", _gather("lane", pos))
        de = cg.pro(f"scande|lane|{pos}", "de", "uint", f"({top}u ^ {dg})") if self.rev else dg

        def lane_of(x: str) -> str:
            d = f"({top}u ^ ({x}))" if self.rev else x
            return f"((lane & ~{mask}u) | {_scatter(d, pos)})"

        incl = [self.locals("si", self.fetch(key | self.hold), const=False) for key in keys]
        src1 = None
        for i in range(w):
            s = 1 << i
            src = cg.pro(f"scansrc|{pos}|{self.rev}|{s}", "ss", "uint",
                         f"({de} >= {s}u ? {lane_of(f'{de} - {s}u')} : lane)")  # fmt: skip
            src1 = src1 or src
            for acc in incl:
                sh = self.locals("sx", [shfl(cg, x, src) for x in acc])
                with cg.e.block(f"if ({de} >= {s}u)"):
                    self.comb(acc, sh, acc)
        for key, acc in zip(keys, incl, strict=True):
            prefix = self.locals("sp", [shfl(cg, x, src1) for x in acc])
            with cg.e.block(f"if ({de} >= 1u)"):
                self.apply(key, prefix)

    def tg_level(self, lvl: _Level) -> None:
        cg, pos, w = self.cg, lvl.pos, len(lvl.pos)
        top = (1 << w) - 1
        idx = lvl.kind
        keys = [r for r in range(self.lay.num_regs) if r & self.low_reg == 0]
        bufs, bid, stride = self.exchange([k | self.hold for k in keys])
        dg = cg.pro(f"scandg|{idx}|{pos}", "dg", "uint", _gather(idx, pos))
        de = cg.pro(f"scande|{idx}|{pos}", "de", "uint", f"({top}u ^ {dg})") if self.rev else dg

        def at(key: int, j: str) -> list[str]:
            d = f"({top}u - {j})" if self.rev else j
            off = f"(int({d}) - int({dg}))"
            return [f"{b}[{bid(key)} + {off} * {stride}]" for b in bufs]

        with cg.e.block(f"if ({de} > 0u)"):
            for key in keys:
                prefix = self.locals("sp", at(key, "0u"), const=False)
                j = cg.fresh("sj")
                with cg.e.block(f"for (uint {j} = 1u; {j} < {de}; ++{j})"):
                    self.comb(prefix, prefix, at(key, j))
                self.apply(key, prefix)


def emit_scan(cg: _Codegen, op: ir.Op) -> None:
    axis = op.attrs["axis"]
    rev = bool(op.attrs.get("reverse", False))
    region = op.regions[0].block if op.regions else None
    lay = cg.plan.layout_of(op.results[0])
    tiles = [cg.mat(v, lay) for v in op.operands]
    elems = [ir.elem_of(v.type) for v in op.operands]
    if region is None:
        acc_ir = [ir.f32 if is_half(elems[0]) else elems[0]]
    else:
        acc_ir = elems
    types = [CTYPES[t.name] for t in acc_ir]
    work = []
    for t, ct, tile in zip(acc_ir, types, tiles, strict=True):
        name = cg.declare(ir.TileType(lay.shape, t), lay, "scan")
        cg.loop(lay.num_regs, f"{name}[{{r}}] = {ct}({tile.get('{r}')});")
        work.append(name)
    s = _Scan(cg, lay, axis, rev, work, types, region)
    for lvl in _levels(lay, axis):
        s.run(lvl)
    for res, name, ct in zip(op.results, work, types, strict=True):
        out = CTYPES[ir.elem_of(res.type).name]
        if out == ct:
            cg.tiles[id(res)] = Tile(lay, name)
            continue
        arr = cg.declare(res.type, lay, res.name_hint or "scan")
        cg.loop(lay.num_regs, f"{arr}[{{r}}] = {out}({name}[{{r}}]);")
        cg.tiles[id(res)] = Tile(lay, arr)
