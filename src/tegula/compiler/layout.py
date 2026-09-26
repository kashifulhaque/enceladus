"""Bit-linear layouts: which thread register holds which tile element.

A `BitLayout` maps the bits of a hardware index (register `r`, lane `l`, SIMD group
`w`) to the bits of a logical tile coordinate. Each basis is a coordinate vector with a
single power-of-two entry, or all zeros. The coordinate held by `(r, l, w)` is the sum
(equivalently, the XOR) of the bases for the set bits of `r`, `l`, and `w`.

A zero basis in the lane or SIMD-group bits means *broadcast*: threads that differ
only in that bit hold the same element. All shapes are powers of two.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

LANE_BITS = 5  # 32-wide SIMD groups

Basis = tuple[int, ...]


def _log2(n: int) -> int:
    if n < 1 or n & (n - 1):
        raise ValueError(f"{n} isn't a power of two")
    return n.bit_length() - 1


def _unit(rank: int, dim: int, bit: int) -> Basis:
    v = [0] * rank
    v[dim] = 1 << bit
    return tuple(v)


def _zero(rank: int) -> Basis:
    return (0,) * rank


def _dim_bit(b: Basis) -> tuple[int, int] | None:
    """Returns (dim, bit) of a single-bit basis, or None for a zero basis."""
    for d, v in enumerate(b):
        if v:
            return d, v.bit_length() - 1
    return None


@dataclass(frozen=True)
class BitLayout:
    """A distribution of a tile over the registers, lanes, and SIMD groups of a threadgroup.

    Attributes:
        shape: The tile shape; every dimension is a power of two.
        reg: One basis per register bit. Registers never hold duplicate elements.
        lane: Exactly five bases, one per lane bit.
        warp: One basis per SIMD-group bit, so `num_warps == 2 ** len(warp)`.
    """

    shape: tuple[int, ...]
    reg: tuple[Basis, ...]
    lane: tuple[Basis, ...]
    warp: tuple[Basis, ...]

    def __post_init__(self) -> None:
        rank = len(self.shape)
        for s in self.shape:
            _log2(s)
        if len(self.lane) != LANE_BITS:
            raise ValueError(f"a layout needs exactly {LANE_BITS} lane bases")
        seen: set[tuple[int, int]] = set()
        for kind, bases in (("reg", self.reg), ("lane", self.lane), ("warp", self.warp)):
            for b in bases:
                if len(b) != rank:
                    raise ValueError(f"{kind} basis {b} doesn't match rank {rank}")
                if sum(1 for v in b if v) > 1 or any(v & (v - 1) for v in b):
                    raise ValueError(f"{kind} basis {b} must have at most one power-of-two entry")
                db = _dim_bit(b)
                if db is None:
                    if kind == "reg":
                        raise ValueError("register bases can't be zero")
                    continue
                if (1 << db[1]) >= self.shape[db[0]]:
                    raise ValueError(f"{kind} basis {b} is outside shape {self.shape}")
                if db in seen:
                    raise ValueError(f"basis {b} appears more than once")
                seen.add(db)
        needed = sum(_log2(s) for s in self.shape)
        if len(seen) != needed:
            raise ValueError(f"layout bases don't cover every element of shape {self.shape}")

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def num_regs(self) -> int:
        return 1 << len(self.reg)

    @property
    def num_warps(self) -> int:
        return 1 << len(self.warp)

    def __str__(self) -> str:
        def fmt(bs: tuple[Basis, ...]) -> str:
            return "[" + ", ".join("(" + ",".join(map(str, b)) + ")" for b in bs) + "]"

        return f"#bits<reg={fmt(self.reg)}, lane={fmt(self.lane)}, warp={fmt(self.warp)}>"


def num_regs(layout: BitLayout) -> int:
    """Returns the number of registers each thread uses for the tile."""
    return layout.num_regs


def owner_mask(layout: BitLayout) -> tuple[int, int]:
    """Returns (lane_mask, warp_mask): the index bits whose bases are zero.

    Threads that differ only in these bits hold the same elements. The owner of an
    element is the thread whose masked bits are all zero; stores and atomics run only
    on owners.
    """
    lm = sum(1 << i for i, b in enumerate(layout.lane) if not any(b))
    wm = sum(1 << i for i, b in enumerate(layout.warp) if not any(b))
    return lm, wm


def is_equivalent(a: BitLayout, b: BitLayout) -> bool:
    """Returns whether `a` and `b` place every element in the same register of the same thread."""
    return a == b


def coords_of(layout: BitLayout, r: int) -> tuple[int, ...]:
    """Returns the compile-time coordinate contribution of register `r`.

    The full coordinate of register `r` in thread `(lane, warp)` is this value plus
    the thread contribution described by `thread_terms`.
    """
    c = [0] * layout.rank
    for i, b in enumerate(layout.reg):
        if r >> i & 1:
            for d, v in enumerate(b):
                c[d] += v
    return tuple(c)


@dataclass(frozen=True)
class ThreadTerm:
    """A run of consecutive index bits that maps to consecutive coordinate bits.

    The run contributes `((index >> src_shift) & ((1 << width) - 1)) << dst_shift` to
    coordinate `dim`, where `index` is the lane or SIMD-group index named by `source`.
    """

    source: str  # "lane" or "warp"
    dim: int
    src_shift: int
    width: int
    dst_shift: int


def thread_terms(layout: BitLayout) -> tuple[ThreadTerm, ...]:
    """Returns the lane and SIMD-group contributions to each coordinate, as bit runs."""
    terms: list[ThreadTerm] = []
    for source, bases in (("lane", layout.lane), ("warp", layout.warp)):
        i = 0
        while i < len(bases):
            db = _dim_bit(bases[i])
            if db is None:
                i += 1
                continue
            d, bit = db
            w = 1
            while i + w < len(bases) and _dim_bit(bases[i + w]) == (d, bit + w):
                w += 1
            terms.append(ThreadTerm(source, d, i, w, bit))
            i += w
    return tuple(terms)


def materialize(layout: BitLayout):
    """Returns an int array [warp, lane, reg, rank] of coordinates. For tests and debugging."""
    import numpy as np

    out = np.zeros((layout.num_warps, 1 << LANE_BITS, layout.num_regs, layout.rank), np.int64)
    for w, l, r in product(range(layout.num_warps), range(1 << LANE_BITS), range(layout.num_regs)):
        c = list(coords_of(layout, r))
        for bases, idx in ((layout.lane, l), (layout.warp, w)):
            for i, b in enumerate(bases):
                if idx >> i & 1:
                    for d, v in enumerate(b):
                        c[d] += v
        out[w, l, r] = c
    return out


# ---- constructors ----


def blocked(
    shape: tuple[int, ...],
    num_warps: int,
    elem_bytes: int,
    order: tuple[int, ...] | None = None,
) -> BitLayout:
    """Returns the default layout for loads, stores, and elementwise ops.

    Bits fill in this order: register bits along the fastest dimension until each
    thread holds 16 contiguous bytes, then lane bits and SIMD-group bits (fastest
    dimension first), then the remaining register bits. Lane and SIMD-group bits
    left over when the tile is smaller than the threadgroup get zero bases. For a 1D
    tile, thread `t` holds elements `t * VEC + j * (threads * VEC)`.

    Args:
        shape: The tile shape.
        num_warps: The number of SIMD groups in the threadgroup (a power of two).
        elem_bytes: The element size in bytes.
        order: Dimensions from fastest to slowest; defaults to row-major.
    """
    rank = len(shape)
    order = tuple(order) if order is not None else tuple(reversed(range(rank)))
    if sorted(order) != list(range(rank)):
        raise ValueError(f"order {order} isn't a permutation of the dimensions")
    bits_left = [_log2(s) for s in shape]
    next_bit = [0] * rank
    numel_bits = sum(bits_left)
    warp_bits = _log2(num_warps)
    threads_bits = LANE_BITS + warp_bits

    def take(n: int) -> list[Basis]:
        out = []
        for d in order:
            while n and bits_left[d]:
                out.append(_unit(rank, d, next_bit[d]))
                next_bit[d] += 1
                bits_left[d] -= 1
                n -= 1
        return out

    # Vector bits: up to 16 bytes along the fastest dimension, but leave enough
    # elements for every thread when the tile is small.
    vec_bits = min(
        _log2(max(1, 16 // elem_bytes)),
        bits_left[order[0]] if rank else 0,
        max(0, numel_bits - threads_bits),
    )
    reg = [_unit(rank, order[0], b) for b in range(vec_bits)]
    if rank:
        next_bit[order[0]] += vec_bits
        bits_left[order[0]] -= vec_bits
    lane = take(LANE_BITS)
    lane += [_zero(rank)] * (LANE_BITS - len(lane))
    warp = take(warp_bits)
    warp += [_zero(rank)] * (warp_bits - len(warp))
    reg += take(sum(bits_left))
    return BitLayout(tuple(shape), tuple(reg), tuple(lane), tuple(warp))


def slice_layout(layout: BitLayout, dim: int) -> BitLayout:
    """Returns the layout of `layout` with dimension `dim` removed.

    This is the layout of a reduction result along `dim`. Register bits that pointed
    into `dim` disappear (the reduction combines those registers), and lane and
    SIMD-group bits that pointed into `dim` become broadcast bits.
    """

    def drop(b: Basis) -> Basis:
        return b[:dim] + b[dim + 1 :]

    reg = tuple(drop(b) for b in layout.reg if not b[dim])
    lane = tuple(_zero(layout.rank - 1) if b[dim] else drop(b) for b in layout.lane)
    warp = tuple(_zero(layout.rank - 1) if b[dim] else drop(b) for b in layout.warp)
    return BitLayout(layout.shape[:dim] + layout.shape[dim + 1 :], reg, lane, warp)


def expand(layout: BitLayout, dim: int) -> BitLayout:
    """Returns `layout` with a size-1 dimension inserted at `dim`."""

    def ins(b: Basis) -> Basis:
        return b[:dim] + (0,) + b[dim:]

    return BitLayout(
        layout.shape[:dim] + (1,) + layout.shape[dim:],
        tuple(ins(b) for b in layout.reg),
        tuple(ins(b) for b in layout.lane),
        tuple(ins(b) for b in layout.warp),
    )


def broadcast(layout: BitLayout, dim: int, size: int) -> BitLayout:
    """Grows size-1 dimension `dim` to `size` by adding register bits.

    Each thread then holds every broadcast copy. When a consumer already has a
    layout, prefer adopting it: `reg_map(layout, consumer)` tells you whether the
    broadcast is free in that layout.
    """
    if layout.shape[dim] != 1:
        raise ValueError(f"dimension {dim} has size {layout.shape[dim]}, not 1")
    shape = layout.shape[:dim] + (size,) + layout.shape[dim + 1 :]
    reg = layout.reg + tuple(_unit(layout.rank, dim, b) for b in range(_log2(size)))
    return BitLayout(shape, reg, layout.lane, layout.warp)


def permute(layout: BitLayout, perm: tuple[int, ...]) -> BitLayout:
    """Returns the layout of the tile transposed by `perm` (result dim i is input dim perm[i])."""

    def p(b: Basis) -> Basis:
        return tuple(b[i] for i in perm)

    return BitLayout(
        tuple(layout.shape[i] for i in perm),
        tuple(p(b) for b in layout.reg),
        tuple(p(b) for b in layout.lane),
        tuple(p(b) for b in layout.warp),
    )


def reshape(layout: BitLayout, shape: tuple[int, ...]) -> BitLayout:
    """Returns the layout of the tile reshaped in row-major element order.

    Reshaping power-of-two shapes regroups the bits of the flat index, so it never
    moves data between threads.
    """
    old_bits = [_log2(s) for s in layout.shape]
    new_bits = [_log2(s) for s in shape]
    if sum(old_bits) != sum(new_bits):
        raise ValueError(f"can't reshape {layout.shape} to {shape}")
    # Flat bit position of bit 0 of each dimension (the last dimension is lowest).
    def starts(bits: list[int]) -> list[int]:
        out, acc = [0] * len(bits), 0
        for d in reversed(range(len(bits))):
            out[d] = acc
            acc += bits[d]
        return out

    old_start, new_start = starts(old_bits), starts(new_bits)

    def conv(b: Basis) -> Basis:
        db = _dim_bit(b)
        if db is None:
            return _zero(len(shape))
        flat = old_start[db[0]] + db[1]
        for d in range(len(shape)):
            if new_start[d] <= flat < new_start[d] + new_bits[d]:
                return _unit(len(shape), d, flat - new_start[d])
        raise AssertionError("unreachable")

    return BitLayout(
        tuple(shape),
        tuple(conv(b) for b in layout.reg),
        tuple(conv(b) for b in layout.lane),
        tuple(conv(b) for b in layout.warp),
    )


def simd_acc(bm: int, bn: int, wm: int, wn: int) -> BitLayout:
    """Returns the `simdgroup_matrix` accumulator layout for a BM x BN tile.

    SIMD groups form a WM x WN grid. Each owns an SM x SN strip (SM = BM / WM,
    SN = BN / WN) made of TM x TN 8x8 fragments. Within a fragment, lane bits 1, 2,
    and 4 select rows 1, 2, and 4, lane bits 0 and 3 select columns 2 and 4, and the
    element index selects column 1.
    """
    if bm % (8 * wm) or bn % (8 * wn):
        raise ValueError(f"BM={bm} and BN={bn} must be multiples of 8*WM and 8*WN")
    sm, sn = bm // wm, bn // wn
    tm, tn = sm // 8, sn // 8
    reg = [(0, 1)]
    reg += [(0, 8 << i) for i in range(_log2(tn))]
    reg += [(8 << i, 0) for i in range(_log2(tm))]
    lane = ((0, 2), (1, 0), (2, 0), (0, 4), (4, 0))
    warp = [(sm << i, 0) for i in range(_log2(wm))]
    warp += [(0, sn << i) for i in range(_log2(wn))]
    return BitLayout((bm, bn), tuple(reg), lane, tuple(warp))


# ---- conversions ----


def reg_map(src: BitLayout, dst: BitLayout) -> tuple[int, ...] | None:
    """Maps each register of `dst` to the `src` register in the same thread with the same element.

    Dimensions where `src` has size 1 and `dst` is larger are broadcast: every
    coordinate along them reads the same `src` element.

    Returns:
        A tuple with one `src` register index per `dst` register, or None when the
        conversion must move data between threads.
    """
    if src.rank != dst.rank or src.num_warps != dst.num_warps:
        return None
    for s, d in zip(src.shape, dst.shape, strict=True):
        if s != d and s != 1:
            return None
    bdims = [i for i in range(src.rank) if src.shape[i] == 1 and dst.shape[i] != 1]

    def proj(b: Basis) -> Basis:
        return tuple(0 if i in bdims else v for i, v in enumerate(b))

    if any(proj(b) != a for a, b in zip(src.lane, dst.lane, strict=True)):
        return None
    if any(proj(b) != a for a, b in zip(src.warp, dst.warp, strict=True)):
        return None
    index = {coords_of(src, r): r for r in range(src.num_regs)}
    out = []
    for r in range(dst.num_regs):
        c = proj(coords_of(dst, r))
        if c not in index:
            return None
        out.append(index[c])
    return tuple(out)
