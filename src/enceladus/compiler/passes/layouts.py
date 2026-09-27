"""Layout assignment: choose a `BitLayout` for every tile value that needs a fixed one.

Tile values fall into three classes:

- *cheap*: built only from constants, scalars, `arange`, and elementwise or shape ops
  over those. Codegen rematerializes a cheap value directly in whatever layout each use
  needs, so it never needs a conversion.
- *view*: a shape op (`expand_dims`, `broadcast`, `reshape`, `trans`) over a non-cheap
  value. Codegen derives it from its input's registers on demand.
- *anchored*: everything else. It gets one layout, chosen from its anchors: `load` and
  `store` use `blocked` with the most contiguous dimension fastest, `reduce` uses the
  slice of its input, and elementwise ops adopt an anchored operand's layout. A load
  that only feeds the left operand of `tl.dot`s inside a loop below it (the query tile of
  attention) uses the dot's register operand layout, so it stays in registers across the
  loop instead of being reloaded or restaged each iteration.

Codegen converts between layouts where a use needs a different one; a conversion is free
when `layout.reg_map` finds the data in the same thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from enceladus.compiler import ir
from enceladus.compiler import layout as L
from enceladus.compiler.errors import CompilationError
from enceladus.compiler.passes.axis_info import AxisAnalysis, contiguous_order

CHEAP_SOURCES = frozenset(["const", "splat", "arange", "full"])
ELEMENTWISE = frozenset(["binary", "cmp", "unary", "fma", "select", "cast", "bitcast", "addptr"])
# `hint` (`tl.multiple_of` and `tl.max_contiguous`) is an identity that only carries
# facts for `axis_info`, so it's a view whose layout is its input's.
VIEWS = frozenset(["expand_dims", "broadcast", "reshape", "trans", "hint"])

CHEAP, VIEW, ANCHORED = "cheap", "view", "anchored"


def elem_bytes(t: ir.Type) -> int:
    """Returns the element size used for layout vectorization (pointee size for pointers)."""
    e = ir.elem_of(t)
    if isinstance(e, ir.PointerType):
        e = e.elem
    return max(1, e.dtype.itemsize)


@dataclass
class LayoutPlan:
    """The result of layout assignment."""

    num_warps: int
    axis: AxisAnalysis
    dot_warps: tuple[int, int] | None = None
    kind: dict[int, str] = field(default_factory=dict)
    fixed: dict[int, L.BitLayout] = field(default_factory=dict)
    defining: dict[int, ir.Op] = field(default_factory=dict)
    loop_of: dict[int, ir.Op] = field(default_factory=dict)  # iter arg or result -> for op
    hoisted: dict[int, ir.Op] = field(default_factory=dict)  # see `hoisted_a_operands`
    arg_index: dict[int, int] = field(default_factory=dict)  # iter arg or result -> index

    def default(self, t: ir.TileType, order: tuple[int, ...] | None = None) -> L.BitLayout:
        return L.blocked(t.shape, self.num_warps, elem_bytes(t), order)

    def classify(self, v: ir.Value) -> str:
        return self.kind.get(id(v), ANCHORED)

    def layout_of(self, v: ir.Value) -> L.BitLayout:
        """Returns the fixed layout of an anchored value."""
        return self.fixed[id(v)]

    def natural(self, v: ir.Value) -> L.BitLayout | None:
        """Returns the layout a value has without conversions, or None if it's flexible."""
        k = self.classify(v)
        if k == ANCHORED:
            return self.fixed.get(id(v))
        if k == CHEAP:
            return None
        op = self.defining[id(v)]
        src = self.natural(op.operands[0])
        if src is None:
            return None
        return view_layout(op, src)


def view_layout(op: ir.Op, src: L.BitLayout) -> L.BitLayout:
    """Returns the layout of a shape op's result given its input layout."""
    t = op.result.type
    if op.name == "hint":
        return src
    if op.name == "expand_dims":
        return L.expand(src, op.attrs["axis"])
    if op.name == "reshape":
        return L.reshape(src, t.shape)
    if op.name == "trans":
        return L.permute(src, tuple(op.attrs["perm"]))
    if op.name == "broadcast":
        out = src
        for d, (a, b) in enumerate(zip(src.shape, t.shape, strict=True)):
            if a != b:
                out = L.broadcast(out, d, b)
        return out
    raise AssertionError(op.name)


def view_source_layout(op: ir.Op, dst: L.BitLayout) -> L.BitLayout:
    """Returns the input layout that makes a shape op produce `dst` without moving data."""
    src_t = op.operands[0].type
    if op.name == "hint":
        return dst
    if op.name == "expand_dims":
        ax = op.attrs["axis"]
        return L.BitLayout(
            src_t.shape,
            tuple(b[:ax] + b[ax + 1 :] for b in dst.reg),
            tuple(b[:ax] + b[ax + 1 :] for b in dst.lane),
            tuple(b[:ax] + b[ax + 1 :] for b in dst.warp),
        )
    if op.name == "reshape":
        return L.reshape(dst, src_t.shape)
    if op.name == "trans":
        perm = tuple(op.attrs["perm"])
        inv = tuple(perm.index(i) for i in range(len(perm)))
        return L.permute(dst, inv)
    if op.name == "broadcast":
        bd = [d for d, (a, b) in enumerate(zip(src_t.shape, dst.shape, strict=True)) if a != b]

        def proj(b):
            return tuple(0 if i in bd else x for i, x in enumerate(b))

        return L.BitLayout(
            src_t.shape,
            tuple(proj(b) for b in dst.reg if any(proj(b))),
            tuple(proj(b) for b in dst.lane),
            tuple(proj(b) for b in dst.warp),
        )
    raise AssertionError(op.name)


def crosses_loop(block: ir.Block, op: ir.Op) -> bool:
    """Returns whether a `for` body lies between `block` and `op`, which `block` encloses."""
    blk = op.parent
    while blk is not None and blk is not block:
        region = blk.parent
        parent = region.parent if region is not None else None
        if parent is None:
            return False
        if parent.name == "for":
            return True
        blk = parent.parent
    return False


# The most 32-bit registers per thread that a left operand may hold across a loop. The
# flash attention query tile at head dimension 128 needs 16 in FP16 and 32 in FP32.
MAX_HOISTED_REGS = 32


def hoisted_a_operands(module: ir.Module, num_warps: int,
                       override: tuple[int, int] | None = None) -> dict[int, ir.Op]:  # fmt: skip
    """Returns the loads that stay in registers as the left operand of `tl.dot`s in a loop.

    A `load` or `desc_load` qualifies when every use is the left operand of a `dot` inside
    a `for` loop nested below the load, and its share per thread fits in
    `MAX_HOISTED_REGS` registers. Loading it once, in the dot's register operand layout,
    beats reloading it from device memory or restaging it through threadgroup memory on
    every iteration.

    Args:
        module: The kernel.
        num_warps: SIMD groups per threadgroup.
        override: The `dot_warps` launch option, if any.

    Returns:
        A dict from the id of each qualifying value to the first `dot` that reads it.
    """
    users: dict[int, list[tuple[ir.Op, int]]] = {}
    for op in module.walk():
        for i, v in enumerate(op.operands):
            users.setdefault(id(v), []).append((op, i))
    out: dict[int, ir.Op] = {}
    for op in module.walk():
        if op.name not in ("load", "desc_load") or not isinstance(op.result.type, ir.TileType):
            continue
        us = users.get(id(op.result), [])
        if not us or not all(u.name == "dot" and i == 0 and crosses_loop(op.parent, u)
                             for u, i in us):  # fmt: skip
            continue
        dot = us[0][0]
        (bm, bk), bn = op.result.type.shape, dot.result.type.shape[1]
        wm, _ = dot_warps(bm, bn, num_warps, dot.loc, override)
        if bk % 8 == 0 and bm * bk // (32 * wm) * elem_bytes(op.result.type) <= \
                4 * MAX_HOISTED_REGS:  # fmt: skip
            out[id(op.result)] = dot
    return out


class _Assigner:
    def __init__(self, module: ir.Module, num_warps: int, dot_warps=None) -> None:
        self.plan = LayoutPlan(num_warps, AxisAnalysis(module), dot_warps)
        self.in_progress: set[int] = set()
        self.module = module
        self.plan.hoisted = hoisted_a_operands(module, num_warps, dot_warps)

    def run(self) -> LayoutPlan:
        self._classify_block(self.module.body)
        for op in self.module.walk():
            for r in op.results:
                if isinstance(r.type, ir.TileType) and self.plan.classify(r) == ANCHORED:
                    self.resolve(r)
            for rg in op.regions:
                for a in rg.block.args:
                    if isinstance(a.type, ir.TileType):
                        self.resolve(a)
        return self.plan

    # ---- classification ----

    def _classify_block(self, block: ir.Block) -> None:
        p = self.plan
        for op in block.ops:
            if op.name == "for":
                body = op.regions[0].block
                for i, (a, r) in enumerate(zip(body.args[1:], op.results, strict=True)):
                    for v in (a, r):
                        p.loop_of[id(v)] = op
                        p.arg_index[id(v)] = i
            for rg in op.regions:
                self._classify_block(rg.block)
            for r in op.results:
                p.defining[id(r)] = op
                if not isinstance(r.type, ir.TileType):
                    continue
                tiles = [v for v in op.operands if isinstance(v.type, ir.TileType)]
                if op.name in CHEAP_SOURCES:
                    k = CHEAP
                elif op.name in ELEMENTWISE:
                    k = CHEAP if all(p.classify(v) == CHEAP for v in tiles) else ANCHORED
                elif op.name in VIEWS:
                    k = CHEAP if p.classify(op.operands[0]) == CHEAP else VIEW
                else:
                    k = ANCHORED
                p.kind[id(r)] = k

    # ---- resolution ----

    def resolve(self, v: ir.Value) -> L.BitLayout | None:
        p = self.plan
        if id(v) in p.fixed:
            return p.fixed[id(v)]
        k = p.classify(v)
        if k == CHEAP:
            return None
        if k == VIEW:
            op = p.defining[id(v)]
            src = self.resolve(op.operands[0])
            return None if src is None else view_layout(op, src)
        if id(v) in self.in_progress:
            return None
        self.in_progress.add(id(v))
        try:
            lay = self._compute(v)
        finally:
            self.in_progress.discard(id(v))
        if lay is None:
            lay = p.default(v.type)
        p.fixed[id(v)] = lay
        return lay

    def _compute(self, v: ir.Value) -> L.BitLayout | None:
        p = self.plan
        t = v.type
        if id(v) in p.loop_of:
            op = p.loop_of[id(v)]
            i = p.arg_index[id(v)]
            arg = op.regions[0].block.args[1 + i]
            if v is not arg:  # a loop result shares its block argument's layout
                return self.resolve(arg)
            init = op.operands[3 + i]
            yielded = op.regions[0].block.ops[-1].operands[i]
            for cand in (init, yielded):
                if p.classify(cand) != CHEAP:
                    lay = self.resolve(cand)
                    if lay is not None:
                        return lay
            return None
        op = p.defining.get(id(v))
        if op is None:
            return None
        if id(v) in p.hoisted:
            dot = p.hoisted[id(v)]
            bm, bk = t.shape
            wm, wn = dot_warps(bm, dot.result.type.shape[1], p.num_warps, dot.loc, p.dot_warps)
            return L.dot_operand_a(bm, bk, wm, wn)
        if op.name == "load":
            order = contiguous_order(p.axis.get(op.operands[0]))
            return p.default(t, order)
        if op.name == "reduce":
            src = op.operands[0]
            lay = self.resolve(src) or p.default(src.type)
            if not isinstance(t, ir.TileType):
                return None
            return L.slice_layout(lay, op.attrs["axis"])
        if op.name == "scan":
            # Every result of a scan keeps the layout of its first input.
            return self.resolve(op.operands[0]) or p.default(op.operands[0].type)
        if op.name in ("atomic_rmw", "atomic_cas"):
            return self._atomic_layout(op)
        if op.name == "if":
            i = op.results.index(v)
            for rg in op.regions:
                y = rg.block.ops[-1].operands[i]
                if p.classify(y) != CHEAP:
                    lay = self.resolve(y)
                    if lay is not None:
                        return lay
            return None
        if op.name in ELEMENTWISE:
            tiles = [x for x in op.operands if isinstance(x.type, ir.TileType)]
            # Prefer operands with a layout of their own, then loop-carried values (whose
            # layout may depend on this op), then views.
            def rank(x: ir.Value) -> int:
                k = p.classify(x)
                return 0 if k == ANCHORED and id(x) not in p.loop_of else 1 if k == ANCHORED \
                    else 2 if k == VIEW else 3
            for x in sorted(tiles, key=rank):
                if rank(x) < 3:
                    lay = self.resolve(x)
                    if lay is not None:
                        return lay
            return None
        if op.name == "dot":
            bm, bn = t.shape
            bk = op.operands[0].type.shape[1]
            wm, wn = dot_warps(bm, bn, p.num_warps, op.loc, p.dot_warps)
            if bk % 8:
                raise CompilationError(
                    f"tl.dot needs a K dimension (the columns of the first operand) that's a "
                    f"multiple of 8, but got {bk}. Use a K block size of at least 8, such as 16 "
                    "or 32.",
                    op.loc,
                )
            return L.simd_acc(bm, bn, wm, wn)
        return None

    def _atomic_layout(self, op: ir.Op) -> L.BitLayout:
        """Returns the layout an atomic runs in: its value's, else its pointer's, like a store."""
        p = self.plan
        ptr = op.operands[0]
        for v in (op.operands[-1] if op.name == "atomic_cas" else op.operands[1], ptr):
            if p.classify(v) != CHEAP:
                lay = self.resolve(v)
                if lay is not None:
                    return lay
        return p.default(op.result.type, contiguous_order(p.axis.get(ptr)))


def dot_warps(bm: int, bn: int, num_warps: int, loc=None,
              override: tuple[int, int] | None = None) -> tuple[int, int]:  # fmt: skip
    """Returns the SIMD-group grid (WM, WN) for a BM x BN `dot`.

    The default stacks every SIMD group along M (WN = 1), the measured best
    configuration. Small BM spreads the remaining SIMD groups along N. `override`
    comes from `dot_warps=(WM, WN)` at launch or in a `enceladus.Config`.
    """
    if override is not None:
        wm, wn = override
        if wm * wn != num_warps:
            raise CompilationError(
                f"dot_warps={override} doesn't multiply to num_warps={num_warps}. Pass a "
                f"(WM, WN) pair with WM * WN == {num_warps}.",
                loc,
            )
    else:
        wm = max(1, min(num_warps, bm // 8))
        wn = num_warps // wm
    if bm % (8 * wm) or bn % (8 * wn):
        raise CompilationError(
            f"tl.dot can't split a {bm}x{bn} tile over {num_warps} SIMD groups, because each "
            "SIMD group needs a strip that's a multiple of 8 in both dimensions. Use larger "
            "blocks, or launch with fewer num_warps.",
            loc,
        )
    return wm, wn


def assign_layouts(module: ir.Module, num_warps: int, dot_warps=None) -> LayoutPlan:
    """Assigns layouts to every anchored tile value and returns the plan."""
    return _Assigner(module, num_warps, dot_warps).run()


def store_layout(plan: LayoutPlan, op: ir.Op) -> L.BitLayout:
    """Returns the layout in which a `store` writes: its value's, else its pointer's."""
    ptr, val = op.operands[0], op.operands[1]
    for v in (val, ptr):
        lay = plan.natural(v)
        if lay is not None:
            return lay
    return plan.default(ptr.type, contiguous_order(plan.axis.get(ptr)))
