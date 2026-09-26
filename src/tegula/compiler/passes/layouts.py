"""Layout assignment: choose a `BitLayout` for every tile value that needs a fixed one.

Tile values fall into three classes:

- *cheap*: built only from constants, scalars, `arange`, and elementwise or shape ops
  over those. Codegen rematerializes a cheap value directly in whatever layout each use
  needs, so it never needs a conversion.
- *view*: a shape op (`expand_dims`, `broadcast`, `reshape`, `trans`) over a non-cheap
  value. Codegen derives it from its input's registers on demand.
- *anchored*: everything else. It gets one layout, chosen from its anchors: `load` and
  `store` use `blocked` with the most contiguous dimension fastest, `reduce` uses the
  slice of its input, and elementwise ops adopt an anchored operand's layout.

Codegen converts between layouts where a use needs a different one; a conversion is free
when `layout.reg_map` finds the data in the same thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tegula.compiler import ir
from tegula.compiler import layout as L
from tegula.compiler.errors import CompilationError
from tegula.compiler.passes.axis_info import AxisAnalysis, contiguous_order

CHEAP_SOURCES = frozenset(["const", "splat", "arange", "full"])
ELEMENTWISE = frozenset(["binary", "cmp", "unary", "fma", "select", "cast", "bitcast", "addptr"])
VIEWS = frozenset(["expand_dims", "broadcast", "reshape", "trans"])

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
    kind: dict[int, str] = field(default_factory=dict)
    fixed: dict[int, L.BitLayout] = field(default_factory=dict)
    defining: dict[int, ir.Op] = field(default_factory=dict)
    loop_of: dict[int, ir.Op] = field(default_factory=dict)  # iter arg or result -> for op
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


class _Assigner:
    def __init__(self, module: ir.Module, num_warps: int) -> None:
        self.plan = LayoutPlan(num_warps, AxisAnalysis(module))
        self.in_progress: set[int] = set()
        self.module = module

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
        if op.name == "load":
            order = contiguous_order(p.axis.get(op.operands[0]))
            return p.default(t, order)
        if op.name == "reduce":
            src = op.operands[0]
            lay = self.resolve(src) or p.default(src.type)
            if not isinstance(t, ir.TileType):
                return None
            return L.slice_layout(lay, op.attrs["axis"])
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
            # Prefer operands with a layout of their own over views and cheap values.
            for want in (ANCHORED, VIEW):
                for x in tiles:
                    if p.classify(x) == want:
                        lay = self.resolve(x)
                        if lay is not None:
                            return lay
            return None
        if op.name == "dot":
            raise CompilationError("tl.dot lowering arrives in M4", op.loc)
        return None


def assign_layouts(module: ir.Module, num_warps: int) -> LayoutPlan:
    """Assigns layouts to every anchored tile value and returns the plan."""
    return _Assigner(module, num_warps).run()


def store_layout(plan: LayoutPlan, op: ir.Op) -> L.BitLayout:
    """Returns the layout in which a `store` writes: its value's, else its pointer's."""
    ptr, val = op.operands[0], op.operands[1]
    for v in (val, ptr):
        lay = plan.natural(v)
        if lay is not None:
            return lay
    return plan.default(ptr.type, contiguous_order(plan.axis.get(ptr)))
