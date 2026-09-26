"""`tl.dot` loops on Metal 4 `matmul2d` (Metal Performance Primitives).

With `dot_backend="mpp"`, codegen replaces an eligible `tl.dot` loop with a
`mpp::tensor_ops::matmul2d` op that all SIMD groups of the threadgroup run together
(`execution_simdgroups<num_warps>`). Its operands are `tensor_inline` tensors built
from the descriptors' device pointers, and its result is a cooperative tensor whose
register layout is opaque. The epilogue walks the cooperative tensor's elements, gets
each element's (column, row) in the tile from `get_multidimensional_index`, computes
the elementwise ops for that element, and writes it with a bounds check.

A `tl.dot` is eligible when all of the following hold:

- It sits in a `for` loop, and its accumulator is a value that the loop carries, starts
  from zeros, and uses only as `acc = tl.dot(a, b, acc)`.
- Both operands are tensor descriptor loads (optionally through `tl.trans`) inside the
  loop, used only by the `tl.dot`, from descriptors created outside the loop, at offsets
  that are provably non-negative.
- `matmul2d` supports the operand and accumulator types.
- After the loop, the result reaches exactly one `desc_store` through elementwise ops
  only. Their other operands must be computable per element: constants, scalars,
  `arange`, broadcasts, pointer-tile loads, and descriptor loads in the same block.

A loop `for k in range(0, K, BK)` whose only work is the `tl.dot`, where `K` is the K
extent of both descriptors and the K offsets are `k`, runs as one `matmul2d` over the
whole K range (`dynamic_length_v`). Any other eligible loop keeps its structure and
runs one `multiply_accumulate` step of BK per iteration.

Anything else falls back to the `simdgroup` lowering in `dot.py`, and the `enceladus`
logger records the reason at debug level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from enceladus.compiler import ir
from enceladus.compiler.codegen.msl import CTYPES, _body_ops, literal

if TYPE_CHECKING:
    from enceladus.compiler.codegen.msl import _Codegen

log = logging.getLogger("enceladus")

LANGUAGE_VERSION = (4, 0)
INCLUDES = ("<metal_tensor>", "<MetalPerformancePrimitives/MetalPerformancePrimitives.h>")
ELEMENTWISE = frozenset(["binary", "cmp", "unary", "fma", "select", "cast", "bitcast"])
CHEAP = frozenset(["const", "full", "splat", "arange"])
VIEWS = frozenset(["expand_dims", "broadcast", "trans"])
# (operand, accumulator) element types that `matmul2d` supports.
TYPES = frozenset([("f16", "f32"), ("f16", "f16"), ("f32", "f32"), ("bf16", "f32"),
                   ("bf16", "bf16")])  # fmt: skip
_MODE = "mpp::tensor_ops::matmul2d_descriptor::mode::"
# The tile sizes that give correct results on macOS 27 (M4 Pro). A 256-row or 256-column
# tile over 1 or 2 SIMD groups returns wrong results, even in Apple's own
# `op.run(A, B, C)` form, and an 8x8 tile doesn't compile.
MIN_TILE, MAX_TILE = 16, 128


class _Ineligible(Exception):
    pass


@dataclass
class _Operand:
    load: ir.Op  # the desc_load
    trans: ir.Op | None  # the tl.trans around it, if any
    k_dim: int  # the descriptor dimension that runs along K

    @property
    def outer_dim(self) -> int:
        return 1 - self.k_dim


@dataclass
class MppLoop:
    """One `tl.dot` loop that lowers to `matmul2d`."""

    loop: ir.Op
    dot: ir.Op
    index: int  # the accumulator's position among the loop-carried values
    a: _Operand
    b: _Operand
    dynamic_k: bool
    store: ir.Op
    ct: str = ""  # the cooperative tensor, named during codegen
    op_name: str = ""
    tensors: tuple[str, str] = ("", "")


@dataclass
class MppPlan:
    """The `tl.dot` loops of a kernel that lower to `matmul2d`, and why others don't."""

    loops: dict[int, MppLoop] = field(default_factory=dict)  # id(for op) -> loop
    dots: dict[int, MppLoop] = field(default_factory=dict)
    stores: dict[int, MppLoop] = field(default_factory=dict)
    skip: set[int] = field(default_factory=set)  # ops that the MPP code replaces
    fallbacks: list[str] = field(default_factory=list)
    handled: set[int] = field(default_factory=set)  # every op id above


def plan_mpp(module: ir.Module) -> MppPlan:
    """Finds the `tl.dot` loops of `module` that lower to `matmul2d`."""
    users: dict[int, list[tuple[ir.Op, int]]] = {}
    for op in module.walk():
        for i, v in enumerate(op.operands):
            users.setdefault(id(v), []).append((op, i))
    p = MppPlan()
    for op in module.walk():
        if op.name != "dot":
            continue
        try:
            ml, skip = _analyze(op, users)
        except _Ineligible as e:
            log.debug("enceladus: %s: the tl.dot at %s uses the simdgroup backend: %s",
                      module.name, op.loc, e)  # fmt: skip
            p.fallbacks.append(f"{op.loc}: {e}")
            continue
        p.loops[id(ml.loop)] = ml
        p.dots[id(op)] = ml
        p.stores[id(ml.store)] = ml
        p.skip |= skip
    p.handled = p.skip | p.loops.keys() | p.dots.keys() | p.stores.keys()
    return p


# ---- analysis ----


def _const(v: ir.Value):
    op = v.defining_op
    return op.attrs["value"] if op is not None and op.name == "const" else None


def _is_zero(v: ir.Value) -> bool:
    op = v.defining_op
    if op is None:
        return False
    if op.name in ("full", "const"):
        return op.attrs["value"] == 0
    return op.name == "splat" and _const(op.operands[0]) == 0


def _inside(op: ir.Op | None, loop: ir.Op) -> bool:
    """Returns whether `op` is nested anywhere in `loop`."""
    while op is not None:
        if op is loop:
            return True
        blk = op.parent
        region = blk.parent if blk is not None else None
        op = region.parent if region is not None else None
    return False


def _nonneg(v: ir.Value) -> bool:
    """Returns whether scalar `v` is provably non-negative."""
    op = v.defining_op
    if op is None:
        blk = v.owner
        region = blk.parent if isinstance(blk, ir.Block) else None
        loop = region.parent if region is not None else None
        if loop is not None and loop.name == "for" and v is blk.args[0]:
            step = _const(loop.operands[2])
            return step is not None and step > 0 and _nonneg(loop.operands[0])
        return False
    if op.name == "const":
        return op.attrs["value"] >= 0
    if op.name in ("program_id", "num_programs"):
        return True
    if op.name == "cast":
        return _nonneg(op.operands[0])
    if op.name == "binary":
        kind = op.attrs["op"]
        if kind in ("add", "mul", "floordiv", "shr", "min"):
            return all(_nonneg(x) for x in op.operands)
        if kind == "max":
            return any(_nonneg(x) for x in op.operands)
    return False


def _operand(v: ir.Value, body: ir.Block, loop: ir.Op, users, left: bool) -> _Operand:
    side = "left" if left else "right"
    src, tr = v.defining_op, None
    if src is not None and src.name == "trans":
        if len(users.get(id(v), [])) != 1:
            raise _Ineligible(f"the {side} operand has uses other than the tl.dot")
        tr, v = src, src.operands[0]
        src = v.defining_op
    if src is None or src.name != "desc_load" or src.parent is not body:
        raise _Ineligible(f"the {side} operand isn't a tensor descriptor load inside the loop")
    if len(users.get(id(v), [])) != 1:
        raise _Ineligible(f"the {side} operand has uses other than the tl.dot")
    mk = src.operands[0].defining_op
    if mk is None or mk.name != "make_desc" or _inside(mk, loop):
        raise _Ineligible(f"the {side} operand's descriptor isn't created outside the loop")
    for x in src.operands[1:]:
        if not _nonneg(x):
            raise _Ineligible(f"an offset of the {side} operand isn't provably non-negative")
    k_dim = (1 if left else 0) if tr is None else (0 if left else 1)
    return _Operand(src, tr, k_dim)


def _side_effects(op: ir.Op) -> bool:
    return any(o.name in ("atomic_rmw", "atomic_cas") or
               (not o.results and not o.regions and o.name not in ("yield", "return"))
               for o in op.walk())  # fmt: skip


def _epilogue(result: ir.Value, loop: ir.Op, users) -> tuple[ir.Op, set[int], list[ir.Op]]:
    """Checks the uses of the loop result.

    Returns:
        The terminal `desc_store`, the ids of the values derived from `result`, and the
        ops whose values the epilogue recomputes per element, in no particular order.
    """
    blk = loop.parent
    derived = {id(result)}
    ops: list[ir.Op] = []
    stores: list[ir.Op] = []
    stack = [result]
    while stack:
        v = stack.pop()
        for u, i in users.get(id(v), []):
            if u.name == "desc_store" and i == len(u.operands) - 1:
                if u.parent is not blk:
                    raise _Ineligible("the store of the tl.dot result isn't in the loop's block")
                if u not in stores:
                    stores.append(u)
            elif u.name in ELEMENTWISE and u.parent is blk:
                if id(u.result) not in derived:
                    derived.add(id(u.result))
                    ops.append(u)
                    stack.append(u.result)
            else:
                raise _Ineligible(f"the tl.dot result feeds `{u.name}`, which isn't an "
                                  "elementwise op in the loop's block")  # fmt: skip
    if len(stores) != 1:
        raise _Ineligible("the tl.dot result doesn't reach exactly one tensor descriptor store")
    trees: list[ir.Op] = []
    seen: set[int] = set()

    def visit(v: ir.Value) -> None:
        if not isinstance(v.type, ir.TileType) or id(v) in derived or id(v) in seen:
            return
        seen.add(id(v))
        op = v.defining_op
        if op is None:
            raise _Ineligible("an epilogue operand is a tile carried by a loop")
        if op.name in ("load", "desc_load") and op.parent is not blk:
            raise _Ineligible("an epilogue operand loads outside the loop's block")
        if op.name in ("select", "addptr", "load") or op.name in ELEMENTWISE or \
                op.name in CHEAP or op.name in VIEWS:  # fmt: skip
            if op.name == "select" and isinstance(ir.elem_of(op.result.type), ir.PointerType):
                raise _Ineligible("an epilogue operand selects between pointers")
            for x in op.operands:
                visit(x)
        elif op.name != "desc_load":
            raise _Ineligible(f"an epilogue operand comes from `{op.name}`, which the matmul2d "
                              "epilogue can't compute per element")  # fmt: skip
        trees.append(op)

    for op in ops:
        for x in op.operands:
            visit(x)
    store = stores[0]
    # The epilogue reads memory at the store, so nothing between an epilogue load and
    # the store may write memory.
    pos = {id(o): k for k, o in enumerate(blk.ops)}
    loads = [pos[id(o)] for o in trees if o.name in ("load", "desc_load")]
    if loads:
        between = {id(o) for o in ops} | {id(o) for o in trees}
        for o in blk.ops[min(loads) + 1 : pos[id(store)]]:
            if id(o) not in between and _side_effects(o):
                raise _Ineligible("memory is written between an epilogue load and the store")
    return store, derived, ops + trees


def _dynamic_k(loop: ir.Op, dot: ir.Op, a: _Operand, b: _Operand, body: ir.Block) -> bool:
    """Returns whether the loop is `for k in range(0, K, BK)` around the tl.dot alone."""
    lb, ub, step = loop.operands[:3]
    if _const(lb) != 0 or _const(step) != dot.operands[0].type.shape[1] or \
            len(loop.results) != 1:  # fmt: skip
        return False
    iv = body.args[0]
    for o in (a, b):
        mk = o.load.operands[0].defining_op
        if o.load.operands[1 + o.k_dim] is not iv or mk.operands[1 + o.k_dim] is not ub:
            return False
    mine = {id(dot), id(a.load), id(b.load), id(body.ops[-1])}
    mine |= {id(o.trans) for o in (a, b) if o.trans is not None}
    for op in body.ops:
        if id(op) in mine:
            continue
        if op.regions or not op.results or \
                any(isinstance(r.type, ir.TileType) for r in op.results) or \
                any(v is iv for v in op.operands) or _side_effects(op):  # fmt: skip
            return False
    return True


def _analyze(dot: ir.Op, users) -> tuple[MppLoop, set[int]]:
    body = dot.parent
    region = body.parent if body is not None else None
    loop = region.parent if region is not None else None
    if loop is None or loop.name != "for":
        raise _Ineligible("the tl.dot isn't directly inside a for loop")
    a, b, c = dot.operands
    if c.owner is not body or c not in body.args[1:]:
        raise _Ineligible("the accumulator isn't carried by the loop around the tl.dot")
    index = body.args.index(c) - 1
    if len(users.get(id(c), [])) != 1:
        raise _Ineligible("the accumulator has uses other than the tl.dot")
    res_users = users.get(id(dot.result), [])
    if len(res_users) != 1 or res_users[0] != (body.ops[-1], index):
        raise _Ineligible("the tl.dot result has uses other than the next iteration")
    if not _is_zero(loop.operands[3 + index]):
        raise _Ineligible("the accumulator doesn't start from zeros")
    bm, bn = dot.result.type.shape
    if not (MIN_TILE <= bm <= MAX_TILE and MIN_TILE <= bn <= MAX_TILE):
        raise _Ineligible(f"the {bm}x{bn} tile is outside the {MIN_TILE}-{MAX_TILE} range "
                          "that Enceladus validates for matmul2d")  # fmt: skip
    in_t, acc_t = a.type.elem.name, dot.result.type.elem.name
    if (in_t, acc_t) not in TYPES:
        raise _Ineligible(f"matmul2d doesn't support {in_t} operands with a {acc_t} accumulator")
    oa = _operand(a, body, loop, users, True)
    ob = _operand(b, body, loop, users, False)
    store, _, epilogue_ops = _epilogue(loop.results[index], loop, users)
    ml = MppLoop(loop, dot, index, oa, ob, _dynamic_k(loop, dot, oa, ob, body), store)
    skip = {id(oa.load), id(ob.load)} | {id(o.trans) for o in (oa, ob) if o.trans is not None}
    # Skip every epilogue op whose users the epilogue replaces entirely. The others still
    # emit their usual code, and the epilogue recomputes them per element.
    absorbed = {id(store)} | {id(o) for o in epilogue_ops}
    changed = True
    while changed:
        changed = False
        for o in epilogue_ops:
            if id(o) in absorbed and not all(id(u) in absorbed for r in o.results
                                             for u, _ in users.get(id(r), [])):  # fmt: skip
                absorbed.discard(id(o))
                changed = True
    skip |= absorbed - {id(store)}
    return ml, skip


# ---- codegen ----


def emit(cg: _Codegen, op: ir.Op) -> None:
    """Emits an op that the MPP plan handles. Ops that it replaces emit nothing."""
    p = cg.mpp
    if id(op) in p.loops:
        _emit_loop(cg, p.loops[id(op)])
    elif id(op) in p.dots:
        _emit_step(cg, p.dots[id(op)])
    elif id(op) in p.stores:
        _emit_epilogue(cg, p.stores[id(op)])


def _desc(cg: _Codegen, o: _Operand):
    return cg.descs[id(o.load.operands[0])]


def _tensor(cg: _Codegen, o: _Operand, elem: str) -> str:
    """Declares a `tensor_inline` over the operand's descriptor, extents innermost first."""
    d = _desc(cg, o)
    name = cg.fresh("mt")
    ty = f"metal::tensor<device {elem}, metal::dextents<int, 2>, metal::tensor_inline>"
    cg.e.line(f"{ty} {name}((device {elem}*)({d.base}), metal::dextents<int, 2>(int({d.shape[1]}), "
              f"int({d.shape[0]})), metal::array<int, 2>{{1, int({d.strides[0]})}});")  # fmt: skip
    return name


def _slice(tensor: str, offsets: list[str]) -> str:
    return f"{tensor}.slice({offsets[1]}, {offsets[0]})"


def _zero(cg: _Codegen, ct: str) -> None:
    i = cg.fresh("mi")
    cg.e.line("#pragma unroll")
    cg.e.line(f"for (uint16_t {i} = 0; {i} < {ct}.get_capacity(); ++{i}) "
              f"if ({ct}.is_valid_element({i})) {ct}[{i}] = 0;")  # fmt: skip


def _guard(cg: _Codegen, ml: MppLoop, offs: tuple[list[str], list[str]], k: bool) -> str:
    """Returns the uniform condition that both operand slices start inside their tensors."""
    conds = []
    for o, off in zip((ml.a, ml.b), offs, strict=True):
        d = _desc(cg, o)
        conds.append(f"{off[o.outer_dim]} < {d.shape[o.outer_dim]}")
        conds.append(f"{off[o.k_dim]} < {d.shape[o.k_dim]}" if k else f"0 < {d.shape[o.k_dim]}")
    return " && ".join(dict.fromkeys(conds))


def _emit_loop(cg: _Codegen, ml: MppLoop) -> None:
    cg.require_language_version(LANGUAGE_VERSION)
    for h in INCLUDES:
        cg.require_include(h)
    a, b = ml.dot.operands[:2]
    bm, bk = a.type.shape
    bn = b.type.shape[1]
    elem = CTYPES[a.type.elem.name]
    acc = CTYPES[ml.dot.result.type.elem.name]
    ta, tb = _tensor(cg, ml.a, elem), _tensor(cg, ml.b, elem)
    ml.tensors = (ta, tb)
    desc = cg.fresh("mm_desc")
    k = "mpp::tensor_ops::dynamic_length_v<int>" if ml.dynamic_k else str(bk)
    mode = "multiply" if ml.dynamic_k else "multiply_accumulate"
    ta_t = "true" if ml.a.trans is not None else "false"
    tb_t = "true" if ml.b.trans is not None else "false"
    cg.e.line(f"constexpr auto {desc} = mpp::tensor_ops::matmul2d_descriptor({bm}, {bn}, {k}, "
              f"{ta_t}, {tb_t}, false, {_MODE}{mode});")  # fmt: skip
    ml.op_name = cg.fresh("mm")
    cg.e.line(f"mpp::tensor_ops::matmul2d<{desc}, metal::execution_simdgroups<{cg.nw}>> "
              f"{ml.op_name};")  # fmt: skip
    ml.ct = cg.fresh("ct")
    cg.e.line(f"auto {ml.ct} = {ml.op_name}.template get_destination_cooperative_tensor<"
              f"decltype({ta}.slice(0, 0)), decltype({tb}.slice(0, 0)), {acc}>();")  # fmt: skip
    if ml.dynamic_k:
        body = ml.loop.regions[0].block
        # Everything else in the body is a scalar that the slice offsets need.
        for op in _body_ops(body).ops:
            if id(op) not in cg.mpp.handled:
                cg.loc = op.loc or cg.loc
                cg.op(op)
        offs = tuple([("0" if i == o.k_dim else cg.s(x)) for i, x in enumerate(o.load.operands[1:])]
                     for o in (ml.a, ml.b))  # fmt: skip
        with cg.e.block(f"if ({_guard(cg, ml, offs, False)})"):
            _run(cg, ml, offs)
        with cg.e.block("else"):
            _zero(cg, ml.ct)
        return
    _zero(cg, ml.ct)
    _emit_manual_loop(cg, ml)


def _run(cg: _Codegen, ml: MppLoop, offs: tuple[list[str], list[str]]) -> None:
    sa, sb = cg.fresh("ms"), cg.fresh("ms")
    cg.e.line(f"auto {sa} = {_slice(ml.tensors[0], offs[0])};")
    cg.e.line(f"auto {sb} = {_slice(ml.tensors[1], offs[1])};")
    cg.e.line(f"{ml.op_name}.run({sa}, {sb}, {ml.ct});")


def _emit_step(cg: _Codegen, ml: MppLoop) -> None:
    """Emits one BK step of the K loop, in place of the tl.dot."""
    offs = tuple([cg.s(x) for x in o.load.operands[1:]] for o in (ml.a, ml.b))
    with cg.e.block(f"if ({_guard(cg, ml, offs, True)})"):
        _run(cg, ml, offs)


def _emit_manual_loop(cg: _Codegen, ml: MppLoop) -> None:
    """Emits the loop like `_Codegen.op_for`, with the accumulator in the cooperative tensor."""
    loop = ml.loop
    lb, ub, step = (cg.s(v) for v in loop.operands[:3])
    body = loop.regions[0].block
    carried: list = []
    for j, (arg, init) in enumerate(zip(body.args[1:], loop.operands[3:], strict=True)):
        if j == ml.index:
            carried.append(None)
        elif isinstance(arg.type, ir.TileType):
            t = cg.mat(init, cg.plan.layout_of(arg))
            carried.append(cg._carried(arg, t, None, arg.name_hint))
        else:
            carried.append(cg._carried(arg, None, cg.s(init), arg.name_hint))
            if id(init) in cg.roots:
                cg.roots[id(arg)] = cg.roots[id(init)]
    for arg, c in zip(body.args[1:], carried, strict=True):
        if c is not None:
            cg._bind(arg, c)
    iv = body.args[0]
    ivn = cg.fresh(iv.name_hint or "i")
    cg.sv[id(iv)] = ivn
    s = _const(loop.operands[2])
    if s is not None:
        cond = f"{ivn} < {ub}" if s > 0 else f"{ivn} > {ub}"
    else:
        cond = f"({step} > 0 ? {ivn} < {ub} : {ivn} > {ub})"
    keep = [j for j, c in enumerate(carried) if c is not None]
    with cg.e.block(f"for ({CTYPES[iv.type.name]} {ivn} = {lb}; {cond}; {ivn} += {step})"):
        cg.push_scope()
        cg.block(_body_ops(body))
        ys = body.ops[-1].operands
        cg._yield_into([carried[j] for j in keep], [ys[j] for j in keep])
        cg.pop_scope()
    for r, c in zip(loop.results, carried, strict=True):
        if c is not None:
            cg._bind(r, c)


@dataclass
class _Ptr:
    base: str
    off: str


class _Elem:
    """Emits the value of one tile element at given coordinates, as MSL locals."""

    def __init__(self, cg: _Codegen, result: ir.Value, acc: str) -> None:
        self.cg, self.result, self.acc = cg, result, acc
        self.memo: dict[tuple, str | _Ptr] = {}

    def __call__(self, v: ir.Value, coords: tuple[str, ...]):
        if not isinstance(v.type, ir.TileType):
            return self.cg.s(v)
        if v is self.result:
            return self.acc
        key = (id(v), coords)
        if key not in self.memo:
            self.memo[key] = self._compute(v, coords)
        return self.memo[key]

    def _bind(self, v: ir.Value, expr: str) -> str:
        name = self.cg.fresh(v.name_hint or "t")
        self.cg.e.line(f"const {self.cg.ctype(v.type)} {name} = {expr};")
        return name

    def _compute(self, v: ir.Value, coords: tuple[str, ...]):
        cg = self.cg
        op = v.defining_op
        n = op.name
        if n in ("const", "full"):
            return literal(op.attrs["value"], v.type.elem)
        if n == "splat":
            x = op.operands[0]
            return _Ptr(cg.s(x), "0") if isinstance(x.type, ir.PointerType) else cg.s(x)
        if n == "arange":
            start = op.attrs["start"]
            return f"({start} + {coords[0]})" if start else coords[0]
        src = op.operands[0]
        if n == "expand_dims":
            ax = op.attrs["axis"]
            return self(src, coords[:ax] + coords[ax + 1 :])
        if n == "broadcast":
            dims = zip(src.type.shape, v.type.shape, coords, strict=True)
            return self(src, tuple("0" if s != d else c for s, d, c in dims))
        if n == "trans":
            out = [""] * len(coords)
            for d, p in enumerate(op.attrs["perm"]):
                out[p] = coords[d]
            return self(src, tuple(out))
        if n == "addptr":
            p, o = self(src, coords), self(op.operands[1], coords)
            if cg.off_t == "long" and ir.elem_of(op.operands[1].type).name not in ("i64", "u64"):
                o = f"long({o})"
            return _Ptr(p.base, o if p.off == "0" else f"({p.off} + {o})")
        if n == "load":
            p = self(src, coords)
            e = f"{p.base}[{p.off}]"
            if len(op.operands) == 3:
                m = self(op.operands[1], coords)
                if m != "true":
                    e = f"({m} ? {e} : {self(op.operands[2], coords)})"
            return self._bind(v, e)
        if n == "desc_load":
            d = cg.descs[id(src)]
            cond, elem = _element(d, [cg.s(x) for x in op.operands[1:]], coords)
            return self._bind(v, f"({cond}) ? {d.base}[{elem}] : {CTYPES[d.elem.name]}(0)")
        args = [self(x, coords) for x in op.operands]
        return self._bind(v, cg.expr(op, args, [x.type for x in op.operands]))


def _element(d, offsets: list[str], coords: tuple[str, ...]) -> tuple[str, str]:
    """Returns (in-bounds condition, element offset) of block element `coords`."""
    idx = [f"({o} + {c})" if o != "0" else c for o, c in zip(offsets, coords, strict=True)]
    cond = " && ".join(f"{x} >= 0 && {x} < {s}" for x, s in zip(idx, d.shape, strict=True))
    elem = " + ".join(f"{x} * {st}" if st != "1" else x
                      for x, st in zip(idx, d.strides, strict=True))  # fmt: skip
    return cond, elem


def _emit_epilogue(cg: _Codegen, ml: MppLoop) -> None:
    """Walks the cooperative tensor, applies the elementwise ops, and stores each element."""
    store = ml.store
    d = cg.descs[id(store.operands[0])]
    offsets = [cg.s(x) for x in store.operands[1:-1]]
    if d.root:
        cg.written.add(d.root)
    ct = ml.ct
    i, idx, row, col = cg.fresh("mi"), cg.fresh("mx"), cg.fresh("er"), cg.fresh("ec")
    cg.e.line("#pragma unroll")
    with cg.e.block(f"for (uint16_t {i} = 0; {i} < {ct}.get_capacity(); ++{i})"):
        cg.e.line(f"if (!{ct}.is_valid_element({i})) continue;")
        cg.e.line(f"const auto {idx} = {ct}.get_multidimensional_index({i});")
        cg.e.line(f"const int {row} = int({idx}[1]);")
        cg.e.line(f"const int {col} = int({idx}[0]);")
        ev = _Elem(cg, ml.loop.results[ml.index], f"{ct}[{i}]")
        val = ev(store.operands[-1], (row, col))
        cond, elem = _element(d, offsets, (row, col))
        cg.e.line(f"if ({cond}) {d.base}[{elem}] = {CTYPES[d.elem.name]}({val});")
