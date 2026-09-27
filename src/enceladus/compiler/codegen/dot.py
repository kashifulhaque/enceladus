"""`tl.dot` on `simdgroup_matrix`, and tensor descriptor loads and stores.

Each SIMD group owns an SM x SN strip of the accumulator, made of TM x TN 8x8
fragments (`layout.simd_acc`). For each K step of 8, it loads TM fragments of A and TN
fragments of B, then runs TM x TN `simdgroup_multiply_accumulate` calls.

Operands come from one of three sources:

- *Direct*: a `desc_load` (optionally through `trans`) used only by the `dot`, from a
  descriptor whose innermost stride is 1. Fragments load straight from device memory;
  no threadgroup memory, no barriers. This matches `best_matmul.metal`.
- *Register* (left operand only): a tile already in the dot's register operand layout
  (`layout.dot_operand_a`), or one that a register remap puts there. Its registers become
  the A fragments through `thread_elements()`. With WN = 1, an accumulator is in that
  layout, so `tl.dot(p.to(tl.float16), v, acc)` after `p = f(tl.dot(q, k))` needs no
  data movement. A load that feeds only the left operand of dots in a nested loop is
  loaded once in this layout and stays in registers (see `passes.layouts`).
- *Staged*: any other tile. It's written to threadgroup memory with rows padded by 16
  bytes, then fragments load from there.

Direct loads take an unmasked fast path when the whole block is in bounds, chosen by a
threadgroup-uniform branch per `dot`. Otherwise each fragment tests its own bounds
(uniform across the SIMD group) and only straddling fragments use masked per-lane loads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from enceladus.compiler import ir
from enceladus.compiler import layout as L
from enceladus.compiler.codegen.msl import BARRIER, CTYPES, Tile, _add, is_float
from enceladus.compiler.passes.layouts import ANCHORED, LayoutPlan

if TYPE_CHECKING:
    from enceladus.compiler.codegen.msl import _Codegen


@dataclass
class DescInfo:
    base: str
    shape: list[str]
    strides: list[str]
    block: tuple[int, ...]
    elem: ir.ScalarType
    root: str | None


def use_counts(module: ir.Module) -> dict[int, int]:
    uses: dict[int, int] = {}
    for op in module.walk():
        for v in op.operands:
            uses[id(v)] = uses.get(id(v), 0) + 1
    return uses


MEMORY_WRITES = frozenset(("store", "desc_store", "atomic_rmw", "atomic_cas"))


def _writes_memory(op: ir.Op) -> bool:
    return any(o.name in MEMORY_WRITES for o in op.walk())


def written_between(load: ir.Op, use: ir.Op) -> bool:
    """Returns whether memory can be written after `load` runs and before `use` runs.

    A direct operand reads device memory at its `dot`, not at its `desc_load`, so the two
    reads agree only when nothing writes memory in between. Enceladus doesn't track which
    buffers alias, so any store or atomic counts, whatever it writes.

    The check is conservative. From the block that holds `use` up to the block that holds
    `load`, it looks at the ops before `use` in each block. When a `for` loop sits between
    them, a write anywhere in that loop counts, because a later iteration runs after it.
    A write in the other branch of an `if` doesn't count.
    """
    blk = load.parent
    x = use
    while x.parent is not blk:
        inner = x.parent
        region = inner.parent if inner is not None else None
        owner = region.parent if region is not None else None
        if owner is None:
            return True  # `load` isn't in an enclosing block of `use`
        if owner.name == "if":
            if any(_writes_memory(o) for o in inner.ops[: inner.ops.index(x)]):
                return True
        elif _writes_memory(owner):
            return True
        x = owner
    start, end = blk.ops.index(load), blk.ops.index(x)
    return any(_writes_memory(o) for o in blk.ops[start + 1 : end])


def find_direct_operands(module: ir.Module, plan: LayoutPlan) -> set[int]:
    """Returns ids of `desc_load` and `trans` values that dots read from device memory.

    A left operand loaded outside the dot's loop stays in registers instead (see
    `passes.layouts.hoisted_a_operands`), and a right operand that `stages_b` picks goes
    through threadgroup memory, so neither is direct. Neither is an operand whose memory
    can be written between its load and the dot (see `written_between`).
    """
    uses = use_counts(module)
    out: set[int] = set()

    def direct(v: ir.Value, dot: ir.Op) -> bool:
        src = v.defining_op
        if src is not None and src.name == "trans" and uses.get(id(v)) == 1:
            inner = src.operands[0].defining_op
            if inner is not None and inner.name == "desc_load" and \
                    uses.get(id(src.operands[0])) == 1 and \
                    not written_between(inner, dot):  # fmt: skip
                out.update((id(v), id(src.operands[0])))
                return True
        elif src is not None and src.name == "desc_load" and uses.get(id(v)) == 1 and \
                not written_between(src, dot):  # fmt: skip
            out.add(id(v))
            return True
        return False

    for op in module.walk():
        if op.name != "dot" or not is_float(op.operands[0].type.elem):
            continue  # integer dots stage their operands (see `emit_int_dot`)
        a, b = op.operands[:2]
        a_direct = id(a) not in plan.hoisted and direct(a, op)
        if not stages_b(plan, op, a_direct):
            direct(b, op)
    return out


def stages_b(plan: LayoutPlan, dot: ir.Op, a_direct: bool) -> bool:
    """Returns whether `dot` stages a right operand that it could read from device memory.

    When A comes from registers or threadgroup memory and several SIMD-group rows need the
    same B fragments (WM > 1), loading B once per threadgroup into threadgroup memory beats
    loading it from device memory in every SIMD group. In flash attention, staging K and V
    measured 4.1 against 4.9 TFLOPS at head dimension 128 and 5.1 against 5.3 at 64. A
    matmul whose operands both load directly keeps direct loads, which measured faster in
    FP16 (M4).
    """
    g = L.frag_grid(plan.layout_of(dot.result))
    return not a_direct and g is not None and g[2] > 1


@dataclass
class _Direct:
    desc: DescInfo
    offsets: list[str]
    transposed: bool


def _operand(cg: _Codegen, v: ir.Value) -> _Direct | None:
    if id(v) not in cg.direct:
        return None
    op = v.defining_op
    transposed = op.name == "trans"
    if transposed:
        op = op.operands[0].defining_op
    return _Direct(cg.descs[id(op.operands[0])], [cg.s(x) for x in op.operands[1:]], transposed)


def _frag_consts(cg: _Codegen) -> tuple[str, str]:
    fm = cg.pro("fm", "fm", "int", "int(((lane >> 2) & 4u) + ((lane >> 1) & 3u))")
    fn = cg.pro("fn", "fn", "int", "int(((lane >> 1) & 4u) + ((lane << 1) & 2u))")
    return fm, fn


def emit_dot(cg: _Codegen, op: ir.Op, lay: L.BitLayout) -> Tile:
    a, b, c = op.operands
    if not is_float(a.type.elem):
        return emit_int_dot(cg, op, lay)
    bm, bk = a.type.shape
    bn = b.type.shape[1]
    tm, tn, wm, wn = L.frag_grid(lay)
    sm, sn = bm // wm, bn // wn
    in_t = CTYPES[a.type.elem.name]
    out_t = CTYPES[op.result.type.elem.name]
    lwm = wm.bit_length() - 1
    sr = cg.pro(f"sr|{wm}|{sm}", "sr", "int", f"int(warp & {wm - 1}u) * {sm}") if wm > 1 else "0"
    sc = cg.pro(f"sc|{lwm}|{sn}", "sc", "int", f"int(warp >> {lwm}) * {sn}") if wn > 1 else "0"
    fm, fn = _frag_consts(cg)

    # Accumulator: start from `c`, in place when this dot is its only use and nothing else
    # reads its storage afterward.
    acc_in = cg.mat(c, lay)
    if acc_in.frag == (tm, tn) and cg.uses.get(id(c)) == 1 and _can_clobber(cg, c, op) and \
            acc_in.name and acc_in.frag and op.result.type == c.type:  # fmt: skip
        out = acc_in
        d = acc_in.name
    else:
        d = cg.fresh(op.result.name_hint or "acc")
        cg.e.line(f"simdgroup_matrix<{out_t}, 8, 8> {d}[{tm}][{tn}];")
        out = Tile(lay, d, frag=(tm, tn))
    if out is acc_in:
        pass
    elif acc_in.frag is not None:
        cg._assign(out, acc_in, lay.num_regs)
    elif acc_in.uniform is not None:
        for i in range(tm):
            for j in range(tn):
                cg.e.line(f"{d}[{i}][{j}] = make_filled_simdgroup_matrix<{out_t}, 8, 8>"
                          f"({out_t}({acc_in.uniform}));")  # fmt: skip
    else:
        cg._assign(out, acc_in, lay.num_regs)

    da, db = _operand(cg, a), _operand(cg, b)
    ra = _register_a(cg, a, bm, bk, wm, wn, in_t) if da is None else None
    # Staged operands go to threadgroup memory: A as BM x BK, B as BK x BN. Staging a
    # transposed operand untransposed and loading its fragments transposed measured 2-3%
    # slower in flash attention, so every operand stages in its logical orientation.
    eb = a.type.elem.dtype.itemsize
    pad = 16 // eb
    lda, ldb = bk + pad, bn + pad
    staged = []
    off = 0
    if da is None and ra is None:
        staged.append((a, off, lda))
        off += -(-bm * lda * eb // 16) * 16
    if db is None:
        staged.append((b, off, ldb))
        off += -(-bk * ldb * eb // 16) * 16
    tg_names = {}
    if staged:
        cg.use_tg(off)
        cg.e.line(BARRIER)
        for v, o, ld in staged:
            t = cg.mat(v, cg.plan.natural(v) or cg.plan.default(v.type))
            name = cg.fresh("tga" if v is a else "tgb")
            tg_names[id(v)] = (name, ld)
            cg.e.line(f"threadgroup {in_t}* {name} = (threadgroup {in_t}*)(tg_mem + {o});")
            flat, consts = cg.flat(t.layout, (ld, 1))
            own = cg.owner(t.layout)
            with cg.e.block(f"if ({own})"):
                for r in range(t.layout.num_regs):
                    cg.e.line(f"{name}[{_add(flat, consts[r])}] = {in_t}({t.get(r)});")
        cg.e.line(BARRIER)

    def a_src(i: str, checked: bool, kc: int | None) -> str:
        """Returns the statement that loads A fragment i at K offset kk (fragment kc)."""
        if ra is not None:
            return f"fa[{i}] = {ra}[{i}][{kc}];"
        if da is None:
            name, ld = tg_names[id(a)]
            return f"simdgroup_load(fa[{i}], {name} + ({_add(sr, i + ' * 8')}) * {ld} + kk, {ld});"
        return _direct_load(da, f"fa[{i}]", _add(sr, f"{i} * 8"), "kk", checked, fm, fn)

    def b_src(j: str, checked: bool, frag: str) -> str:
        """Returns the statement that loads B fragment j at K offset kk into `frag`."""
        if db is None:
            name, ld = tg_names[id(b)]
            return f"simdgroup_load({frag}, {name} + kk * {ld} + {_add(sc, j + ' * 8')}, {ld});"
        return _direct_load(db, frag, "kk", _add(sc, f"{j} * 8"), checked, fm, fn)

    # Loading every B fragment of a K step before the MMAs keeps TN fragments live. When
    # they and the accumulator exceed 48 registers per thread, load and use one fragment at
    # a time: FP32 attention at head dimension 128 (TN = 16) spilled and ran at 0.35
    # TFLOPS the first way and 4.3 this way. Below the threshold, which includes every
    # matmul configuration, loading all of them first measured the same or faster.
    out_eb, in_eb = op.result.type.elem.dtype.itemsize, a.type.elem.dtype.itemsize
    one_b = tm * tn * 2 * out_eb // 4 + tn * 2 * in_eb // 4 > 48

    def mma_step(checked: bool, kc: int | None) -> None:
        if one_b:
            cg.e.line(f"simdgroup_matrix<{in_t}, 8, 8> fa[{tm}], fb;")
            cg.e.line(f"for (int i = 0; i < {tm}; ++i) {a_src('i', checked, kc)}")
            with cg.e.block(f"for (int j = 0; j < {tn}; ++j)"):
                cg.e.line(b_src("j", checked, "fb"))
                cg.e.line(f"for (int i = 0; i < {tm}; ++i) simdgroup_multiply_accumulate("
                          f"{d}[i][j], fa[i], fb, {d}[i][j]);")  # fmt: skip
            return
        cg.e.line(f"simdgroup_matrix<{in_t}, 8, 8> fa[{tm}], fb[{tn}];")
        cg.e.line(f"for (int i = 0; i < {tm}; ++i) {a_src('i', checked, kc)}")
        cg.e.line(f"for (int j = 0; j < {tn}; ++j) {b_src('j', checked, 'fb[j]')}")
        cg.e.line(f"for (int i = 0; i < {tm}; ++i)")
        cg.e.line(f"  for (int j = 0; j < {tn}; ++j) simdgroup_multiply_accumulate("
                  f"{d}[i][j], fa[i], fb[j], {d}[i][j]);")  # fmt: skip

    def mma_loop(checked: bool) -> None:
        if ra is not None:
            # Register fragments stay in registers only under constant indices, and Metal
            # doesn't always unroll a 16-step loop: indexing them with `kk / 8` spilled
            # them to the stack and halved attention throughput at head dimension 128.
            for kc in range(bk // 8):
                with cg.e.block(""):
                    cg.e.line(f"const int kk = {kc * 8};")
                    mma_step(checked, kc)
            return
        # Small loops, as in the reference kernel: Metal unrolls them itself. Fully
        # unrolled straight-line loads and MMAs measured 12-15% slower.
        cg.e.line("#pragma unroll")
        with cg.e.block(f"for (int kk = 0; kk < {bk}; kk += 8)"):
            mma_step(checked, None)

    conds = [x for x in (_in_bounds(da, bm, bk), _in_bounds(db, bk, bn)) if x]
    if not conds:
        mma_loop(False)
    else:
        with cg.e.block(f"if ({' && '.join(conds)})"):
            mma_loop(False)
        with cg.e.block("else"):
            mma_loop(True)
    return out


# The largest K that one float32 partial sum of integer products can cover exactly,
# for each operand type that `simdgroup_matrix` takes as `half`: float32 holds every
# integer up to 2^24, and |a * b| is at most 2^14 for int8 and 255^2 for uint8.
_INT_MMA_CHUNK = {"i8": (1 << 24) // (1 << 14), "u8": (1 << 24) // (255 * 255) // 8 * 8}


def emit_int_dot(cg: _Codegen, op: ir.Op, lay: L.BitLayout) -> Tile:
    """Emits an integer `tl.dot` that accumulates exactly in int32.

    `simdgroup_matrix` has no integer types, so `int8` and `uint8` operands convert to
    `half`, which holds them exactly, and multiply-accumulate in float32 fragments. A
    float32 sum is exact while every partial sum stays within 2^24, so the fragments add
    into the int32 accumulator and restart from zero at least every `_INT_MMA_CHUNK`
    steps of K. Wider integer operands have products that float32 can't hold, so they
    run as scalar 32-bit multiply-adds from threadgroup memory. Both paths wrap modulo
    2^32, like Triton's int32 accumulation.
    """
    a, b, c = op.operands
    bm, bk = a.type.shape
    bn = b.type.shape[1]
    tm, tn, wm, wn = L.frag_grid(lay)
    elem = a.type.elem.name
    n = lay.num_regs

    acc_in = cg.mat(c, lay)
    if acc_in.name and acc_in.frag is None and acc_in.uniform is None and \
            cg.uses.get(id(c)) == 1 and _can_clobber(cg, c, op) and \
            op.result.type == c.type:  # fmt: skip
        out = acc_in
    else:
        out = Tile(lay, cg.declare(op.result.type, lay, op.result.name_hint or "acc"))
        cg._assign(out, acc_in, n)
    d = out.name

    mma = elem in _INT_MMA_CHUNK
    in_t = "half" if mma else CTYPES[elem]
    ra = _register_a(cg, a, bm, bk, wm, wn, in_t) if mma else None
    tg = _stage(cg, ([a] if ra is None else []) + [b], in_t)

    if not mma:
        # acc[r] += A[row(r), k] * B[k, col(r)], in uint so that overflow wraps.
        ta, lda = tg[id(a)]
        tb, ldb = tg[id(b)]
        rows = [cg.coord(lay, r, 0) for r in range(n)]
        cols = [cg.coord(lay, r, 1) for r in range(n)]
        with cg.e.block(f"for (int kk = 0; kk < {bk}; ++kk)"):
            for r in range(n):
                cg.e.line(f"{d}[{r}] = as_type<int>(as_type<uint>({d}[{r}]) + "
                          f"uint({ta}[({rows[r]}) * {lda} + kk]) * "
                          f"uint({tb}[kk * {ldb} + {cols[r]}]));")  # fmt: skip
        return out

    sm, sn = bm // wm, bn // wn
    lwm = wm.bit_length() - 1
    sr = cg.pro(f"sr|{wm}|{sm}", "sr", "int", f"int(warp & {wm - 1}u) * {sm}") if wm > 1 else "0"
    sc = cg.pro(f"sc|{lwm}|{sn}", "sc", "int", f"int(warp >> {lwm}) * {sn}") if wn > 1 else "0"
    fs = cg.fresh("fsum")
    part = Tile(lay, fs, frag=(tm, tn))
    cg.e.line(f"simdgroup_matrix<float, 8, 8> {fs}[{tm}][{tn}];")
    chunk = _INT_MMA_CHUNK[elem]
    for k0 in range(0, bk, chunk):
        k1 = min(bk, k0 + chunk)
        for i in range(tm):
            for j in range(tn):
                cg.e.line(f"{fs}[{i}][{j}] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);")
        if ra is not None:
            # Register fragments need constant indices (see `emit_dot`).
            steps = [(str(kc * 8), kc) for kc in range(k0 // 8, k1 // 8)]
        else:
            steps = [("kk", None)]
            cg.e.line("#pragma unroll")
        for kk, kc in steps:
            head = f"for (int kk = {k0}; kk < {k1}; kk += 8)" if kc is None else ""
            with cg.e.block(head):
                cg.e.line(f"simdgroup_matrix<half, 8, 8> fa[{tm}], fb[{tn}];")
                if ra is not None:
                    cg.e.line(f"for (int i = 0; i < {tm}; ++i) fa[i] = {ra}[i][{kc}];")
                else:
                    name, ld = tg[id(a)]
                    cg.e.line(f"for (int i = 0; i < {tm}; ++i) simdgroup_load(fa[i], {name} + "
                              f"({_add(sr, 'i * 8')}) * {ld} + {kk}, {ld});")  # fmt: skip
                name, ld = tg[id(b)]
                cg.e.line(f"for (int j = 0; j < {tn}; ++j) simdgroup_load(fb[j], {name} + "
                          f"{kk} * {ld} + {_add(sc, 'j * 8')}, {ld});")  # fmt: skip
                cg.e.line(f"for (int i = 0; i < {tm}; ++i)")
                cg.e.line(f"  for (int j = 0; j < {tn}; ++j) simdgroup_multiply_accumulate("
                          f"{fs}[i][j], fa[i], fb[j], {fs}[i][j]);")  # fmt: skip
        # Each partial sum is an exact integer, so the conversion to int is exact.
        for r in range(n):
            cg.e.line(f"{d}[{r}] = as_type<int>(as_type<uint>({d}[{r}]) + "
                      f"as_type<uint>(int({part.get(r)})));")  # fmt: skip
    return out


def _stage(cg: _Codegen, vals: list[ir.Value], in_t: str) -> dict[int, tuple[str, int]]:
    """Writes 2D dot operands to threadgroup memory as `in_t`, rows padded by 16 bytes.

    Returns {id(operand): (pointer name, row stride in elements)}.
    """
    if not vals:
        return {}
    eb = 2 if in_t in ("half", "short", "ushort") else 1 if in_t in ("char", "uchar") else 4
    pad = 16 // eb
    off, out, plan = 0, {}, []
    for v in vals:
        rows, cols = v.type.shape
        ld = cols + pad
        plan.append((v, off, ld))
        off += -(-rows * ld * eb // 16) * 16
    cg.use_tg(off)
    cg.e.line(BARRIER)
    for v, o, ld in plan:
        t = cg.mat(v, cg.plan.natural(v) or cg.plan.default(v.type))
        name = cg.fresh("tgop")
        out[id(v)] = (name, ld)
        cg.e.line(f"threadgroup {in_t}* {name} = (threadgroup {in_t}*)(tg_mem + {o});")
        flat, consts = cg.flat(t.layout, (ld, 1))
        with cg.e.block(f"if ({cg.owner(t.layout)})"):
            for r in range(t.layout.num_regs):
                cg.e.line(f"{name}[{_add(flat, consts[r])}] = {in_t}({t.get(r)});")
    cg.e.line(BARRIER)
    return out


def _register_a(cg: _Codegen, a: ir.Value, bm: int, bk: int, wm: int, wn: int,
                in_t: str) -> str | None:  # fmt: skip
    """Returns a `simdgroup_matrix` array [TM][BK / 8] that holds A, or None to stage A.

    A qualifies when it's cheap (rematerialized in the operand layout) or when its layout
    reaches the operand layout by a register remap. Anything else would need a
    threadgroup exchange, which costs as much as staging, so it's staged.
    """
    lay = L.dot_operand_a(bm, bk, wm, wn)
    nat = cg.plan.natural(a)
    if nat is not None and nat != lay and L.reg_map(nat, lay) is None:
        return None
    t = cg.mat(a, lay)
    tm, kc = bm // wm // 8, bk // 8
    if t.frag == (tm, kc) and cg._frag_elem.get(t.name) == in_t:
        return t.name
    name = cg.fresh("ra")
    cg.e.line(f"simdgroup_matrix<{in_t}, 8, 8> {name}[{tm}][{kc}];")
    lkc = kc.bit_length() - 1
    for r in range(lay.num_regs):
        i, j, e = r >> (1 + lkc), (r >> 1) & (kc - 1), r & 1
        cg.e.line(f"{name}[{i}][{j}].thread_elements()[{e}] = {in_t}({t.get(r)});")
    return name


def _can_clobber(cg: _Codegen, c: ir.Value, dot: ir.Op) -> bool:
    """Returns whether `dot` may overwrite the storage of its accumulator `c`.

    `c` must own its storage, so it can't be a view of another tile. The dot must also run
    at most once per definition of `c`: only `if` regions may sit between the block that
    defines `c` and the dot. A loop in between would reuse the updated value on the next
    iteration. The loop-carried `acc = tl.dot(a, b, acc)` pattern still qualifies, because
    `acc` is an argument of the loop body block that holds the dot.
    """
    if cg.plan.classify(c) != ANCHORED:
        return False
    home = c.owner if isinstance(c.owner, ir.Block) else c.owner.parent
    blk = dot.parent
    while blk is not home:
        region = blk.parent if blk is not None else None
        parent = region.parent if region is not None else None
        if parent is None or parent.name != "if":
            return False
        blk = parent.parent
    return True


def _in_bounds(dop: _Direct | None, rows: int, cols: int) -> str | None:
    """Returns the uniform condition that the whole logical rows x cols block is in bounds."""
    if dop is None:
        return None
    (o0, o1), (s0, s1) = dop.offsets, dop.desc.shape
    r0, r1 = (cols, rows) if dop.transposed else (rows, cols)
    return f"({o0} >= 0 && {o1} >= 0 && {o0} + {r0} <= {s0} && {o1} + {r1} <= {s1})"


def _direct_load(dop: _Direct, frag: str, row: str, col: str, checked: bool, fm: str,
                 fn: str) -> str:  # fmt: skip
    """Returns a fragment load of logical (row, col) of a direct operand block."""
    (o0, o1), (s0, s1), ld = dop.offsets, dop.desc.shape, dop.desc.strides[0]
    if dop.transposed:
        mrow, mcol = _add2(o0, col), _add2(o1, row)
        bounds = f"{mcol}, {s1}, {mrow}, {s0}"
    else:
        mrow, mcol = _add2(o0, row), _add2(o1, col)
        bounds = f"{mrow}, {s0}, {mcol}, {s1}"
    # Add each column term to the pointer separately: grouping them into one int sum
    # first, as in `p + r * ld + (c0 + j * 8)`, stops Metal from folding the constant
    # into the address and measured 17% slower.
    ptr = f"{dop.desc.base} + ({mrow}) * {ld} + {mcol}"
    t = "true" if dop.transposed else "false"
    if not checked:
        if not dop.transposed:
            return f"simdgroup_load({frag}, {ptr}, {ld});"
        return f"simdgroup_load({frag}, {ptr}, {ld}, ulong2(0, 0), true);"
    return f"tg_load_frag({frag}, {ptr}, {ld}, {bounds}, {t}, {fm}, {fn});"


def _add2(a: str, b: str) -> str:
    if b == "0":
        return a
    if a == "0":
        return b
    return f"{a} + {b}"


# ---- descriptor loads and stores outside direct dots ----


def _block_coords(cg: _Codegen, desc: DescInfo, offsets: list[str], lay: L.BitLayout, r: int):
    """Returns (index expressions per dim, in-bounds condition, element offset) for register r."""
    idx = [_add2(o, cg.coord(lay, r, d)) for d, o in enumerate(offsets)]
    cond = " && ".join(f"({x}) >= 0 && ({x}) < {s}" for x, s in zip(idx, desc.shape, strict=True))
    elem = " + ".join(f"({x}) * {st}" if st != "1" else f"({x})"
                      for x, st in zip(idx, desc.strides, strict=True))  # fmt: skip
    return idx, cond, elem


def emit_desc_load(cg: _Codegen, op: ir.Op, lay: L.BitLayout) -> Tile:
    desc = cg.descs[id(op.operands[0])]
    offsets = [cg.s(v) for v in op.operands[1:]]
    if L.frag_grid(lay) is not None and is_float(desc.elem):
        return _desc_load_frags(cg, desc, offsets, lay, op.result.name_hint)
    arr = cg.declare(op.result.type, lay, op.result.name_hint)
    zero = f"{CTYPES[desc.elem.name]}(0)"
    for r in range(lay.num_regs):
        _, cond, elem = _block_coords(cg, desc, offsets, lay, r)
        cg.e.line(f"{arr}[{r}] = ({cond}) ? {desc.base}[{elem}] : {zero};")
    return Tile(lay, arr)


def _desc_load_frags(cg: _Codegen, desc: DescInfo, offsets: list[str], lay: L.BitLayout,
                     hint: str | None) -> Tile:  # fmt: skip
    """Loads a descriptor block in an accumulator layout as `simdgroup_matrix` fragments.

    Each fragment that fits in the tensor loads with one `simdgroup_load`; fragments that
    straddle an edge load per lane and zero-fill.
    """
    tm, tn, wm, wn = L.frag_grid(lay)
    bm, bn = lay.shape
    sm, sn = bm // wm, bn // wn
    lwm = wm.bit_length() - 1
    sr = cg.pro(f"sr|{wm}|{sm}", "sr", "int", f"int(warp & {wm - 1}u) * {sm}") if wm > 1 else "0"
    sc = cg.pro(f"sc|{lwm}|{sn}", "sc", "int", f"int(warp >> {lwm}) * {sn}") if wn > 1 else "0"
    fm, fn = _frag_consts(cg)
    ct = CTYPES[desc.elem.name]
    name = cg.fresh(hint or "fr")
    cg._frag_elem[name] = ct
    cg.e.line(f"simdgroup_matrix<{ct}, 8, 8> {name}[{tm}][{tn}];")
    d = _Direct(desc, offsets, False)
    for i in range(tm):
        for j in range(tn):
            cg.e.line(_direct_load(d, f"{name}[{i}][{j}]", _add(sr, i * 8), _add(sc, j * 8),
                                   True, fm, fn))  # fmt: skip
    return Tile(lay, name, frag=(tm, tn))


def emit_desc_store(cg: _Codegen, op: ir.Op) -> None:
    desc = cg.descs[id(op.operands[0])]
    offsets = [cg.s(v) for v in op.operands[1:-1]]
    val = op.operands[-1]
    lay = cg.plan.natural(val) or cg.plan.default(val.type)
    t = cg.mat(val, lay)
    if desc.root:
        cg.written.add(desc.root)
    ct = CTYPES[desc.elem.name]
    own = cg.owner(lay)
    with cg.e.block(f"if ({own})"):
        for r in range(lay.num_regs):
            _, cond, elem = _block_coords(cg, desc, offsets, lay, r)
            cg.e.line(f"if ({cond}) {desc.base}[{elem}] = {ct}({t.get(r)});")
