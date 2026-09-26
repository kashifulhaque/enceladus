"""`tl.dot` on `simdgroup_matrix`, and tensor descriptor loads and stores.

Each SIMD group owns an SM x SN strip of the accumulator, made of TM x TN 8x8
fragments (`layout.simd_acc`). For each K step of 8, it loads TM fragments of A and TN
fragments of B, then runs TM x TN `simdgroup_multiply_accumulate` calls.

Operands come from one of two sources:

- *Direct*: a `desc_load` (optionally through `trans`) used only by the `dot`, from a
  descriptor whose innermost stride is 1. Fragments load straight from device memory;
  no threadgroup memory, no barriers. This matches `best_matmul.metal`.
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
from enceladus.compiler.codegen.msl import BARRIER, CTYPES, Tile, _add

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


def find_direct_operands(module: ir.Module) -> set[int]:
    """Returns ids of `desc_load` and `trans` values that dots read from device memory."""
    uses = use_counts(module)
    out: set[int] = set()
    for op in module.walk():
        if op.name != "dot":
            continue
        for v in op.operands[:2]:
            src = v.defining_op
            if src is not None and src.name == "trans" and uses.get(id(v)) == 1:
                inner = src.operands[0].defining_op
                if inner is not None and inner.name == "desc_load" and \
                        uses.get(id(src.operands[0])) == 1:  # fmt: skip
                    out.update((id(v), id(src.operands[0])))
            elif src is not None and src.name == "desc_load" and uses.get(id(v)) == 1:
                out.add(id(v))
    return out


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

    # Accumulator: start from `c`, in place when this dot is its only use.
    acc_in = cg.mat(c, lay)
    if acc_in.frag == (tm, tn) and cg.uses.get(id(c)) == 1 and \
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
    # Staged operands go to threadgroup memory: A as BM x BK, B as BK x BN.
    eb = a.type.elem.dtype.itemsize
    pad = 16 // eb
    lda, ldb = bk + pad, bn + pad
    staged = []
    off = 0
    if da is None:
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

    def a_src(i: str, checked: bool) -> str:
        """Returns the statement that loads A fragment i at K offset kk."""
        if da is None:
            name, ld = tg_names[id(a)]
            return f"simdgroup_load(fa[{i}], {name} + ({_add(sr, i + ' * 8')}) * {ld} + kk, {ld});"
        return _direct_load(da, f"fa[{i}]", _add(sr, f"{i} * 8"), "kk", checked, fm, fn)

    def b_src(j: str, checked: bool) -> str:
        if db is None:
            name, ld = tg_names[id(b)]
            return f"simdgroup_load(fb[{j}], {name} + kk * {ld} + {_add(sc, j + ' * 8')}, {ld});"
        return _direct_load(db, f"fb[{j}]", "kk", _add(sc, f"{j} * 8"), checked, fm, fn)

    def mma_loop(checked: bool) -> None:
        # Small loops, as in the reference kernel: Metal unrolls them itself. Fully
        # unrolled straight-line loads and MMAs measured 12-15% slower.
        cg.e.line("#pragma unroll")
        with cg.e.block(f"for (int kk = 0; kk < {bk}; kk += 8)"):
            cg.e.line(f"simdgroup_matrix<{in_t}, 8, 8> fa[{tm}], fb[{tn}];")
            cg.e.line(f"for (int i = 0; i < {tm}; ++i) {a_src('i', checked)}")
            cg.e.line(f"for (int j = 0; j < {tn}; ++j) {b_src('j', checked)}")
            cg.e.line(f"for (int i = 0; i < {tm}; ++i)")
            cg.e.line(f"  for (int j = 0; j < {tn}; ++j) simdgroup_multiply_accumulate("
                      f"{d}[i][j], fa[i], fb[j], {d}[i][j]);")  # fmt: skip

    conds = [x for x in (_in_bounds(da, bm, bk), _in_bounds(db, bk, bn)) if x]
    if not conds:
        mma_loop(False)
    else:
        with cg.e.block(f"if ({' && '.join(conds)})"):
            mma_loop(False)
        with cg.e.block("else"):
            mma_loop(True)
    return out


def _in_bounds(dop: _Direct | None, rows: int, cols: int) -> str | None:
    """Returns the uniform condition that the whole logical rows x cols block is in bounds."""
    if dop is None:
        return None
    (o0, o1), (s0, s1) = dop.offsets, dop.desc.shape
    r0, r1 = (cols, rows) if dop.transposed else (rows, cols)
    return f"({o0} + {r0} <= {s0} && {o1} + {r1} <= {s1})"


def _direct_load(dop: _Direct, frag: str, row: str, col: str, checked: bool, fm: str,
                 fn: str) -> str:  # fmt: skip
    """Returns a fragment load of logical (row, col) of a direct operand block."""
    (o0, o1), (s0, s1), ld = dop.offsets, dop.desc.shape, dop.desc.strides[0]
    if dop.transposed:
        mrow, mcol = _add2(o0, col), _add2(o1, row)
        rows_left, cols_left = f"{s1} - ({mcol})", f"{s0} - ({mrow})"
    else:
        mrow, mcol = _add2(o0, row), _add2(o1, col)
        rows_left, cols_left = f"{s0} - ({mrow})", f"{s1} - ({mcol})"
    # Add each column term to the pointer separately: grouping them into one int sum
    # first, as in `p + r * ld + (c0 + j * 8)`, stops Metal from folding the constant
    # into the address and measured 17% slower.
    ptr = f"{dop.desc.base} + ({mrow}) * {ld} + {mcol}"
    t = "true" if dop.transposed else "false"
    if not checked:
        if not dop.transposed:
            return f"simdgroup_load({frag}, {ptr}, {ld});"
        return f"simdgroup_load({frag}, {ptr}, {ld}, ulong2(0, 0), true);"
    return f"tg_load_frag({frag}, {ptr}, {ld}, {rows_left}, {cols_left}, {t}, {fm}, {fn});"


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
    cond = " && ".join(f"({x}) < {s}" for x, s in zip(idx, desc.shape, strict=True))
    elem = " + ".join(f"({x}) * {st}" if st != "1" else f"({x})"
                      for x, st in zip(idx, desc.strides, strict=True))  # fmt: skip
    return idx, cond, elem


def emit_desc_load(cg: _Codegen, op: ir.Op, lay: L.BitLayout) -> Tile:
    desc = cg.descs[id(op.operands[0])]
    offsets = [cg.s(v) for v in op.operands[1:]]
    arr = cg.declare(op.result.type, lay, op.result.name_hint)
    zero = f"{CTYPES[desc.elem.name]}(0)"
    for r in range(lay.num_regs):
        _, cond, elem = _block_coords(cg, desc, offsets, lay, r)
        cg.e.line(f"{arr}[{r}] = ({cond}) ? {desc.base}[{elem}] : {zero};")
    return Tile(lay, arr)


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
