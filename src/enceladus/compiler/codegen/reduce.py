"""Reduction lowering: registers, then SIMD-group shuffles, then one threadgroup exchange.

For `reduce(x, axis)`, the bases of `x`'s layout that point into `axis` say where the
reduced elements live:

1. Register bases: combine registers inside each thread.
2. Lane bases: `simd_shuffle_xor` over each such lane bit (or `simd_sum`, `simd_max`,
   `simd_min` when all five lane bits reduce).
3. SIMD-group bases: write per-SIMD-group partials to threadgroup memory, barrier, then
   every thread combines the partials it needs.

The result has layout `slice(x.layout, axis)`, or is a uniform scalar for a full
reduction. `float16` and `bfloat16` inputs accumulate in `float32`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from enceladus.compiler import ir
from enceladus.compiler import layout as L
from enceladus.compiler.codegen.msl import BARRIER, CTYPES, _add, _body_ops, is_float, is_half
from enceladus.compiler.errors import internal_error

if TYPE_CHECKING:
    from enceladus.compiler.codegen.msl import _Codegen

NATIVE_SIMD = {"float", "half", "int", "uint", "short", "ushort", "char", "uchar"}


def _combine_builtin(kind: str, is_fp: bool):
    """Returns f(acc_names, other_exprs) -> list of assignment statements."""
    if kind == "sum":
        return lambda a, b: [f"{a[0]} = {a[0]} + {b[0]};"]
    if kind in ("max", "min"):
        fn = ("f" if is_fp else "") + kind
        return lambda a, b: [f"{a[0]} = {fn}({a[0]}, {b[0]});"]
    cmp = ">" if kind == "argmax" else "<"

    def arg(a, b):
        # Take b when it's strictly better, or equal with a lower index (NumPy ties).
        take = f"({b[0]} {cmp} {a[0]} || ({b[0]} == {a[0]} && {b[1]} < {a[1]}))"
        return [
            f"{{ const bool tk = {take};",
            f"  {a[0]} = tk ? {b[0]} : {a[0]}; {a[1]} = tk ? {b[1]} : {a[1]}; }}",
        ]

    return arg


def emit_reduce(cg: _Codegen, op: ir.Op) -> None:
    axis = op.attrs["axis"]
    inputs = list(op.operands)
    region = op.regions[0].block if op.regions else None
    kind = op.attrs.get("kind")
    x0 = inputs[0]
    lay = cg.plan.natural(x0) or cg.plan.default(x0.type)
    tiles = [cg.mat(v, lay) for v in inputs]
    in_elems = [ir.elem_of(v.type) for v in inputs]
    results = op.results

    # Accumulator components and their types.
    if region is not None:
        acc_types = [CTYPES[e.name] for e in in_elems]
    else:
        e = in_elems[0]
        vt = "float" if is_half(e) else CTYPES[e.name]
        acc_types = [vt, "int"] if kind in ("argmax", "argmin") else [vt]
    ncomp = len(acc_types)

    # Group input registers by their coordinates outside `axis`.
    rank = len(lay.shape)
    groups: dict[tuple[int, ...], list[int]] = {}
    for r in range(lay.num_regs):
        c = L.coords_of(lay, r)
        groups.setdefault(c[:axis] + c[axis + 1 :], []).append(r)
    if isinstance(results[0].type, ir.TileType):
        rlay = cg.plan.layout_of(results[0])
        order = [L.coords_of(rlay, r) for r in range(rlay.num_regs)]
        if sorted(order) != sorted(groups):
            raise internal_error("reduction result layout doesn't match its input", cg.loc)
    else:
        rlay = None
        order = list(groups)

    def value(i: int, r: int) -> str:
        v = tiles[i].get(r)
        if region is None and i == 0 and is_half(in_elems[0]):
            return f"float({v})"
        return v

    # Combine functions.
    if region is not None:
        combine = _region_combiner(cg, region, acc_types)
    else:
        combine = _combine_builtin(kind, is_float(in_elems[0]))

    def emit_combine(accs: list[str], others: list[str]) -> None:
        for line in combine(accs, others):
            cg.e.line(line)

    # 1. Registers.
    accs: list[list[str]] = []
    for key in order:
        regs = groups[key]
        names = [cg.fresh("acc") for _ in range(ncomp)]
        accs.append(names)
        for c in range(ncomp):
            if region is None and c == 1:
                init = cg.coord(lay, regs[0], axis)
            else:
                init = value(c, regs[0])
            cg.e.line(f"{acc_types[c]} {names[c]} = {init};")
        for r in regs[1:]:
            others = [cg.coord(lay, r, axis) if (region is None and c == 1) else value(c, r)
                      for c in range(ncomp)]  # fmt: skip
            emit_combine(names, others)

    # 2. Lanes.
    lane_bits = [i for i, b in enumerate(lay.lane) if b[axis]]
    if (
        len(lane_bits) == 5
        and region is None
        and kind in ("sum", "max", "min")
        and acc_types[0] in NATIVE_SIMD
    ):
        for names in accs:
            cg.e.line(f"{names[0]} = simd_{kind}({names[0]});")
    else:
        for bit in lane_bits:
            for names in accs:
                tmp = [cg.fresh("sh") for _ in range(ncomp)]
                for c in range(ncomp):
                    shfl = f"tg_shfl_xor({names[c]}, ushort({1 << bit}))"
                    cg.e.line(f"const {acc_types[c]} {tmp[c]} = {shfl};")
                emit_combine(names, tmp)

    # 3. SIMD groups.
    warp_bits = [i for i, b in enumerate(lay.warp) if b[axis]]
    if warp_bits:
        wa = len(warp_bits)
        rn = 1 if rlay is None else rlay_numel(rlay)
        parts = [f"(((warp >> {j}) & 1u) << {k})" for k, j in enumerate(warp_bits)]
        widx = cg.pro(f"widx|{warp_bits}", "wi", "int", "int(" + " | ".join(parts) + ")")
        lane_mask = sum(1 << i for i in lane_bits)
        writer = cg.pro(f"lw|{lane_mask}", "lw", "bool", f"(lane & {lane_mask}u) == 0") \
            if lane_mask else "true"
        # Flat result index per group.
        if rlay is None:
            flat_expr, flat_c = "0", [0] * len(order)
        else:
            strides, acc = [], 1
            for d in reversed(range(rank - 1)):
                strides.append(acc)
                acc *= rlay.shape[d]
            flat_expr, flat_c = cg.flat(rlay, tuple(reversed(strides)))
        offsets, total = [], 0
        for t in acc_types:
            offsets.append(total)
            total += -(-((1 << wa) * rn * _size(t)) // 16) * 16
        cg.use_tg(total)
        cg.e.line(BARRIER)
        with cg.e.block(""):
            bufs = [cg.fresh("red") for _ in acc_types]
            for b, t, off in zip(bufs, acc_types, offsets, strict=True):
                cg.e.line(f"threadgroup {t}* {b} = (threadgroup {t}*)(tg_mem + {off});")
            with cg.e.block(f"if ({writer})"):
                for g, names in enumerate(accs):
                    idx = _add(flat_expr, flat_c[g])
                    for b, n in zip(bufs, names, strict=True):
                        cg.e.line(f"{b}[{widx} * {rn} + {idx}] = {n};")
            cg.e.line(BARRIER)
            for g, names in enumerate(accs):
                idx = _add(flat_expr, flat_c[g])
                for b, n in zip(bufs, names, strict=True):
                    cg.e.line(f"{n} = {b}[{idx}];")
                for w in range(1, 1 << wa):
                    emit_combine(names, [f"{b}[{w * rn} + {idx}]" for b in bufs])

    # Results.
    if region is None:
        out_elem = ir.elem_of(results[0].type)
        comp = 1 if kind in ("argmax", "argmin") else 0
        _bind_result(cg, results[0], rlay, [names[comp] for names in accs], out_elem)
    else:
        for c, res in enumerate(results):
            _bind_result(cg, res, rlay, [names[c] for names in accs], ir.elem_of(res.type))


def rlay_numel(lay: L.BitLayout) -> int:
    n = 1
    for s in lay.shape:
        n *= s
    return n


def _size(t: str) -> int:
    return {"float": 4, "int": 4, "uint": 4, "half": 2, "bfloat": 2, "short": 2, "ushort": 2,
            "char": 1, "uchar": 1, "bool": 1, "long": 8, "ulong": 8}[t]  # fmt: skip


def _bind_result(cg, res: ir.Value, rlay, names: list[str], elem: ir.ScalarType) -> None:
    ct = CTYPES[elem.name]
    if rlay is None:
        cg.bind_scalar(res, f"{ct}({names[0]})")
        return
    from enceladus.compiler.codegen.msl import Tile

    arr = cg.declare(res.type, rlay, res.name_hint or "red")
    for r, n in enumerate(names):
        cg.e.line(f"{arr}[{r}] = {ct}({n});")
    cg.tiles[id(res)] = Tile(rlay, arr)


def _region_combiner(cg: _Codegen, region: ir.Block, acc_types: list[str]):
    """Returns a combiner that inlines a `tl.reduce` combine region."""
    n = len(acc_types)

    def combine(accs: list[str], others: list[str]) -> list[str]:
        # Emit into a nested block; the region's ops become scalar code.
        cg.e.line("{")
        with cg.e.indented():
            a_copies = []
            for t, a in zip(acc_types, accs, strict=True):
                c = cg.fresh("ca")
                cg.e.line(f"const {t} {c} = {a};")
                a_copies.append(c)
            for arg, expr in zip(region.args, a_copies + list(others), strict=True):
                cg.sv[id(arg)] = expr
            cg.push_scope()
            cg.block(_body_ops(region))
            ys = [cg.s(y) for y in region.ops[-1].operands]
            cg.pop_scope()
            for a, y in zip(accs, ys, strict=True):
                cg.e.line(f"{a} = {y};")
        cg.e.line("}")
        return []

    assert len(region.args) == 2 * n
    return combine
