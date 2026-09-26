"""`kernel.explain`: a readable report of the compiler's decisions for one kernel."""

from __future__ import annotations

import os
from typing import Any

from enceladus.compiler import ir
from enceladus.compiler import layout as L
from enceladus.compiler.codegen.dot import find_direct_operands
from enceladus.compiler.codegen.msl import GeneratedKernel, offset_type
from enceladus.compiler.errors import Loc
from enceladus.compiler.passes.layouts import ANCHORED, LayoutPlan, dot_warps
from enceladus.compiler.pipeline import compile_module


def describe_layout(lay: L.BitLayout) -> str:
    """Summarizes how a layout spreads a tile over registers, lanes, and SIMD groups.

    For example, `regs 1x4, lanes 4x8, warps 4x1` means each thread holds 4 elements
    along dimension 1, the 32 lanes of a SIMD group cover a 4 x 8 block of those, and 4
    SIMD groups stack along dimension 0.
    """
    rank = len(lay.shape)

    def split(bases: tuple[L.Basis, ...]) -> tuple[list[int], int]:
        counts, zeros = [1] * rank, 0
        for b in bases:
            dims = [d for d, v in enumerate(b) if v]
            if dims:
                counts[dims[0]] *= 2
            else:
                zeros += 1
        return counts, zeros

    def x(counts: list[int]) -> str:
        return "x".join(map(str, counts)) if counts else "1"

    reg, _ = split(lay.reg)
    lane, lz = split(lay.lane)
    warp, wz = split(lay.warp)
    kind = "simdgroup_matrix fragments" if L.frag_grid(lay) is not None else "blocked"
    out = f"{kind}: regs {x(reg)}, lanes {x(lane)}, warps {x(warp)}"
    if lz or wz:
        out += f", each element held by {1 << (lz + wz)} threads"
    return out


def register_estimate(t: ir.TileType, lay: L.BitLayout, offset_bytes: int = 4) -> int:
    """Returns the 32-bit registers per thread that a tile needs, as codegen counts them."""
    e = ir.elem_of(t)
    nbytes = offset_bytes if isinstance(e, ir.PointerType) else max(1, e.dtype.itemsize)
    return lay.num_regs * max(1, nbytes // 4) if nbytes >= 4 else lay.num_regs


def _where(loc: Loc | None) -> str:
    return f"{os.path.basename(loc.file)}:{loc.line}" if loc is not None else "?"


def _source(loc: Loc | None) -> str:
    text = loc.source_line() if loc is not None else None
    return f"  `{text.strip()}`" if text else ""


def _type(t: ir.Type) -> str:
    e = ir.elem_of(t)
    name = f"*{e.elem}" if isinstance(e, ir.PointerType) else str(e)
    shape = ir.shape_of(t)
    return f"{name}[{', '.join(map(str, shape))}]" if shape else name


def _tile_values(module: ir.Module, plan: LayoutPlan) -> list[tuple[ir.Value, ir.Op, str]]:
    """Returns (value, op, role) for every tile with a fixed layout, in program order."""
    out = []
    for op in module.walk():
        for rg in op.regions:
            if op.name == "for":
                for a in rg.block.args[1:]:
                    if isinstance(a.type, ir.TileType) and id(a) in plan.fixed:
                        out.append((a, op, "loop-carried"))
        for r in op.results:
            if isinstance(r.type, ir.TileType) and plan.classify(r) == ANCHORED \
                    and id(r) in plan.fixed and op.name != "for":  # fmt: skip
                out.append((r, op, op.name))
    return out


def explain_kernel(module: ir.Module, max_threadgroup_memory: int = 32768,
                   grid: tuple[int, int, int] | None = None) -> str:  # fmt: skip
    """Compiles `module` to MSL and returns the report text. Modifies `module`."""
    gen: GeneratedKernel = compile_module(module, max_threadgroup_memory)
    plan: LayoutPlan = gen.plan
    lines: list[str] = []
    add = lines.append
    nw = gen.num_warps
    head = f"Kernel `{gen.name}`: num_warps={nw} ({nw * 32} threads per program)"
    if grid is not None:
        head += f", grid {grid}"
    add(head)
    lv = gen.language_version
    extras = []
    if gen.enable_logging:
        extras.append("shader logging on")
    if gen.asserts:
        extras.append(f"{len(gen.asserts)} device asserts")
    add(f"MSL language version {lv[0]}.{lv[1]}" + (f" ({', '.join(extras)})" if extras else ""))

    dots = [op for op in module.walk() if op.name == "dot"]
    backend = module.attrs.get("dot_backend", "simdgroup")
    if not dots:
        add("dot backend: none (the kernel has no tl.dot)")
    else:
        add(f"dot backend: {backend}")
        for op in dots:
            (bm, bk), bn = op.operands[0].type.shape, op.result.type.shape[1]
            wm, wn = dot_warps(bm, bn, nw, op.loc, plan.dot_warps)
            add(f"  {_where(op.loc)}  {bm}x{bn}x{bk} (MxNxK), SIMD-group grid {wm}x{wn}"
                f"{_source(op.loc)}")  # fmt: skip
        for reason in gen.dot_fallbacks:  # dots that dot_backend="mpp" couldn't lower
            add(f"  uses simdgroup instead of mpp: {reason}")

    add("")
    add("Tiles (layout; registers per thread):")
    offset_bytes = 8 if offset_type(module) == "long" else 4
    tiles = _tile_values(module, plan)
    direct = find_direct_operands(module, plan)
    if not tiles:
        add("  (none)")
    for v, op, role in tiles:
        lay = plan.layout_of(v)
        name = f" `{v.name_hint}`" if v.name_hint else ""
        loc = op.loc
        if id(v) in direct:
            what = "read by tl.dot straight from device memory; 0 registers"
        else:
            regs = register_estimate(v.type, lay, offset_bytes)
            what = f"{describe_layout(lay)}; {regs} registers"
        add(f"  {_where(loc)}  {role}{name} {_type(v.type)}: {what}{_source(loc)}")

    add("")
    add("Layout conversions:")
    convs = gen.report.get("conversions", [])
    if not convs:
        add("  (none)")
    for c in convs:
        v: ir.Value = c["value"]
        name = f" `{v.name_hint}`" if v.name_hint else ""
        how = ("register moves, no memory traffic" if c["kind"] == "registers"
               else f"threadgroup memory exchange, {c['bytes']} bytes, 2 barriers")  # fmt: skip
        frm = f" (defined at {_where(c['value_loc'])})" if c["value_loc"] is not None else ""
        add(f"  {_where(c['loc'])}  {_type(v.type)}{name}{frm}: {how}{_source(c['loc'])}")
        add(f"      from {describe_layout(c['src'])}")
        add(f"      to   {describe_layout(c['dst'])}")

    add("")
    add(f"Threadgroup memory: {gen.threadgroup_memory} bytes of {max_threadgroup_memory} "
        "(one arena that each operation reuses)")  # fmt: skip
    uses = gen.report.get("threadgroup", [])
    seen: set[tuple[Any, ...]] = set()
    for u in uses:
        key = (u["loc"], u["op"], u["bytes"])
        if key in seen:
            continue
        seen.add(key)
        add(f"  {_where(u['loc'])}  {u['op']}: {u['bytes']} bytes{_source(u['loc'])}")
    if gen.warnings:
        add("")
        add("Warnings:")
        for w in gen.warnings:
            add(f"  {w}")
    return "\n".join(lines)

