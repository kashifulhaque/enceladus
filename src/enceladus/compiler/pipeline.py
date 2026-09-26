"""Runs the compiler passes in order and produces MSL."""

from __future__ import annotations

from enceladus.compiler import ir
from enceladus.compiler.codegen.msl import GeneratedKernel, generate
from enceladus.compiler.passes.layouts import assign_layouts
from enceladus.compiler.passes.simplify import simplify


def compile_module(module: ir.Module, max_threadgroup_memory: int = 32768) -> GeneratedKernel:
    """Lowers a verified module to MSL.

    The passes are: `simplify` (CSE and DCE), `axis_info` and `assign_layouts`, then MSL
    codegen, which also lowers layout conversions and reductions, allocates threadgroup
    memory, and places barriers. The module is modified in place.

    Raises:
        CompilationError: The kernel uses something the MSL backend can't lower.
    """
    simplify(module)
    if ir.verify_enabled():
        ir.verify(module)
    num_warps = int(module.attrs.get("num_warps", 4))
    plan = assign_layouts(module, num_warps, module.attrs.get("dot_warps"))
    return generate(module, plan, max_threadgroup_memory)
