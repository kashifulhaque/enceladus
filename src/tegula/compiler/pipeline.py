"""Runs the compiler passes in order and produces MSL."""

from __future__ import annotations

from tegula.compiler import ir
from tegula.compiler.codegen.msl import GeneratedKernel, generate
from tegula.compiler.passes.layouts import assign_layouts
from tegula.compiler.passes.simplify import simplify


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
    plan = assign_layouts(module, num_warps)
    return generate(module, plan, max_threadgroup_memory)
