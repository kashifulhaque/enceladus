"""Compile one kernel specialization: IR, passes, MSL, and a Metal pipeline.

The disk cache stores the generated MSL, the IR, and argument metadata, so a later
process skips Tegula's own compiler. Metal's disk cache covers the binary.

Environment variables:
    TEGULA_ALWAYS_COMPILE: Ignore Tegula's caches.
    TEGULA_DUMP: Write the IR and MSL to the cache entry and print their paths.
    TEGULA_OVERRIDE_DIR: Load `<dir>/<kernel_name>.metal` instead of the generated MSL.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import tegula
from tegula.compiler.codegen.msl import KernelArg
from tegula.compiler.frontend import build_ir
from tegula.compiler.pipeline import compile_module
from tegula.runtime import cache
from tegula.runtime.device import check_simdgroup_layout, get_device
from tegula.runtime.launcher import CompiledKernel

if TYPE_CHECKING:
    from tegula.runtime.jit import JITFunction, Specialization


def _key(fn: JITFunction, spec: Specialization, num_warps: int, dot_warps) -> str:
    caps = get_device().caps
    return cache.stable_hash(
        fn.cache_key, spec.key(num_warps, fn.math_mode), dot_warps, tegula.__version__,
        cache.compiler_hash(), cache.os_build(), caps.architecture,
    )  # fmt: skip


def compile_specialization(fn: JITFunction, spec: Specialization, num_warps: int,
                           dot_warps: tuple[int, int] | None = None):  # fmt: skip
    """Returns a launchable `CompiledKernel` for one specialization."""
    dev = get_device()
    key = _key(fn, spec, num_warps, dot_warps)
    dump = cache.env_flag("TEGULA_DUMP")
    ck = None
    meta_text = cache.read_entry(key, "meta.json")
    msl = cache.read_entry(key, "kernel.metal")
    if meta_text and msl and not dump:
        meta = json.loads(meta_text)
        ck = CompiledKernel(
            meta["name"], msl, cache.read_entry(key, "ir.txt") or "",
            [KernelArg(**a) for a in meta["args"]], meta["num_warps"],
            meta["threadgroup_memory_bytes"],
        )  # fmt: skip
    if ck is None:
        module = build_ir(fn, spec.arg_types, spec.arg_facts, spec.constexprs, num_warps,
                          fn.math_mode)  # fmt: skip
        if dot_warps is not None:
            module.attrs["dot_warps"] = tuple(dot_warps)
        gen = compile_module(module, dev.caps.max_threadgroup_memory)
        ck = CompiledKernel(gen.name, gen.source, str(module), gen.args, gen.num_warps,
                            gen.threadgroup_memory, warnings=gen.warnings)  # fmt: skip
        meta = {
            "name": ck.name,
            "args": [dataclasses.asdict(a) for a in ck.args],
            "num_warps": ck.num_warps,
            "threadgroup_memory_bytes": ck.threadgroup_memory_bytes,
            "kernel": fn.__qualname__,
            "specialization": spec.key(num_warps, fn.math_mode),
        }
        try:
            d = cache.write_entry(key, {"kernel.metal": ck.msl, "ir.txt": ck.ir}, meta)
            ck.cache_dir = str(d)
        except OSError:
            pass
        for w in gen.warnings:
            print(f"tegula warning: {w}", file=sys.stderr)
    override = os.environ.get("TEGULA_OVERRIDE_DIR")
    if override:
        p = Path(override) / f"{fn.__name__}.metal"
        if p.exists():
            ck.msl = p.read_text()
    if dump:
        where = ck.cache_dir or "(not cached)"
        print(f"tegula: {fn.__name__}: IR and MSL in {where}", file=sys.stderr)
    if "simdgroup_multiply_accumulate" in ck.msl:
        check_simdgroup_layout()
    ck.pipeline = cache.get_pipeline(ck.msl, ck.name, None, fn.math_mode)
    return ck
