"""Compile one kernel specialization: IR, passes, MSL, and a Metal pipeline.

The disk cache stores the generated MSL, the IR, and argument metadata, so a later
process skips Enceladus's own compiler. Metal's disk cache covers the binary.

Environment variables:
    ENCELADUS_ALWAYS_COMPILE: Ignore Enceladus's caches.
    ENCELADUS_DUMP: Write the IR and MSL to the cache entry and print their paths.
    ENCELADUS_OVERRIDE_DIR: Load `<dir>/<kernel_name>.metal` instead of the generated MSL.
    ENCELADUS_DEBUG: Keep `tl.device_assert` checks. The flag is part of the cache key.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import enceladus
from enceladus.compiler.codegen.msl import KernelArg
from enceladus.compiler.frontend import build_ir
from enceladus.compiler.pipeline import compile_module
from enceladus.language import core
from enceladus.runtime import cache
from enceladus.runtime.device import check_simdgroup_layout, get_device
from enceladus.runtime.launcher import CompiledKernel

if TYPE_CHECKING:
    from enceladus.runtime.jit import JITFunction, Specialization


def _key(fn: JITFunction, spec: Specialization, num_warps: int, dot_warps,
         debug: bool = False) -> str:  # fmt: skip
    caps = get_device().caps
    parts = (fn.cache_key, spec.key(num_warps, fn.math_mode), dot_warps, enceladus.__version__,
             cache.compiler_hash(), cache.os_build(), caps.architecture)  # fmt: skip
    return cache.stable_hash(*parts, "debug") if debug else cache.stable_hash(*parts)


def build_module(fn: JITFunction, spec: Specialization, num_warps: int,
                 dot_warps: tuple[int, int] | None = None, debug: bool | None = None):  # fmt: skip
    """Returns the verified IR module for one specialization, with its module attributes."""
    module = build_ir(fn, spec.arg_types, spec.arg_facts, spec.constexprs, num_warps,
                      fn.math_mode, debug=debug)  # fmt: skip
    if dot_warps is not None:
        module.attrs["dot_warps"] = tuple(dot_warps)
    module.attrs["apple_family"] = get_device().caps.apple_family
    return module


def compile_specialization(fn: JITFunction, spec: Specialization, num_warps: int,
                           dot_warps: tuple[int, int] | None = None):  # fmt: skip
    """Returns a launchable `CompiledKernel` for one specialization."""
    dev = get_device()
    debug = core.debug_enabled()
    key = _key(fn, spec, num_warps, dot_warps, debug)
    dump = cache.env_flag("ENCELADUS_DUMP")
    ck = None
    meta_text = cache.read_entry(key, "meta.json")
    msl = cache.read_entry(key, "kernel.metal")
    if meta_text and msl and not dump:
        meta = json.loads(meta_text)
        ck = CompiledKernel(
            meta["name"], msl, cache.read_entry(key, "ir.txt") or "",
            [KernelArg(**a) for a in meta["args"]], meta["num_warps"],
            meta["threadgroup_memory_bytes"],
            language_version=tuple(meta.get("language_version", (3, 2))),
            enable_logging=meta.get("enable_logging", False),
            asserts=meta.get("asserts", []),
            assert_buffer_index=meta.get("assert_buffer_index"),
        )  # fmt: skip
    if ck is None:
        module = build_module(fn, spec, num_warps, dot_warps, debug)
        gen = compile_module(module, dev.caps.max_threadgroup_memory)
        ck = CompiledKernel(gen.name, gen.source, str(module), gen.args, gen.num_warps,
                            gen.threadgroup_memory, warnings=gen.warnings,
                            language_version=gen.language_version,
                            enable_logging=gen.enable_logging, asserts=gen.asserts,
                            assert_buffer_index=gen.assert_buffer_index)  # fmt: skip
        meta = {
            "name": ck.name,
            "args": [dataclasses.asdict(a) for a in ck.args],
            "num_warps": ck.num_warps,
            "threadgroup_memory_bytes": ck.threadgroup_memory_bytes,
            "kernel": fn.__qualname__,
            "specialization": spec.key(num_warps, fn.math_mode),
            "language_version": list(ck.language_version),
            "enable_logging": ck.enable_logging,
            "asserts": ck.asserts,
            "assert_buffer_index": ck.assert_buffer_index,
        }
        try:
            d = cache.write_entry(key, {"kernel.metal": ck.msl, "ir.txt": ck.ir}, meta)
            ck.cache_dir = str(d)
        except OSError:
            pass
        for w in gen.warnings:
            print(f"enceladus warning: {w}", file=sys.stderr)
    override = os.environ.get("ENCELADUS_OVERRIDE_DIR")
    if override:
        p = Path(override) / f"{fn.__name__}.metal"
        if p.exists():
            ck.msl = p.read_text()
    if dump:
        where = ck.cache_dir or "(not cached)"
        print(f"enceladus: {fn.__name__}: IR and MSL in {where}", file=sys.stderr)
    if "simdgroup_multiply_accumulate" in ck.msl:
        check_simdgroup_layout()
    ck.math_mode = fn.math_mode
    ck.pipeline = cache.get_pipeline(ck.msl, ck.name, ck.language_version, fn.math_mode,
                                     ck.enable_logging)  # fmt: skip
    return ck
