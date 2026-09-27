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
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import enceladus
from enceladus.compiler.codegen.msl import KernelArg
from enceladus.compiler.frontend import build_ir
from enceladus.compiler.pipeline import compile_module
from enceladus.language import core
from enceladus.runtime import cache, dot_backend
from enceladus.runtime.device import check_simdgroup_layout, get_device
from enceladus.runtime.launcher import CompiledKernel

if TYPE_CHECKING:
    from enceladus.runtime.jit import JITFunction, Specialization

log = logging.getLogger("enceladus")


def _key(fn: JITFunction, spec: Specialization, num_warps: int, dot_warps,
         backend: str = "simdgroup", debug: bool = False) -> str:  # fmt: skip
    caps = get_device().caps
    parts = (fn.cache_key, spec.key(num_warps, fn.math_mode), dot_warps, enceladus.__version__,
             cache.compiler_hash(), cache.os_build(), caps.architecture, backend)  # fmt: skip
    if not debug:
        return cache.stable_hash(*parts)
    # A debug build stores the file and line of each tl.device_assert, so a kernel that
    # moved within its file, or to another file, must not reuse an entry with stale lines.
    return cache.stable_hash(*parts, "debug", *_source_locations(fn))


def _source_locations(fn: JITFunction) -> list[str]:
    """Returns `file:first_line` of `fn` and of every @enceladus.jit function it reaches."""
    fn.cache_key  # noqa: B018 - computing the key finds the reachable functions
    out = []
    for f in (fn, *fn._jit_deps):
        src = f.source_info()
        out.append(f"{src.file}:{src.first_line}")
    return out


def _from_cache_entry(key: str, meta_text: str, msl: str) -> CompiledKernel | None:
    """Returns the kernel that a disk cache entry describes, or `None` if the entry is
    corrupt or incomplete, which the caller treats as a miss and overwrites."""
    try:
        meta = json.loads(meta_text)
        ck = CompiledKernel(
            meta["name"], msl, cache.read_entry(key, "ir.txt") or "",
            [KernelArg(**a) for a in meta["args"]], meta["num_warps"],
            meta["threadgroup_memory_bytes"],
            language_version=tuple(meta.get("language_version", (3, 2))),
            enable_logging=meta.get("enable_logging", False),
            asserts=meta.get("asserts", []),
            assert_buffer_index=meta.get("assert_buffer_index"),
            dot_backend=meta.get("dot_backend"),
            dot_fallbacks=list(meta.get("dot_fallbacks", [])),
        )  # fmt: skip
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        log.debug("enceladus: ignoring the corrupt cache entry %s: %s", key[:32], e)
        return None
    # Codegen logs each MPP fallback; log them again when the disk cache skips codegen.
    for reason in ck.dot_fallbacks:
        log.debug("enceladus: %s: a tl.dot uses the simdgroup backend (cached): %s",
                  ck.name, reason)  # fmt: skip
    return ck


def build_module(fn: JITFunction, spec: Specialization, num_warps: int,
                 dot_warps: tuple[int, int] | None = None, debug: bool | None = None,
                 backend: str = "simdgroup"):  # fmt: skip
    """Returns the verified IR module for one specialization, with its module attributes.

    `backend` is the resolved `tl.dot` backend, "simdgroup" or "mpp". Codegen resets the
    `dot_backend` attribute to "simdgroup" when no `tl.dot` of the kernel can use MPP.
    """
    module = build_ir(fn, spec.arg_types, spec.arg_facts, spec.constexprs, num_warps,
                      fn.math_mode, debug=debug)  # fmt: skip
    if dot_warps is not None:
        module.attrs["dot_warps"] = tuple(dot_warps)
    module.attrs["apple_family"] = get_device().caps.apple_family
    if backend == "mpp":
        module.attrs["dot_backend"] = backend
    return module


def compile_specialization(fn: JITFunction, spec: Specialization, num_warps: int,
                           dot_warps: tuple[int, int] | None = None,
                           dot_backend_option: str = "auto"):  # fmt: skip
    """Returns a launchable `CompiledKernel` for one specialization."""
    dev = get_device()
    backend = dot_backend.resolve(dot_backend_option)
    debug = core.debug_enabled()
    key = _key(fn, spec, num_warps, dot_warps, backend, debug)
    dump = cache.env_flag("ENCELADUS_DUMP")
    ck = None
    meta_text = cache.read_entry(key, "meta.json")
    msl = cache.read_entry(key, "kernel.metal")
    if meta_text and msl and not dump:
        ck = _from_cache_entry(key, meta_text, msl)
    if ck is None:
        module = build_module(fn, spec, num_warps, dot_warps, debug, backend)
        gen = compile_module(module, dev.caps.max_threadgroup_memory)
        ck = CompiledKernel(gen.name, gen.source, str(module), gen.args, gen.num_warps,
                            gen.threadgroup_memory, warnings=gen.warnings,
                            language_version=gen.language_version,
                            enable_logging=gen.enable_logging, asserts=gen.asserts,
                            assert_buffer_index=gen.assert_buffer_index,
                            dot_backend=gen.dot_backend,
                            dot_fallbacks=gen.dot_fallbacks)  # fmt: skip
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
            "dot_backend": gen.dot_backend,
            "dot_fallbacks": gen.dot_fallbacks,
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
