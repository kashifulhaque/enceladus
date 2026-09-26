"""In-memory and on-disk caches for compiled kernels.

The in-memory cache maps MSL source text to Metal pipelines. The on-disk cache in
`~/.cache/tegula/<key-hash>/` stores the generated source, IR, and metadata; Metal's
own disk cache stores compiled binaries.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import threading
from functools import cache
from pathlib import Path
from typing import Any

from tegula import _C
from tegula.runtime.raw import compile_pipeline


def cache_dir() -> Path:
    """Returns the root of Tegula's disk cache (`TEGULA_CACHE_DIR` overrides it)."""
    root = os.environ.get("TEGULA_CACHE_DIR")
    return Path(root) if root else Path.home() / ".cache" / "tegula"


def env_flag(name: str) -> bool:
    return os.environ.get(name, "") not in ("", "0", "false", "False")


@cache
def os_build() -> str:
    """Returns the macOS version and build, for example "27.0 (26A123)"."""
    build = ""
    try:
        libc = ctypes.CDLL(None)
        size = ctypes.c_size_t(64)
        buf = ctypes.create_string_buffer(64)
        if libc.sysctlbyname(b"kern.osversion", buf, ctypes.byref(size), None, 0) == 0:
            build = buf.value.decode()
    except OSError:
        pass
    return f"{platform.mac_ver()[0]} ({build})"


def stable_hash(*parts: Any) -> str:
    """Returns a SHA-256 hex digest of the `repr` of `parts`, which must be deterministic."""
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode())
        h.update(b"\0")
    return h.hexdigest()


def write_entry(key: str, files: dict[str, str], meta: dict[str, Any]) -> Path:
    """Writes files for one compiled specialization and returns its directory."""
    d = cache_dir() / key[:32]
    d.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        tmp = d / (name + ".tmp")
        tmp.write_text(text)
        tmp.replace(d / name)
    (d / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    return d


def read_entry(key: str, name: str) -> str | None:
    """Returns the cached file `name` for `key`, or None."""
    if env_flag("TEGULA_ALWAYS_COMPILE"):
        return None
    p = cache_dir() / key[:32] / name
    try:
        return p.read_text()
    except OSError:
        return None


_pipelines: dict[tuple, _C.Pipeline] = {}
_lock = threading.Lock()


def get_pipeline(
    source: str,
    name: str,
    language_version: Any = None,
    math_mode: str = "relaxed",
    enable_logging: bool = False,
) -> _C.Pipeline:
    """Compiles `source` or returns the cached pipeline for identical inputs.

    Safe to call from several threads; compilation releases the GIL, so autotuning
    compiles candidates in parallel.
    """
    key = (hashlib.sha256(source.encode()).digest(), name, language_version, math_mode,
           enable_logging)
    p = _pipelines.get(key)
    if p is not None:
        return p
    p = compile_pipeline(source, name, language_version, math_mode, enable_logging=enable_logging)
    with _lock:
        return _pipelines.setdefault(key, p)
