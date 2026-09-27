"""In-memory and on-disk caches for compiled kernels.

The in-memory cache maps MSL source text to Metal pipelines. The on-disk cache in
`~/.cache/enceladus/<key-hash>/` stores the generated source, IR, and metadata; Metal's
own disk cache stores compiled binaries.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import json
import os
import platform
import tempfile
import threading
from functools import cache
from pathlib import Path
from typing import Any

from enceladus import _C
from enceladus.runtime.raw import compile_pipeline


def cache_dir() -> Path:
    """Returns the root of Enceladus's disk cache (`ENCELADUS_CACHE_DIR` overrides it)."""
    root = os.environ.get("ENCELADUS_CACHE_DIR")
    return Path(root) if root else Path.home() / ".cache" / "enceladus"


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


# The files that decide the generated MSL, relative to the `enceladus` package: the
# compiler, the language surface, and the runtime code that builds specializations and
# module attributes.
_COMPILER_GLOBS = ("compiler/**/*.py", "compiler/**/*.metal", "language/**/*.py")
_COMPILER_FILES = ("runtime/compile.py", "runtime/jit.py", "runtime/dot_backend.py")


@cache
def compiler_hash() -> str:
    """Returns a hash of the compiler's own source, so compiler changes invalidate caches."""
    root = Path(__file__).resolve().parents[1]
    files = {f for g in _COMPILER_GLOBS for f in root.glob(g)}
    files |= {root / f for f in _COMPILER_FILES}
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(f.relative_to(root).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def stable_hash(*parts: Any) -> str:
    """Returns a SHA-256 hex digest of the `repr` of `parts`, which must be deterministic."""
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode())
        h.update(b"\0")
    return h.hexdigest()


def write_atomic(path: Path, text: str) -> None:
    """Writes `text` to `path` so that readers see either the old or the new contents.

    The temporary file has a unique name, so concurrent writers from several processes
    or threads don't clobber each other's partial files.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def write_entry(key: str, files: dict[str, str], meta: dict[str, Any]) -> Path:
    """Writes files for one compiled specialization and returns its directory.

    `meta.json` goes last, so an entry whose `meta.json` exists has all its files.
    """
    d = cache_dir() / key[:32]
    d.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        write_atomic(d / name, text)
    write_atomic(d / "meta.json", json.dumps(meta, indent=1, sort_keys=True))
    return d


def read_entry(key: str, name: str) -> str | None:
    """Returns the cached file `name` for `key`, or None."""
    if env_flag("ENCELADUS_ALWAYS_COMPILE"):
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
