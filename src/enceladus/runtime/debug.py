"""GPU capture: record Enceladus's GPU work into a trace that Xcode opens."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

from enceladus import _C

_active = False


def capture_active() -> bool:
    """Returns whether `enceladus.capture` is recording."""
    return _active


@contextlib.contextmanager
def capture(path: str | os.PathLike[str]) -> Iterator[str]:
    """Records all GPU work inside the block into a GPU trace document.

    Open the resulting `.gputrace` bundle in Xcode to inspect each dispatch, its
    buffers, and its shader. Metal allows capture only when the process starts with
    `MTL_CAPTURE_ENABLED=1` in its environment, for example
    `MTL_CAPTURE_ENABLED=1 uv run python script.py`.

    Printing is incompatible with capture. Metal refuses to capture once the process
    has created a command queue that collects shader logs, which Enceladus does at the
    first launch of a kernel that calls `tl.device_print`. Capture in a process that
    doesn't print; a printing kernel that launches inside the block raises
    `RuntimeError`.

    Example:

        with enceladus.capture("add.gputrace"):
            add_kernel[grid](x, y, out, n, BLOCK=1024)

    Args:
        path: Where to write the trace. It must end in `.gputrace` and must not exist.

    Yields:
        The trace path as a string.

    Raises:
        RuntimeError: `MTL_CAPTURE_ENABLED=1` isn't set, a capture is already running, or
            a kernel that calls `tl.device_print` has run in this process.
        FileExistsError: `path` already exists.
        ValueError: `path` doesn't end in `.gputrace`.
        enceladus.MetalError: Metal refused to start the capture.
    """
    global _active
    path = os.fspath(path)
    if os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        raise RuntimeError(
            "enceladus.capture() needs MTL_CAPTURE_ENABLED=1 in the environment before the "
            "process starts, because Metal loads its capture layer only then. Run, for "
            "example, `MTL_CAPTURE_ENABLED=1 uv run python script.py`."
        )
    if not path.endswith(".gputrace"):
        raise ValueError(f"the capture path must end in .gputrace, but got {path!r}")
    if os.path.exists(path):
        raise FileExistsError(f"{path} already exists; remove it or choose another path")
    if _active:
        raise RuntimeError("enceladus.capture() is already recording; captures can't nest")
    from enceladus.runtime.device import get_device

    dev = get_device()
    if dev.stream.log_queue is not None:
        raise RuntimeError(
            "enceladus.capture() can't record in this process, because a kernel that calls "
            "tl.device_print already ran, and Metal refuses to capture once shader logging "
            "is set up. Capture in a separate process that doesn't print."
        )
    dev.stream.synchronize()
    _C.capture_start(dev.native, path)
    _active = True
    try:
        yield path
    finally:
        try:
            dev.stream.synchronize()
        finally:
            _C.capture_stop()
            _active = False
