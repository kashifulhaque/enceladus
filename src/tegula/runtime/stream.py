"""The batching stream: launches append to an open command buffer.

The stream commits its command buffer every `flush_every` dispatches (64 by default,
`TEGULA_FLUSH_EVERY` overrides it) and whenever the host needs results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tegula import _C

if TYPE_CHECKING:
    from tegula.runtime.device import Device

# Keep-alive lists longer than this force a sync, which bounds memory held by
# asynchronous launches over host memory.
_MAX_KEEPALIVE = 4096


class Stream:
    """An ordered queue of GPU work, like a CUDA stream."""

    def __init__(self, device: Device, flush_every: int = 64) -> None:
        self.device = device
        self.native = _C.Stream(device.queue, flush_every)
        # Objects whose memory the GPU may still use; cleared at each sync.
        self._keepalive: list[Any] = []
        # Callbacks that run after the next sync, such as copy-back of host memory.
        self._after_sync: list[Any] = []

    @property
    def pending(self) -> int:
        """The number of dispatches in the open command buffer."""
        return self.native.pending

    @property
    def flush_every(self) -> int:
        return self.native.flush_every

    @flush_every.setter
    def flush_every(self, value: int) -> None:
        self.native.flush_every = value

    def keep_alive(self, obj: Any) -> None:
        """Holds a reference to `obj` until the next sync."""
        self._keepalive.append(obj)
        if len(self._keepalive) > _MAX_KEEPALIVE:
            self.synchronize()

    def after_sync(self, fn) -> None:
        """Runs `fn()` after the next sync completes."""
        self._after_sync.append(fn)

    def flush(self) -> None:
        """Commits the open command buffer without waiting."""
        self.native.flush()

    def synchronize(self) -> None:
        """Commits pending work and waits for all of it to finish.

        Raises:
            tegula.MetalError: A command buffer since the previous sync failed.
        """
        try:
            self.native.sync()
        finally:
            callbacks, self._after_sync = self._after_sync, []
            self._keepalive.clear()
        for fn in callbacks:
            fn()


def synchronize() -> None:
    """Waits for all work launched on the default stream to finish."""
    from tegula.runtime.device import _device

    if _device is not None:
        _device.stream.synchronize()
