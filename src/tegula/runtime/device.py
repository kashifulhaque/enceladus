"""The Metal device singleton and its capabilities."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

from tegula import _C

PAGE_SIZE = 16384


@dataclass(frozen=True)
class Capabilities:
    """Static facts about a Metal device that affect compilation and launch."""

    name: str
    architecture: str
    apple_families: tuple[int, ...]
    metal3: bool
    metal4: bool
    max_threadgroup_memory: int
    max_threads_per_threadgroup: int
    max_buffer_length: int
    recommended_max_working_set: int
    max_concurrent_compilations: int

    def supports_apple(self, family: int) -> bool:
        """Returns whether the device supports `MTLGPUFamilyApple<family>`."""
        return family in self.apple_families

    @property
    def apple_family(self) -> int:
        """The highest supported Apple GPU family, for example 9 on M4."""
        return max(self.apple_families, default=0)


class Device:
    """A Metal device with its command queue and default stream."""

    def __init__(self) -> None:
        from tegula.runtime.stream import Stream

        self.native = _C.default_device()
        self.queue = _C.new_queue(self.native)
        info = self.native.query()
        self.caps = Capabilities(
            name=self.native.name,
            architecture=self.native.architecture,
            apple_families=tuple(info["apple_families"]),
            metal3=info["metal3"],
            metal4=info["metal4"],
            max_threadgroup_memory=info["max_threadgroup_memory"],
            max_threads_per_threadgroup=info["max_threads_per_threadgroup"],
            max_buffer_length=info["max_buffer_length"],
            recommended_max_working_set=info["recommended_max_working_set"],
            max_concurrent_compilations=info["max_concurrent_compilations"],
        )
        flush_every = int(os.environ.get("TEGULA_FLUSH_EVERY", "64"))
        self.stream = Stream(self, flush_every=flush_every)

    def __repr__(self) -> str:
        return f"<tegula.Device {self.caps.name} ({self.caps.architecture})>"


_device: Device | None = None
_lock = threading.Lock()


def get_device() -> Device:
    """Returns the process-wide default device, creating it on first use."""
    if _device is not None:
        return _device
    return _create_device()


def _create_device() -> Device:
    global _device
    with _lock:
        if _device is None:
            _device = Device()
    return _device


def current_stream():
    """Returns the default stream of the default device."""
    return get_device().stream
