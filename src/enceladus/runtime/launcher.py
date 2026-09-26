"""Compiled kernels and the native launch path."""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from enceladus import _C
from enceladus.compiler.codegen.msl import KernelArg, scalar_slots
from enceladus.runtime.device import get_device
from enceladus.runtime.interop import as_kernel_arg
from enceladus.runtime.tensor import Tensor

MAX_GRID = (1 << 32) - 1


@dataclass
class CompiledKernel:
    """One specialization of a kernel, compiled and ready to launch.

    Attributes:
        name: The MSL kernel name.
        msl: The generated (or overridden) MSL source.
        ir: The IR after the compiler passes, as text.
        args: The runtime arguments in binding order.
        num_warps: SIMD groups per threadgroup.
        threadgroup_memory_bytes: Threadgroup memory the kernel declares.
        cache_dir: The on-disk cache entry, if one was written.
    """

    name: str
    msl: str
    ir: str
    args: list[KernelArg]
    num_warps: int
    threadgroup_memory_bytes: int
    pipeline: Any = None
    cache_dir: str | None = None
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._ptr_idx = [i for i, a in enumerate(self.args) if a.is_pointer]
        self._scalar_idx = [i for i, a in enumerate(self.args) if not a.is_pointer]
        self._packer = struct.Struct("<" + "".join(self.args[i].struct_format
                                                   for i in self._scalar_idx))  # fmt: skip
        self._plan = _C.LaunchPlan([self.args[i].index for i in self._ptr_idx],
                                   scalar_slots(self.args))  # fmt: skip
        self._tg = (self.num_warps * 32, 1, 1)

    def launch(self, grid: tuple[int, int, int], values: Sequence[Any]) -> None:
        """Launches with runtime argument values in signature order."""
        if grid[0] == 0 or grid[1] == 0 or grid[2] == 0:
            return
        if grid[0] > MAX_GRID or grid[1] > MAX_GRID or grid[2] > MAX_GRID:
            raise ValueError(f"grid {grid} exceeds the device limit of {MAX_GRID} per dimension")
        stream = get_device().stream
        bufs, offsets = [], []
        sync = False
        for i in self._ptr_idx:
            a = values[i]
            if type(a) is Tensor:
                bufs.append(a.buffer)
                offsets.append(a.offset * a.np_dtype.itemsize if a.offset else 0)
                continue
            ba = as_kernel_arg(a)
            bufs.append(ba.buffer)
            offsets.append(ba.byte_offset)
            if ba.needs_sync:
                sync = True
            else:
                stream.keep_alive(ba.owner)
            if ba.writeback is not None:
                stream.after_sync(ba.writeback)
        scalars = self._packer.pack(*[values[i] for i in self._scalar_idx])
        stream.native.dispatch(self.pipeline, self._plan, bufs, offsets, scalars, grid, self._tg)
        if sync:
            stream.synchronize()

    def timed_launch(self, grid: tuple[int, int, int], values: Sequence[Any]) -> float:
        """Runs one launch in its own command buffer and returns its GPU time in seconds."""
        bufs = [as_kernel_arg(values[i]) for i in self._ptr_idx]
        scalars = self._packer.pack(*[values[i] for i in self._scalar_idx])
        t0, t1 = get_device().stream.native.timed_run(
            self.pipeline, self._plan, [b.buffer for b in bufs], [b.byte_offset for b in bufs],
            scalars, grid, self._tg,
        )  # fmt: skip
        return t1 - t0
