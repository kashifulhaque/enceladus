"""`enceladus.metal_kernel`: launch hand-written MSL, the expert escape hatch."""

from __future__ import annotations

import struct
from typing import Any

import numpy as np

from enceladus import _C
from enceladus.runtime import interop
from enceladus.runtime.device import get_device
from enceladus.runtime.interop import as_kernel_arg, is_array_like
from enceladus.runtime.tensor import Tensor

MATH_MODES = {"safe": 0, "relaxed": 1, "fast": 2}

# MTLDataType raw value -> struct format for scalar arguments.
_SCALAR_FORMATS = {
    3: "f",  # float
    16: "e",  # half
    29: "i",  # int
    33: "I",  # uint
    37: "h",  # short
    41: "H",  # ushort
    45: "b",  # char
    49: "B",  # uchar
    53: "?",  # bool
    81: "q",  # long
    85: "Q",  # ulong
}
_MTL_BFLOAT = 121
# MTLDataType raw value -> IR scalar name, for the PyTorch launch path.
_SCALAR_TYPES = {
    3: "f32", 16: "f16", 29: "i32", 33: "u32", 37: "i16", 41: "u16", 45: "i8", 49: "u8",
    53: "i1", 81: "i64", 85: "u64", _MTL_BFLOAT: "bf16",
}  # fmt: skip


def language_version_code(version: str | tuple[int, int] | int | None) -> int:
    """Returns the raw `MTLLanguageVersion` for "3.2", (3, 2), or a raw value."""
    if version is None:
        return (3 << 16) + 2
    if isinstance(version, int):
        return version
    if isinstance(version, str):
        major, minor = (int(p) for p in version.split("."))
    else:
        major, minor = version
    return (major << 16) + minor


def compile_pipeline(
    source: str,
    name: str,
    language_version: Any = None,
    math_mode: str = "relaxed",
    math_fp32_functions: str = "fast",
    enable_logging: bool = False,
) -> _C.Pipeline:
    """Compiles MSL source at run time and returns the pipeline for kernel `name`.

    Raises:
        enceladus.MetalError: The source doesn't compile or has no kernel `name`.
    """
    if math_mode not in MATH_MODES:
        raise ValueError(f"math_mode must be one of {sorted(MATH_MODES)}, not {math_mode!r}")
    dev = get_device()
    lib, _warnings = _C.compile(
        dev.native,
        source,
        language_version=language_version_code(language_version),
        math_mode=MATH_MODES[math_mode],
        math_fp32_functions=0 if math_fp32_functions == "fast" else 1,
        enable_logging=enable_logging,
    )
    return _C.pipeline(dev.native, lib, name)


def _dim3(v: Any, what: str) -> tuple[int, int, int]:
    if isinstance(v, (int, np.integer)):
        v = (int(v),)
    v = tuple(int(x) for x in v)
    if not 1 <= len(v) <= 3 or any(x < 0 for x in v):
        raise ValueError(f"{what} must be 1 to 3 non-negative integers, got {v}")
    return v + (1,) * (3 - len(v))


class MetalKernel:
    """A compiled hand-written MSL kernel.

    Launch it with `kernel[grid, threads_per_group](*args)`. `grid` counts
    threadgroups, not threads. Argument `i` binds at `[[buffer(i)]]`: array-likes
    bind as buffers, and Python numbers bind as scalars of the type the kernel
    declares.
    """

    def __init__(
        self,
        source: str,
        name: str,
        language_version: Any = None,
        math_mode: str = "relaxed",
    ) -> None:
        self.name = name
        self.source = source
        self.pipeline = compile_pipeline(source, name, language_version, math_mode)
        bindings = sorted(
            (b for b in self.pipeline.bindings() if b["kind"] == "buffer"),
            key=lambda b: b["index"],
        )
        if [b["index"] for b in bindings] != list(range(len(bindings))):
            raise ValueError(
                f"kernel '{name}' must use consecutive buffer indices starting at 0"
            )
        self._bindings = bindings
        self._plans: dict[tuple[bool, ...], _C.LaunchPlan] = {}
        self._max_threads = self.pipeline.max_total_threads_per_threadgroup
        self.math_mode = math_mode
        self._torch: dict[tuple[bool, ...], Any] = {}  # TorchLaunch or a reason string
        self._fallback_logged = False

    @property
    def arg_names(self) -> list[str]:
        return [b["name"] for b in self._bindings]

    def __getitem__(self, key: Any) -> _Launcher:
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError("launch a raw kernel with kernel[grid, threads_per_group](*args)")
        grid, tg = _dim3(key[0], "grid"), _dim3(key[1], "threads_per_group")
        if tg[0] * tg[1] * tg[2] > self._max_threads:
            raise ValueError(
                f"threads_per_group {tg} exceeds the pipeline limit of {self._max_threads}"
            )
        return _Launcher(self, grid, tg)

    def _plan(self, mask: tuple[bool, ...]) -> _C.LaunchPlan:
        plan = self._plans.get(mask)
        if plan is None:
            buf_index, scalars, off = [], [], 0
            for i, is_buf in enumerate(mask):
                if is_buf:
                    buf_index.append(i)
                else:
                    size = self._bindings[i]["data_size"]
                    scalars.append((i, off, size))
                    off += size
            plan = self._plans[mask] = _C.LaunchPlan(buf_index, scalars)
        return plan

    def _pack(self, i: int, value: Any) -> bytes:
        b = self._bindings[i]
        fmt = _SCALAR_FORMATS.get(b["data_type"])
        if fmt is not None:
            return struct.pack("<" + fmt, value)
        if b["data_type"] == _MTL_BFLOAT:
            import ml_dtypes

            return np.array(value, dtype=ml_dtypes.bfloat16).tobytes()
        raise TypeError(
            f"argument {i} ('{b['name']}') of kernel '{self.name}' has a type that "
            "Enceladus can't pack from a Python value; pass an array instead"
        )

    def launch(self, grid: tuple, tg: tuple, args: tuple) -> None:
        if len(args) != len(self._bindings):
            raise TypeError(
                f"kernel '{self.name}' takes {len(self._bindings)} arguments "
                f"({', '.join(self.arg_names)}), but {len(args)} were given"
            )
        stream = get_device().stream
        bufs, offsets, scalar_parts, mask = [], [], [], []
        sync = False
        for i, a in enumerate(args):
            if type(a) is Tensor:  # fast path for the common case
                bufs.append(a.buffer)
                offsets.append(a.offset * a.np_dtype.itemsize if a.offset else 0)
                mask.append(True)
            elif is_array_like(a):
                if interop.framework_of(a) is not None:
                    self._launch_foreign(grid, tg, args)
                    return
                ba = as_kernel_arg(a)
                bufs.append(ba.buffer)
                offsets.append(ba.byte_offset)
                mask.append(True)
                if ba.needs_sync:
                    sync = True
                else:
                    stream.keep_alive(ba.owner)
                if ba.writeback is not None:
                    stream.after_sync(ba.writeback)
            else:
                scalar_parts.append(self._pack(i, a))
                mask.append(False)
        mask = tuple(mask)
        plan = self._plans.get(mask) or self._plan(mask)
        stream.native.dispatch(
            self.pipeline, plan, bufs, offsets, b"".join(scalar_parts), grid, tg
        )
        if sync:
            stream.synchronize()


    def _launch_foreign(self, grid: tuple, tg: tuple, args: tuple) -> None:
        """Launches with at least one PyTorch or MLX array argument.

        When every array argument is a PyTorch MPS tensor, the kernel runs on PyTorch's
        stream through `torch.mps.compile_shader`. Otherwise, the launch synchronizes.
        """
        # launcher imports the compiler, which imports this module through cache.
        from enceladus.runtime.launcher import (
            TorchLaunch,
            launch_synced,
            log_fallback,
            make_torch_launch,
        )

        mask = tuple(is_array_like(a) for a in args)
        kinds = {interop.framework_of(a) for a, m in zip(args, mask, strict=True) if m}
        reason = None
        if kinds == {interop.KIND_TORCH}:
            for a in args:
                if interop.framework_of(a) is not None:
                    interop.torch_np_dtype(a)  # refuses CPU and float64 tensors
            tl = self._torch.get(mask)
            if tl is None:
                types = [None if m else _SCALAR_TYPES.get(b["data_type"], "?")
                         for m, b in zip(mask, self._bindings, strict=True)]  # fmt: skip
                tl = self._torch[mask] = make_torch_launch(self.source, self.name,
                                                           self.math_mode, types)  # fmt: skip
            if isinstance(tl, TorchLaunch):
                threads = (grid[0] * tg[0], grid[1] * tg[1], grid[2] * tg[2])
                tl.launch(args, threads, tg)
                return
            reason = f"torch.mps.compile_shader can't run it ({tl})"
        elif interop.KIND_TORCH in kinds:
            reason = "it mixes PyTorch tensors with other array types"
        if reason is not None and not self._fallback_logged:
            self._fallback_logged = True
            log_fallback(self.name, reason)
        scalars = b"".join(self._pack(i, a) for i, a in enumerate(args) if not mask[i])
        launch_synced(get_device().stream, self.pipeline, self._plans.get(mask) or
                      self._plan(mask), [a for a, m in zip(args, mask, strict=True) if m],
                      scalars, grid, tg)  # fmt: skip


class _Launcher:
    __slots__ = ("kernel", "grid", "tg")

    def __init__(self, kernel: MetalKernel, grid: tuple, tg: tuple) -> None:
        self.kernel, self.grid, self.tg = kernel, grid, tg

    def __call__(self, *args: Any) -> None:
        self.kernel.launch(self.grid, self.tg, args)


def metal_kernel(
    source: str, name: str, language_version: Any = None, math_mode: str = "relaxed"
) -> MetalKernel:
    """Compiles hand-written MSL and returns a launchable kernel.

    Args:
        source: Complete MSL source, including `#include <metal_stdlib>`.
        name: The name of the `kernel` function to launch.
        language_version: The MSL version, such as "3.2" (the default) or "4.0".
        math_mode: "safe", "relaxed" (the default), or "fast".

    Returns:
        A kernel that you launch with `kernel[grid, threads_per_group](*args)`.
    """
    return MetalKernel(source, name, language_version, math_mode)
