"""Compiled kernels and their launch paths.

A launch takes one of three paths:

- The native path encodes the dispatch into Enceladus's batching stream. Launches with
  `enceladus.Tensor` and NumPy arguments use it.
- The PyTorch path runs the same MSL through `torch.mps.compile_shader`, so the kernel
  runs on PyTorch's MPS stream and orders with the surrounding PyTorch operations.
  Launches whose array arguments are all PyTorch MPS tensors use it.
- The synchronized native path serves every other launch that involves PyTorch or MLX
  memory, such as a mix of PyTorch tensors and `enceladus.Tensor` objects, or MLX arrays.
  It waits for PyTorch's stream (MLX arrays are evaluated), dispatches on Enceladus's
  stream, and waits for that stream. That costs about 100 µs per launch.
"""

from __future__ import annotations

import hashlib
import logging
import struct
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from enceladus import _C
from enceladus.compiler.codegen.msl import KernelArg, scalar_slots
from enceladus.runtime import interop
from enceladus.runtime.device import get_device
from enceladus.runtime.interop import as_kernel_arg
from enceladus.runtime.tensor import Tensor

MAX_GRID = (1 << 32) - 1
# The MSL language version that `torch.mps.compile_shader` compiles with (torch 2.14).
TORCH_LANGUAGE_VERSION = (4, 0)

log = logging.getLogger("enceladus")

# ---- PyTorch launch path ----

# SHA-256 of the MSL source -> the library that `torch.mps.compile_shader` returned.
_torch_libs: dict[bytes, Any] = {}

# `arg_casts` names that `torch.mps.compile_shader` accepts for Python ints. It binds an
# uncast int as int64 and a float as float32.
_INT_CASTS = {8: "int8", 16: "int16", 32: "int32"}
_INT_BITS = {"i8": 8, "i16": 16, "i32": 32, "i64": 64, "u8": 8, "u16": 16, "u32": 32, "u64": 64}


def _int_converter(bits: int, signed: bool) -> Callable[[Any], int]:
    """Returns a function that range-checks an int and wraps unsigned values to signed.

    `compile_shader` only binds signed casts (and uint8), so a uint16, uint32, or uint64
    argument binds as the signed type of the same width with the same bit pattern.
    """
    lo, hi = (-(1 << (bits - 1)), 1 << (bits - 1)) if signed else (0, 1 << bits)
    half, full = 1 << (bits - 1), 1 << bits

    def convert(v: Any) -> int:
        v = int(v)
        if not lo <= v < hi:
            raise OverflowError(f"{v} is out of range for a {bits}-bit argument")
        return v - full if (not signed and bits > 8 and v >= half) else v

    return convert


def _float_tensor_converter(dtype_name: str) -> Callable[[Any], Any]:
    """Returns a function that makes a 0-d CPU tensor, which binds with `setBytes`.

    `compile_shader` can't cast a Python float to half or bfloat, but it binds a 0-d CPU
    tensor by value with its own element size.
    """
    import torch

    dt = getattr(torch, dtype_name)
    return lambda v: torch.tensor(float(v), dtype=dt)


class TorchLaunch:
    """Launches one kernel through `torch.mps.compile_shader`.

    Argument `i` binds at `[[buffer(i)]]`, as it does on the native path: tensors bind
    at their storage offset, and scalars bind with the type the kernel declares.

    Raises:
        RuntimeError: `compile_shader` rejected the source.
    """

    def __init__(self, source: str, name: str, math_mode: str,
                 arg_types: Sequence[str | None], checked: bool = False) -> None:  # fmt: skip
        """Compiles `source` with `compile_shader` and prepares argument conversion.

        Args:
            source: The complete MSL source.
            name: The kernel function to launch.
            math_mode: "safe", "relaxed", or "fast".
            arg_types: For each argument in binding order, `None` for a buffer or the IR
                scalar type name, such as "i32".
            checked: Whether callers pass only in-range signed ints, bools, and floats,
                as `@enceladus.jit` specialization does, so conversion can skip range
                checks.
        """
        import torch

        # compile_shader compiles with safe math; match Enceladus's math mode instead.
        src = f"#pragma METAL fp math_mode({math_mode})\n{source}"
        key = hashlib.sha256(src.encode()).digest()
        lib = _torch_libs.get(key)
        if lib is None:
            lib = _torch_libs[key] = torch.mps.compile_shader(src)
        self.fn = getattr(lib, name)
        self.max_threads = self.fn.max_threads_per_threadgroup
        casts: dict[int, str] = {}
        convs: list[tuple[int, Callable[[Any], Any]]] = []
        for i, t in enumerate(arg_types):
            if t is None:
                continue
            if t == "i1":
                casts[i] = "int8"
                convs.append((i, int if checked else (lambda v: 1 if v else 0)))
            elif checked and t in ("i32", "i64"):
                if t == "i32":
                    casts[i] = "int32"
                convs.append((i, int))
            elif t in _INT_BITS:
                bits = _INT_BITS[t]
                if bits in _INT_CASTS:
                    casts[i] = "uint8" if t == "u8" else _INT_CASTS[bits]
                convs.append((i, _int_converter(bits, t[0] == "i")))
            elif t == "f32":
                convs.append((i, float))
            elif t == "f16":
                convs.append((i, _float_tensor_converter("float16")))
            elif t == "bf16":
                convs.append((i, _float_tensor_converter("bfloat16")))
            else:
                raise TypeError(f"argument {i} has type {t}, which the PyTorch path can't bind")
        # Generate the call once, like JITFunction's binder: a per-launch loop over the
        # arguments costs more than the rest of this path.
        ns: dict[str, Any] = {"fn": self.fn, "casts": casts or None}
        call = [f"v[{i}]" for i in range(len(arg_types))]
        for i, conv in convs:
            ns[f"c{i}"] = conv
            call[i] = f"c{i}(v[{i}])"
        call += ["threads=threads", "group_size=group", "arg_casts=casts"]
        src = f"def launch(v, threads, group):\n    fn({', '.join(call)})\n"
        exec(src, ns)  # noqa: S102 - the source is built from argument indices only
        self.launch: Callable[[Sequence[Any], tuple, tuple], None] = ns["launch"]
        """Launches with values in binding order, `threads` total threads in groups of
        `group` threads."""


def make_torch_launch(source: str, name: str, math_mode: str, arg_types: Sequence[str | None],
                      checked: bool = False) -> TorchLaunch | str:  # fmt: skip
    """Returns a `TorchLaunch`, or the reason that the kernel can't use the PyTorch path."""
    try:
        return TorchLaunch(source, name, math_mode, arg_types, checked)
    except Exception as e:  # noqa: BLE001 - any failure selects the fallback path
        first = str(e).strip().splitlines()
        return f"{type(e).__name__}: {first[0] if first else ''}"


def log_fallback(name: str, reason: str) -> None:
    """Logs, once per kernel, that a launch takes the synchronized native path."""
    log.warning(
        "enceladus: kernel `%s` runs on Enceladus's own queue with a sync before and after "
        "each launch (about 100 µs per launch): %s", name, reason,
    )  # fmt: skip


def launch_synced(stream: Any, pipeline: Any, plan: Any, bufs_values: Sequence[Any],
                  scalars: bytes, grid: tuple, tg: tuple) -> None:  # fmt: skip
    """Dispatches on Enceladus's stream between syncs with PyTorch's stream.

    Waits for PyTorch's pending work, binds every array argument (evaluating MLX arrays),
    dispatches, and waits until the GPU finishes, so PyTorch and MLX read the results.
    """
    interop.torch_synchronize()
    bufs, offsets = [], []
    for a in bufs_values:
        ba = as_kernel_arg(a)
        bufs.append(ba.buffer)
        offsets.append(ba.byte_offset)
        stream.keep_alive(ba.owner)
        if ba.writeback is not None:
            stream.after_sync(ba.writeback)
    stream.native.dispatch(pipeline, plan, bufs, offsets, scalars, grid, tg)
    stream.synchronize()


# ---- Compiled kernels ----


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
        math_mode: The Metal math mode the kernel compiles with.
        language_version: The MSL language version the kernel needs, or None for the
            default (3.2).
        dot_backend: The `tl.dot` backend the kernel uses ("mpp" or "simdgroup"), or None
            if it has no `tl.dot`.
        dot_fallbacks: Why each `tl.dot` that `dot_backend="mpp"` asked for uses
            `simdgroup` instead.
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
    math_mode: str = "relaxed"
    language_version: tuple[int, int] | None = None
    dot_backend: str | None = None
    dot_fallbacks: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._ptr_idx = [i for i, a in enumerate(self.args) if a.is_pointer]
        self._scalar_idx = [i for i, a in enumerate(self.args) if not a.is_pointer]
        self._packer = struct.Struct("<" + "".join(self.args[i].struct_format
                                                   for i in self._scalar_idx))  # fmt: skip
        self._plan = _C.LaunchPlan([self.args[i].index for i in self._ptr_idx],
                                   scalar_slots(self.args))  # fmt: skip
        self._tg = (self.num_warps * 32, 1, 1)
        self._torch: TorchLaunch | str | None = None  # a reason string if unavailable
        self._fallback_logged = False

    def launch(self, grid: tuple[int, int, int], values: Sequence[Any]) -> None:
        """Launches with runtime argument values in signature order."""
        if grid[0] == 0 or grid[1] == 0 or grid[2] == 0:
            return
        if grid[0] > MAX_GRID or grid[1] > MAX_GRID or grid[2] > MAX_GRID:
            raise ValueError(f"grid {grid} exceeds the device limit of {MAX_GRID} per dimension")
        tl = self._torch
        if type(tl) is TorchLaunch:  # the PyTorch hot path: every array is an MPS tensor
            tensor_type = sys.modules["torch"].Tensor
            for i in self._ptr_idx:
                a = values[i]
                if type(a) is not tensor_type or not a.is_mps:
                    break
            else:
                tl.launch(values, (grid[0] * self._tg[0], grid[1], grid[2]), self._tg)
                return
        stream = get_device().stream
        bufs, offsets = [], []
        sync = False
        for i in self._ptr_idx:
            a = values[i]
            if type(a) is Tensor:
                bufs.append(a.buffer)
                offsets.append(a.offset * a.np_dtype.itemsize if a.offset else 0)
                continue
            if interop.framework_of(a) is not None:
                self._launch_foreign(grid, values)
                return
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

    def _launch_foreign(self, grid: tuple[int, int, int], values: Sequence[Any]) -> None:
        """Launches a kernel that has at least one PyTorch or MLX array argument."""
        kinds = {interop.framework_of(values[i]) for i in self._ptr_idx}
        if kinds == {interop.KIND_TORCH}:
            for i in self._ptr_idx:
                interop.torch_np_dtype(values[i])  # refuses CPU and float64 tensors
            tl = self._torch
            if tl is None and (self.language_version or (0, 0)) > TORCH_LANGUAGE_VERSION:
                tl = self._torch = (f"it needs MSL {self.language_version}, and compile_shader "
                                    "compiles with MSL 4.0")  # fmt: skip
            if tl is None:
                types = [None if a.is_pointer else a.dtype for a in self.args]
                tl = make_torch_launch(self.msl, self.name, self.math_mode, types, checked=True)
                if isinstance(tl, TorchLaunch) and tl.max_threads < self._tg[0]:
                    tl = (f"its PyTorch pipeline allows {tl.max_threads} threads per "
                          f"threadgroup, and the kernel needs {self._tg[0]}")  # fmt: skip
                self._torch = tl
            if isinstance(tl, TorchLaunch):
                tl.launch(values, (grid[0] * self._tg[0], grid[1], grid[2]), self._tg)
                return
            reason = f"torch.mps.compile_shader can't run it ({tl})"
        elif interop.KIND_TORCH in kinds:
            reason = "it mixes PyTorch tensors with other array types"
        else:
            reason = None  # MLX launches are synchronous by design
        if reason is not None and not self._fallback_logged:
            self._fallback_logged = True
            log_fallback(self.name, reason)
        for i in self._ptr_idx:
            if self.args[i].written:
                _check_writable(values[i], self.args[i].name)
        scalars = self._packer.pack(*[values[i] for i in self._scalar_idx])
        launch_synced(get_device().stream, self.pipeline, self._plan,
                      [values[i] for i in self._ptr_idx], scalars, grid, self._tg)  # fmt: skip

    def timed_launch(self, grid: tuple[int, int, int], values: Sequence[Any]) -> float:
        """Runs one launch in its own command buffer and returns its GPU time in seconds."""
        interop.torch_synchronize()
        bufs = [as_kernel_arg(values[i]) for i in self._ptr_idx]
        scalars = self._packer.pack(*[values[i] for i in self._scalar_idx])
        t0, t1 = get_device().stream.native.timed_run(
            self.pipeline, self._plan, [b.buffer for b in bufs], [b.byte_offset for b in bufs],
            scalars, grid, self._tg,
        )  # fmt: skip
        return t1 - t0


def _check_writable(a: Any, name: str) -> None:
    """Refuses to write through a broadcast MLX array, whose elements share memory."""
    if interop.framework_of(a) != interop.KIND_MLX:
        return
    ba = as_kernel_arg(a)
    if any(s == 0 and n > 1 for s, n in zip(ba.strides, ba.shape, strict=True)):
        raise ValueError(
            f"argument `{name}` is a broadcast MLX array, and the kernel writes to it; "
            "allocate outputs with, for example, `mx.zeros(shape)` followed by `mx.eval()`"
        )
