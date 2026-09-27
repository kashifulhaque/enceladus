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

import ctypes
import hashlib
import logging
import math
import struct
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from enceladus import _C
from enceladus.compiler.codegen.msl import KernelArg, scalar_slots
from enceladus.runtime import interop
from enceladus.runtime.device import get_device
from enceladus.runtime.interop import as_kernel_arg
from enceladus.runtime.raw import (
    MAX_THREADS_PER_DIM,
    after_dispatch,
    check_grid,
    check_writable_numpy,
)
from enceladus.runtime.tensor import Tensor

MAX_ELEMENT_INDEX = (1 << 31) - 1
"""The largest element offset from an array argument's first element that a kernel
compiled with 32-bit offsets can address. A larger offset would wrap around, so
`@enceladus.jit` compiles an `idx64` variant, with 64-bit index math, for any launch with
an array argument that spans more elements."""
# A buffer of at most this many bytes can't hold an element past MAX_ELEMENT_INDEX, so
# launches check the exact span only of arguments in larger buffers.
_INDEX_CHECK_BYTES = 1 << 31
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
                  scalars: bytes, grid: tuple, tg: tuple,
                  extra_bufs: Sequence[Any] = ()) -> None:  # fmt: skip
    """Dispatches on Enceladus's stream between syncs with PyTorch's stream.

    Waits for PyTorch's pending work, binds every array argument (evaluating MLX arrays),
    dispatches, and waits until the GPU finishes, so PyTorch and MLX read the results.
    `extra_bufs` are native buffers bound after the arguments, such as an error buffer.
    """
    interop.torch_synchronize()
    bufs, offsets, args = [], [], []
    for a in bufs_values:
        ba = as_kernel_arg(a)
        bufs.append(ba.buffer)
        offsets.append(ba.byte_offset)
        args.append(ba)
    for b in extra_bufs:
        bufs.append(b)
        offsets.append(0)
    stream.native.dispatch(pipeline, plan, bufs, offsets, scalars, grid, tg)
    # Register keep-alives and copy-backs after the dispatch, so that a sync on another
    # thread can't release the memory or copy back before the dispatch runs.
    for ba in args:
        stream.keep_alive(ba.owner)
        if ba.writeback is not None:
            stream.after_sync(ba.writeback)
    stream.synchronize()


def _last_index(value: Any) -> int:
    """Returns how many elements the last element of array `value` lies past its first."""
    shape = tuple(value.shape)
    if 0 in shape:
        return -1
    strides = interop.element_strides(value)
    return sum((n - 1) * abs(s) for n, s in zip(shape, strides, strict=True))


def needs_idx64(value: Any) -> bool:
    """Returns whether array `value` needs 64-bit offsets: some element lies more than
    `MAX_ELEMENT_INDEX` elements past its first.

    It checks the exact span only of arrays in buffers larger than 2 GB, and of
    non-contiguous NumPy arrays, so the common case costs one attribute read.
    """
    t = type(value)
    if t is Tensor:
        if value.buffer.nbytes <= _INDEX_CHECK_BYTES:
            return False
    elif t is np.ndarray:
        if value.nbytes <= _INDEX_CHECK_BYTES and value.flags.c_contiguous:
            return False
    elif interop.framework_of(value) == interop.KIND_TORCH:
        if value.untyped_storage().nbytes() <= _INDEX_CHECK_BYTES:
            return False
    return _last_index(value) > MAX_ELEMENT_INDEX


def check_index_range(value: Any, name: str) -> None:
    """Refuses an array argument whose elements lie too far apart for 32-bit offsets.

    Raises:
        ValueError: Some element of `value` is more than `MAX_ELEMENT_INDEX` elements
            past its first element.
    """
    last = _last_index(value)
    if last > MAX_ELEMENT_INDEX:
        raise ValueError(
            f"argument `{name}` spans {last + 1} elements, but this kernel was compiled with "
            "32-bit offsets, which address at most 2^31 elements of each array argument. "
            "Launch it with `kernel[grid](...)`, which compiles a variant with 64-bit "
            "offsets for such arrays, or compile one with `kernel.warmup(...)` on the large "
            "array."
        )


def _no_check(value: Any, name: str) -> None:
    """Accepts any span: a kernel with 64-bit offsets (`idx64`) addresses every element."""


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
        language_version: The MSL version the kernel compiles with, as (major, minor).
        enable_logging: Whether the kernel prints with `tl.device_print`. Such a kernel
            runs on the stream's logging queue and never on the PyTorch path.
        asserts: The kernel's `tl.device_assert` calls, as dicts with `message`, `file`,
            `line`, and `col`, indexed as in the error buffer.
        assert_buffer_index: The buffer index of the error buffer, or None.
        dot_backend: The `tl.dot` backend the kernel uses ("mpp" or "simdgroup"), or None
            if it has no `tl.dot`.
        dot_fallbacks: Why each `tl.dot` that `dot_backend="mpp"` asked for uses
            `simdgroup` instead.
        idx64: Whether the kernel computes index math and offsets in 64 bits, so it can
            address array arguments that span more than 2^31 elements.
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
    language_version: tuple[int, int] = (3, 2)
    enable_logging: bool = False
    asserts: list[dict[str, Any]] = field(default_factory=list)
    assert_buffer_index: int | None = None
    dot_backend: str | None = None
    dot_fallbacks: list[str] = field(default_factory=list)
    idx64: bool = False

    def __post_init__(self) -> None:
        # A kernel with 32-bit offsets refuses arrays that it can't address.
        self._span_check = _no_check if self.idx64 else check_index_range
        self._ptr_idx = [i for i, a in enumerate(self.args) if a.is_pointer]
        self._scalar_idx = [i for i, a in enumerate(self.args) if not a.is_pointer]
        self._packer = struct.Struct("<" + "".join(self.args[i].struct_format
                                                   for i in self._scalar_idx))  # fmt: skip
        buf_index = [self.args[i].index for i in self._ptr_idx]
        # Buffers bound after the arguments: the device-assert error buffer, if any.
        self._extra_bufs: list[Any] = []
        self.assert_buffer = None
        if self.assert_buffer_index is not None:
            self.assert_buffer = _C.new_buffer(get_device().native, 32)
            ctypes.memset(self.assert_buffer.ptr, 0, 32)
            self._extra_bufs.append(self.assert_buffer)
            buf_index.append(self.assert_buffer_index)
        self._plan = _C.LaunchPlan(buf_index, scalar_slots(self.args))
        self._tg = (self.num_warps * 32, 1, 1)
        self._max_grid0 = MAX_THREADS_PER_DIM // self._tg[0]
        self._checked_pipeline: Any = None  # the pipeline that `_check_pipeline` accepted
        self._torch: TorchLaunch | str | None = None  # a reason string if unavailable
        self._fallback_logged = False
        self._debug = self.enable_logging or self.assert_buffer is not None
        if self.enable_logging:
            self._torch = ("it calls tl.device_print, which needs Enceladus's logging queue; "
                           "torch.mps.compile_shader can't attach one")  # fmt: skip
        elif self.assert_buffer is not None:
            self._torch = "it has device asserts (ENCELADUS_DEBUG=1), which bind an error buffer"

    def _prepare_debug(self, stream: Any) -> None:
        """Readies the stream for a launch that prints or asserts."""
        if self.enable_logging:
            stream.enable_logging()
            stream.native.mark_logging()
        if self.assert_buffer is not None:
            stream.watch_asserts(self)

    def assert_error(self, index: int, program_id: tuple[int, int, int]) -> Exception:
        """Returns the `DeviceAssertionError` for assert `index` failing in `program_id`."""
        from enceladus.compiler.errors import DeviceAssertionError, Loc

        a = self.asserts[index] if index < len(self.asserts) else {"message": "?", "file": ""}
        loc = Loc(a["file"], a["line"], a.get("col", 1)) if a["file"] else None
        return DeviceAssertionError(a["message"], loc, program_id)

    def _check_pipeline(self) -> None:
        """Checks that the pipeline can run a threadgroup of `num_warps * 32` threads.

        Raises:
            ValueError: The pipeline allows fewer threads per threadgroup, for example
                because the kernel uses too many registers.
        """
        limit = self.pipeline.max_total_threads_per_threadgroup
        if limit < self._tg[0]:
            raise ValueError(
                f"kernel `{self.name}` launches {self._tg[0]} threads per threadgroup "
                f"(num_warps={self.num_warps}), but its Metal pipeline allows at most {limit}. "
                "Use a smaller num_warps."
            )
        self._checked_pipeline = self.pipeline

    def _pack_scalars(self, values: Sequence[Any]) -> bytes:
        """Packs the scalar arguments, converting them as C converts to the parameter type.

        A float outside the range of `float32` (or `float16`) packs as an infinity of the
        same sign, which is also what the interpreter computes.

        Raises:
            OverflowError: An integer is out of range for its parameter type.
        """
        vals = [values[i] for i in self._scalar_idx]
        try:
            return self._packer.pack(*vals)
        except (OverflowError, struct.error):
            pass
        for k, i in enumerate(self._scalar_idx):
            a = self.args[i]
            try:
                struct.pack("<" + a.struct_format, vals[k])
            except OverflowError:
                if a.struct_format not in ("f", "e"):
                    raise OverflowError(
                        f"argument `{a.name}` = {vals[k]} is out of range for {a.dtype}"
                    ) from None
                vals[k] = math.copysign(math.inf, float(vals[k]))
            except struct.error as e:
                raise OverflowError(
                    f"argument `{a.name}` = {vals[k]} is out of range for {a.dtype}: {e}"
                ) from None
        return self._packer.pack(*vals)

    def launch(self, grid: tuple[int, int, int], values: Sequence[Any]) -> None:
        """Launches with runtime argument values in signature order."""
        if grid[0] == 0 or grid[1] == 0 or grid[2] == 0:
            return
        if (grid[0] > self._max_grid0 or grid[1] > MAX_THREADS_PER_DIM
                or grid[2] > MAX_THREADS_PER_DIM):  # fmt: skip
            check_grid(grid, self._tg)  # raises
        if self.pipeline is not self._checked_pipeline:
            self._check_pipeline()
        debug = self._debug
        if debug:
            self._prepare_debug(get_device().stream)
        tl = self._torch
        if type(tl) is TorchLaunch:  # the PyTorch hot path: every array is an MPS tensor
            tensor_type = sys.modules["torch"].Tensor
            for i in self._ptr_idx:
                a = values[i]
                if type(a) is not tensor_type or not a.is_mps:
                    break
                if a.untyped_storage().nbytes() > _INDEX_CHECK_BYTES:
                    self._span_check(a, self.args[i].name)
            else:
                tl.launch(values, (grid[0] * self._tg[0], grid[1], grid[2]), self._tg)
                return
        stream = get_device().stream
        bufs, offsets = [], []
        host: list[Any] = []  # BufferArgs over host memory
        for i in self._ptr_idx:
            a = values[i]
            if type(a) is Tensor:
                buf = a.buffer
                if buf.nbytes > _INDEX_CHECK_BYTES:
                    self._span_check(a, self.args[i].name)
                bufs.append(buf)
                offsets.append(a.offset * a.np_dtype.itemsize if a.offset else 0)
                continue
            if interop.framework_of(a) is not None:
                self._launch_foreign(grid, values)
                return
            if isinstance(a, np.ndarray):
                # Check before wrapping, so that a huge strided view never reaches the
                # copy fallback.
                self._span_check(a, self.args[i].name)
                if self.args[i].written:
                    check_writable_numpy(a, f"argument `{self.args[i].name}`")
            ba = as_kernel_arg(a)
            bufs.append(ba.buffer)
            offsets.append(ba.byte_offset)
            host.append(ba)
        if debug and self._extra_bufs:
            bufs += self._extra_bufs
            offsets += [0] * len(self._extra_bufs)
        try:
            scalars = self._packer.pack(*[values[i] for i in self._scalar_idx])
        except (OverflowError, struct.error):
            scalars = self._pack_scalars(values)
        stream.native.dispatch(self.pipeline, self._plan, bufs, offsets, scalars, grid, self._tg)
        if debug and self.assert_buffer is not None:
            # Watch again after the dispatch, in case a sync on another thread took the
            # watch that `_prepare_debug` registered before the dispatch.
            stream.watch_asserts(self)
        if host:
            after_dispatch(stream, host)

    def _launch_foreign(self, grid: tuple[int, int, int], values: Sequence[Any]) -> None:
        """Launches a kernel that has at least one PyTorch or MLX array argument."""
        kinds = {interop.framework_of(values[i]) for i in self._ptr_idx}
        if kinds == {interop.KIND_TORCH}:
            for i in self._ptr_idx:
                interop.torch_np_dtype(values[i])  # refuses CPU and float64 tensors
                self._span_check(values[i], self.args[i].name)
            tl = self._torch
            if tl is None and self.language_version > TORCH_LANGUAGE_VERSION:
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
            self._span_check(values[i], self.args[i].name)
            if self.args[i].written:
                _check_writable(values[i], self.args[i].name)
        scalars = self._pack_scalars(values)
        launch_synced(get_device().stream, self.pipeline, self._plan,
                      [values[i] for i in self._ptr_idx], scalars, grid, self._tg,
                      self._extra_bufs)  # fmt: skip

    def timed_launch(self, grid: tuple[int, int, int], values: Sequence[Any]) -> float:
        """Runs one launch in its own command buffer and returns its GPU time in seconds."""
        check_grid(grid, self._tg)
        if self.pipeline is not self._checked_pipeline:
            self._check_pipeline()
        for i in self._ptr_idx:
            self._span_check(values[i], self.args[i].name)
        interop.torch_synchronize()
        stream = get_device().stream
        if self._debug:
            self._prepare_debug(stream)
        bufs = [as_kernel_arg(values[i]) for i in self._ptr_idx]
        scalars = self._pack_scalars(values)
        t0, t1 = stream.native.timed_run(
            self.pipeline, self._plan, [b.buffer for b in bufs] + self._extra_bufs,
            [b.byte_offset for b in bufs] + [0] * len(self._extra_bufs), scalars, grid, self._tg,
        )  # fmt: skip
        return t1 - t0


def _check_writable(a: Any, name: str) -> None:
    """Refuses to write through a read-only NumPy array or a broadcast MLX array."""
    check_writable_numpy(a, f"argument `{name}`")
    if interop.framework_of(a) != interop.KIND_MLX:
        return
    ba = as_kernel_arg(a)
    if any(s == 0 and n > 1 for s, n in zip(ba.strides, ba.shape, strict=True)):
        raise ValueError(
            f"argument `{name}` is a broadcast MLX array, and the kernel writes to it; "
            "allocate outputs with, for example, `mx.zeros(shape)` followed by `mx.eval()`"
        )
