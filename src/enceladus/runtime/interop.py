"""Argument adapters: turn array-like kernel arguments into bindable Metal buffers.

M0 handles `enceladus.Tensor` and NumPy arrays. PyTorch MPS tensors and MLX arrays
arrive in M6.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from enceladus import _C
from enceladus.runtime.device import PAGE_SIZE, get_device
from enceladus.runtime.tensor import Tensor, to_np_dtype

KIND_TENSOR = "tensor"
KIND_NUMPY = "numpy"


@dataclass
class BufferArg:
    """A kernel argument resolved to a Metal buffer.

    Attributes:
        buffer: The Metal buffer to bind.
        byte_offset: The offset of the first element within `buffer`.
        dtype: The element type as a NumPy dtype.
        shape: The array shape.
        strides: The array strides, in elements.
        owner: The object whose memory the buffer references.
        kind: Where the memory came from, such as "tensor" or "numpy".
        writeback: If set, copies results back to host memory after the GPU is done.
        needs_sync: Whether a launch using this argument must wait before returning.
    """

    buffer: _C.Buffer
    byte_offset: int
    dtype: np.dtype
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    owner: Any
    kind: str
    writeback: Callable[[], None] | None = None
    needs_sync: bool = False


def is_array_like(obj: Any) -> bool:
    """Returns whether `obj` binds as a pointer argument."""
    return isinstance(obj, (Tensor, np.ndarray))


# ---- NumPy wrapping ----

# id(base array) -> (weakref to base, {(page address, length): Buffer}). The buffers
# don't own the array; the entry disappears when the array is freed, so a later
# allocation at the same address never reuses a stale wrapper.
_np_cache: dict[int, tuple[weakref.ref, dict[tuple[int, int], _C.Buffer]]] = {}
_async_numpy = False


def async_numpy(enabled: bool = True) -> None:
    """Sets whether launches that touch NumPy arrays return before the GPU finishes.

    By default, such launches wait, because NumPy has no notion of a stream. With
    `async_numpy(True)`, you must call `enceladus.synchronize()` before you read the
    arrays on the host.
    """
    global _async_numpy
    _async_numpy = bool(enabled)


def _root(a: np.ndarray) -> Any:
    base = a
    while isinstance(base, np.ndarray) and base.base is not None:
        base = base.base
    return base


def _wrap_host_range(owner: Any, start: int, nbytes: int) -> tuple[_C.Buffer, int] | None:
    """Wraps the pages covering [start, start + nbytes) without copying."""
    page = start - start % PAGE_SIZE
    length = -(-(start - page + nbytes) // PAGE_SIZE) * PAGE_SIZE
    key = (page, length)
    try:
        ref = weakref.ref(owner)
    except TypeError:
        ref = None
    entry = _np_cache.get(id(owner)) if ref is not None else None
    if entry is not None and entry[0]() is owner and key in entry[1]:
        return entry[1][key], start - page
    buf = _C.buffer_nocopy(get_device().native, page, length, None)
    if buf is None:
        return None
    if ref is not None:
        if entry is None or entry[0]() is not owner:
            oid = id(owner)

            def _drop(_ref, oid=oid):
                _np_cache.pop(oid, None)

            entry = (weakref.ref(owner, _drop), {})
            _np_cache[oid] = entry
        if len(entry[1]) < 8:
            entry[1][key] = buf
    return buf, start - page


def _from_numpy(a: np.ndarray) -> BufferArg:
    d = to_np_dtype(a.dtype)
    itemsize = d.itemsize
    if any(s < 0 for s in a.strides):
        raise ValueError(
            "NumPy arguments with negative strides aren't supported; "
            "pass np.ascontiguousarray(x) instead"
        )
    if any(s % itemsize for s in a.strides):
        raise ValueError("NumPy argument strides must be multiples of the element size")
    strides = tuple(s // itemsize for s in a.strides)
    start = a.__array_interface__["data"][0]
    pairs = zip(a.shape, strides, strict=True)
    extent = 0 if a.size == 0 else 1 + sum((n - 1) * s for n, s in pairs)
    wrapped = _wrap_host_range(_root(a), start, max(extent * itemsize, 1)) if a.size else None
    if wrapped is not None:
        buf, off = wrapped
        return BufferArg(buf, off, d, a.shape, strides, a, KIND_NUMPY, None, not _async_numpy)
    # Fallback: copy into a device buffer, and copy back after the launch.
    t = Tensor(_C.new_buffer(get_device().native, max(extent, 1) * itemsize), (extent,), d)
    flat = np.lib.stride_tricks.as_strided(a, shape=(extent,), strides=(itemsize,))
    t._view()[...] = flat

    def writeback(a=a, t=t, extent=extent) -> None:
        if a.flags.writeable:
            np.lib.stride_tricks.as_strided(a, shape=(extent,), strides=(itemsize,))[...] = (
                t._view()
            )

    return BufferArg(t.buffer, 0, d, a.shape, strides, a, KIND_NUMPY, writeback, True)


def as_kernel_arg(obj: Any) -> BufferArg:
    """Resolves an array-like kernel argument to a `BufferArg`.

    Raises:
        TypeError: `obj` isn't a supported array type.
    """
    if isinstance(obj, Tensor):
        return BufferArg(
            obj.buffer, obj.byte_offset, obj.np_dtype, obj.shape, obj.strides, obj, KIND_TENSOR
        )
    if isinstance(obj, np.ndarray):
        return _from_numpy(obj)
    raise TypeError(
        f"unsupported kernel argument of type {type(obj).__name__}; "
        "pass a enceladus.Tensor or a NumPy array"
    )
