"""`tegula.Tensor`: an array in shared GPU memory, and allocation helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from tegula import _C
from tegula.runtime.device import get_device


def to_np_dtype(dtype: Any) -> np.dtype:
    """Converts a tl dtype, NumPy dtype, or dtype name to a NumPy dtype."""
    if hasattr(dtype, "to_numpy"):
        return np.dtype(dtype.to_numpy())
    if isinstance(dtype, str):
        if dtype in ("bfloat16", "bf16"):
            import ml_dtypes

            return np.dtype(ml_dtypes.bfloat16)
        try:
            from tegula.language.core import dtype_from_name

            return np.dtype(dtype_from_name(dtype).to_numpy())
        except (ImportError, KeyError, ValueError, TypeError):
            pass
    d = np.dtype(dtype)
    if d == np.float64:
        raise TypeError("Tegula has no float64 support; use float32 instead.")
    return d


def _tl_dtype(d: np.dtype):
    from tegula.language.core import dtype_from_numpy

    return dtype_from_numpy(d)


def _shape(shape: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(shape, (int, np.integer)):
        return (int(shape),)
    return tuple(int(s) for s in shape)


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides, acc = [], 1
    for s in reversed(shape):
        strides.append(acc)
        acc *= max(s, 1)
    return tuple(reversed(strides))


class Tensor:
    """An n-dimensional array in Metal shared memory.

    Launches that use only `Tensor` arguments are asynchronous. Reading a tensor
    on the host (`numpy()`, `tolist()`, `print()`) waits for pending GPU work.

    Attributes:
        shape: The size of each dimension.
        strides: The step between elements of each dimension, in elements.
        offset: The position of the first element in the buffer, in elements.
    """

    __slots__ = ("buffer", "shape", "strides", "offset", "np_dtype", "__weakref__")

    def __init__(
        self,
        buffer: _C.Buffer,
        shape: tuple[int, ...],
        dtype: Any,
        strides: tuple[int, ...] | None = None,
        offset: int = 0,
    ) -> None:
        self.buffer = buffer
        self.shape = tuple(shape)
        self.np_dtype = to_np_dtype(dtype)
        self.strides = tuple(strides) if strides is not None else _contiguous_strides(self.shape)
        self.offset = offset

    # ---- metadata ----

    @property
    def dtype(self):
        """The element type as a `tegula.language` dtype, for example `tl.float32`."""
        return _tl_dtype(self.np_dtype)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def itemsize(self) -> int:
        return self.np_dtype.itemsize

    @property
    def nbytes(self) -> int:
        return self.numel * self.itemsize

    @property
    def byte_offset(self) -> int:
        return self.offset * self.itemsize

    @property
    def data_ptr(self) -> int:
        """The host address of the first element."""
        return self.buffer.ptr + self.byte_offset

    def stride(self, dim: int) -> int:
        """Returns the stride of dimension `dim`, in elements."""
        return self.strides[dim]

    def size(self, dim: int | None = None):
        return self.shape if dim is None else self.shape[dim]

    def is_contiguous(self) -> bool:
        return self.strides == _contiguous_strides(self.shape)

    def __len__(self) -> int:
        return self.shape[0]

    # ---- host access ----

    def _view(self) -> np.ndarray:
        """Returns a NumPy view of the elements without waiting for the GPU."""
        return _numpy_view(self.buffer, self.np_dtype, self.shape, self.strides, self.offset)

    def numpy(self) -> np.ndarray:
        """Waits for pending GPU work, then returns a zero-copy NumPy view."""
        from tegula.runtime.stream import synchronize

        synchronize()
        return self._view()

    def __array__(self, dtype=None, copy=None):
        a = self.numpy()
        if dtype is not None:
            a = a.astype(dtype, copy=False)
        return a.copy() if copy else a

    def tolist(self) -> list:
        return self.numpy().tolist()

    def __repr__(self) -> str:
        return f"tegula.Tensor({np.array2string(self.numpy())}, dtype={self.dtype})"

    def __getitem__(self, key) -> Tensor:
        """Returns a view that shares this tensor's buffer. Supports basic slicing."""
        v = self._view()[key]
        if not isinstance(v, np.ndarray) or v.flags.owndata:
            raise TypeError(
                "Tensor indexing supports only slices that give a view; "
                "use .numpy() to read single elements"
            )
        if any(s < 0 for s in v.strides):
            raise ValueError("negative strides aren't supported")
        itemsize = self.itemsize
        byte_off = v.__array_interface__["data"][0] - self.buffer.ptr
        return Tensor(
            self.buffer,
            v.shape,
            self.np_dtype,
            tuple(s // itemsize for s in v.strides),
            byte_off // itemsize,
        )

    def copy_(self, src: Any) -> Tensor:
        """Copies `src` (array-like) into this tensor on the host, then returns it."""
        dst = self.numpy()
        dst[...] = np.asarray(src, dtype=self.np_dtype)
        return self


def _numpy_view(
    buffer: _C.Buffer,
    dtype: np.dtype,
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    offset: int,
) -> np.ndarray:
    itemsize = dtype.itemsize
    if math.prod(shape) == 0:
        return np.empty(shape, dtype)
    extent = 1 + sum((d - 1) * s for d, s in zip(shape, strides, strict=True))
    raw = _BufferMemory(buffer).array()
    flat = raw[offset * itemsize : (offset + extent) * itemsize].view(dtype)
    return np.lib.stride_tricks.as_strided(
        flat, shape=shape, strides=tuple(s * itemsize for s in strides), writeable=True
    )


class _BufferMemory:
    """Exposes a buffer's contents to NumPy and keeps the buffer alive."""

    __slots__ = ("buffer", "__array_interface__")

    def __init__(self, buffer: _C.Buffer) -> None:
        self.buffer = buffer
        self.__array_interface__ = {
            "data": (buffer.ptr, False),
            "shape": (buffer.nbytes,),
            "typestr": "|u1",
            "version": 3,
        }

    def array(self) -> np.ndarray:
        return np.asarray(self)


# ---- allocation ----


def empty(shape: int | Sequence[int], dtype: Any = "float32") -> Tensor:
    """Returns an uninitialized tensor."""
    shape = _shape(shape)
    d = to_np_dtype(dtype)
    dev = get_device()
    nbytes = math.prod(shape) * d.itemsize
    if nbytes > dev.caps.max_buffer_length:
        raise MemoryError(
            f"{nbytes} bytes exceeds the device's maximum buffer length "
            f"({dev.caps.max_buffer_length} bytes)"
        )
    return Tensor(_C.new_buffer(dev.native, nbytes), shape, d)


def full(shape: int | Sequence[int], value: Any, dtype: Any = "float32") -> Tensor:
    t = empty(shape, dtype)
    t._view()[...] = value
    return t


def zeros(shape: int | Sequence[int], dtype: Any = "float32") -> Tensor:
    return full(shape, 0, dtype)


def ones(shape: int | Sequence[int], dtype: Any = "float32") -> Tensor:
    return full(shape, 1, dtype)


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def randn(*shape: Any, dtype: Any = "float32", seed: int | None = None) -> Tensor:
    """Returns a tensor of samples from the standard normal distribution."""
    shape = _shape(shape[0] if len(shape) == 1 and not isinstance(shape[0], int) else shape)
    t = empty(shape, dtype)
    t._view()[...] = _rng(seed).standard_normal(shape, dtype=np.float32)
    return t


def rand(*shape: Any, dtype: Any = "float32", seed: int | None = None) -> Tensor:
    """Returns a tensor of samples from the uniform distribution on [0, 1)."""
    shape = _shape(shape[0] if len(shape) == 1 and not isinstance(shape[0], int) else shape)
    t = empty(shape, dtype)
    t._view()[...] = _rng(seed).random(shape, dtype=np.float32)
    return t


def arange(start: int, end: int | None = None, step: int = 1, dtype: Any = "int32") -> Tensor:
    if end is None:
        start, end = 0, start
    values = np.arange(start, end, step)
    t = empty(values.shape, dtype)
    t._view()[...] = values
    return t


def empty_like(t: Any, dtype: Any = None) -> Tensor:
    return empty(tuple(t.shape), dtype if dtype is not None else _dtype_of(t))


def zeros_like(t: Any, dtype: Any = None) -> Tensor:
    return zeros(tuple(t.shape), dtype if dtype is not None else _dtype_of(t))


def _dtype_of(t: Any) -> np.dtype:
    return t.np_dtype if isinstance(t, Tensor) else to_np_dtype(t.dtype)


def from_numpy(a: np.ndarray) -> Tensor:
    """Returns a tensor that shares `a`'s memory when possible.

    The tensor aliases `a` when `a` is C-contiguous and starts on a 16 KB page.
    Otherwise, it holds a copy.
    """
    from tegula.runtime.device import PAGE_SIZE

    d = to_np_dtype(a.dtype)
    dev = get_device()
    ptr = a.__array_interface__["data"][0]
    if a.flags.c_contiguous and a.nbytes > 0 and ptr % PAGE_SIZE == 0:
        length = -(-a.nbytes // PAGE_SIZE) * PAGE_SIZE
        buf = _C.buffer_nocopy(dev.native, ptr, length, a)
        if buf is not None:
            return Tensor(buf, a.shape, d)
    t = empty(a.shape, d)
    t._view()[...] = a
    return t
