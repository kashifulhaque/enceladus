"""Argument adapters: turn array-like kernel arguments into bindable Metal buffers.

The adapters accept `enceladus.Tensor` objects, NumPy arrays, PyTorch tensors on the
`mps` device, and MLX arrays. Enceladus detects PyTorch and MLX objects by their type's
module, so importing Enceladus never imports either framework.

- A PyTorch MPS tensor binds its storage's `id<MTLBuffer>`
  (`t.untyped_storage().data_ptr()`) at the byte offset
  `t.storage_offset() * t.element_size()`.
- An MLX array is evaluated with `mx.eval()`, then binds the `id<MTLBuffer>` and byte
  offset from its `kDLMetal` DLPack capsule.
"""

from __future__ import annotations

import sys
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
KIND_TORCH = "torch"
KIND_MLX = "mlx"

K_DL_METAL = 8


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
    return isinstance(obj, (Tensor, np.ndarray)) or framework_of(obj) is not None


def framework_of(obj: Any) -> str | None:
    """Returns "torch" for a PyTorch tensor, "mlx" for an MLX array, and `None` otherwise.

    The check reads `type(obj).__module__` first, so it never imports a framework.
    """
    mod = type(obj).__module__
    if mod.startswith("torch"):
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(obj, torch.Tensor):
            return KIND_TORCH
    elif mod == "mlx.core" and type(obj).__name__ == "array":
        return KIND_MLX
    return None


# ---- PyTorch ----

_torch_dtypes: dict[Any, np.dtype] = {}


def _torch_dtype_table() -> dict[Any, np.dtype]:
    if not _torch_dtypes:
        import ml_dtypes
        import torch

        _torch_dtypes.update({
            torch.float32: np.dtype(np.float32), torch.float16: np.dtype(np.float16),
            torch.bfloat16: np.dtype(ml_dtypes.bfloat16), torch.bool: np.dtype(np.bool_),
            torch.int8: np.dtype(np.int8), torch.int16: np.dtype(np.int16),
            torch.int32: np.dtype(np.int32), torch.int64: np.dtype(np.int64),
            torch.uint8: np.dtype(np.uint8), torch.uint16: np.dtype(np.uint16),
            torch.uint32: np.dtype(np.uint32), torch.uint64: np.dtype(np.uint64),
        })  # fmt: skip
    return _torch_dtypes


def torch_np_dtype(t: Any) -> np.dtype:
    """Checks that `t` can be a kernel argument and returns its element type.

    Raises:
        TypeError: `t` isn't on the `mps` device, or its dtype is `float64` or another
            type that Enceladus doesn't support.
    """
    if not t.is_mps:
        raise TypeError(
            f"a torch tensor on the {t.device.type} device can't be a kernel argument. "
            'Move it to the GPU with `.to("mps")`.'
        )
    d = _torch_dtype_table().get(t.dtype)
    if d is None:
        if str(t.dtype) == "torch.float64":
            raise TypeError(
                "Enceladus has no float64 type. Convert the tensor to float32 first, for "
                "example with `x.float()`."
            )
        raise TypeError(
            f"torch dtype {t.dtype} has no Enceladus equivalent. Convert the tensor to a "
            "supported dtype, such as float32 or int32."
        )
    return d


def torch_spec_key(t: Any, no_facts: bool) -> tuple[np.dtype, bool]:
    """Returns the (dtype, 16-byte aligned) specialization key of MPS tensor `t`.

    Raises:
        TypeError: `t` can't be a kernel argument; see `torch_np_dtype`.
    """
    d = _torch_dtypes.get(t.dtype) if t.is_mps else None
    if d is None:
        d = torch_np_dtype(t)  # raises with the reason
    return (d, True if no_facts else t.storage_offset() * t.element_size() % 16 == 0)


def torch_aligned16(t: Any) -> bool:
    """Returns whether the first element of MPS tensor `t` is 16-byte aligned.

    Storage buffers start on a page, so the byte offset decides the alignment.
    """
    return t.storage_offset() * t.element_size() % 16 == 0


def _torch_buffer(t: Any) -> _C.Buffer:
    handle = t.untyped_storage().data_ptr()
    if not handle:  # an empty tensor has no storage buffer
        return _C.new_buffer(get_device().native, 16)
    return _C.buffer_from_mtl(handle, t)


def _from_torch(t: Any) -> BufferArg:
    d = torch_np_dtype(t)
    return BufferArg(
        _torch_buffer(t), t.storage_offset() * t.element_size(), d, tuple(t.shape),
        tuple(t.stride()), t, KIND_TORCH, None, True,
    )  # fmt: skip


def torch_synchronize() -> None:
    """Waits for PyTorch's MPS stream if PyTorch is imported."""
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.mps.synchronize()


# ---- MLX ----

_MLX_DTYPES = {
    "float32": np.dtype(np.float32), "float16": np.dtype(np.float16),
    "bool": np.dtype(np.bool_), "int8": np.dtype(np.int8), "int16": np.dtype(np.int16),
    "int32": np.dtype(np.int32), "int64": np.dtype(np.int64), "uint8": np.dtype(np.uint8),
    "uint16": np.dtype(np.uint16), "uint32": np.dtype(np.uint32),
    "uint64": np.dtype(np.uint64),
}  # fmt: skip


def mlx_np_dtype(a: Any) -> np.dtype:
    """Returns the element type of MLX array `a` as a NumPy dtype.

    Raises:
        TypeError: The dtype is `float64` or another type that Enceladus doesn't support.
    """
    name = str(a.dtype).rsplit(".", 1)[-1]
    d = _MLX_DTYPES.get(name)
    if d is not None:
        return d
    if name == "bfloat16":
        import ml_dtypes

        return np.dtype(ml_dtypes.bfloat16)
    if name == "float64":
        raise TypeError(
            "Enceladus has no float64 type. Convert the array to float32 first, for example "
            "with `x.astype(mx.float32)`."
        )
    raise TypeError(
        f"MLX dtype {a.dtype} has no Enceladus equivalent. Convert the array to a supported "
        "dtype, such as float32 or int32."
    )


@dataclass
class _MlxView:
    handle: int
    byte_offset: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    capsule: Any


def _mlx_view(a: Any) -> _MlxView:
    """Evaluates `a` and reads its buffer, offset, shape, and strides from DLPack."""
    import mlx.core as mx

    mx.eval(a)
    cap = a.__dlpack__()
    handle, dev_type, _, off, shape, strides, _ = _C.dlpack_inspect(cap)
    if dev_type != K_DL_METAL:
        raise TypeError(
            f"MLX array exports DLPack device type {dev_type}, not kDLMetal (8). Create the "
            "array on the GPU device."
        )
    if strides is None:
        strides = _contiguous(shape)
    return _MlxView(handle, off, tuple(shape), tuple(strides), cap)


def mlx_aligned16(a: Any) -> bool:
    """Returns whether the first element of MLX array `a` is 16-byte aligned."""
    return _mlx_view(a).byte_offset % 16 == 0


def _from_mlx(a: Any) -> BufferArg:
    d = mlx_np_dtype(a)
    v = _mlx_view(a)
    if v.handle and a.size:
        # The capsule holds MLX's reference to the data until the launch completes.
        buf = _C.buffer_from_mtl(v.handle, (a, v.capsule))
    else:
        buf = _C.new_buffer(get_device().native, 16)
    return BufferArg(buf, v.byte_offset, d, v.shape, v.strides, a, KIND_MLX, None, True)


def _contiguous(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides, acc = [], 1
    for s in reversed(shape):
        strides.append(acc)
        acc *= max(s, 1)
    return tuple(reversed(strides))


# ---- Framework-neutral helpers ----


def element_dtype(obj: Any) -> np.dtype | None:
    """Returns the element type of a PyTorch or MLX array, or `None` for other objects."""
    fw = framework_of(obj)
    if fw == KIND_TORCH:
        return torch_np_dtype(obj)
    if fw == KIND_MLX:
        return mlx_np_dtype(obj)
    return None


def aligned16(obj: Any) -> bool | None:
    """Returns whether a PyTorch or MLX array starts on 16 bytes, or `None` for others."""
    fw = framework_of(obj)
    if fw == KIND_TORCH:
        return torch_aligned16(obj)
    if fw == KIND_MLX:
        return mlx_aligned16(obj)
    return None


def element_strides(obj: Any) -> tuple[int, ...]:
    """Returns the strides of any supported array, in elements.

    NumPy reports strides in bytes, PyTorch through `stride()`, and MLX only through
    DLPack; this function gives them one form for passing to kernels.
    """
    if isinstance(obj, Tensor):
        return obj.strides
    if isinstance(obj, np.ndarray):
        return tuple(s // obj.itemsize for s in obj.strides)
    fw = framework_of(obj)
    if fw == KIND_TORCH:
        return tuple(obj.stride())
    if fw == KIND_MLX:
        return _mlx_view(obj).strides
    raise TypeError(
        f"{type(obj).__name__} isn't a supported array type. Pass an enceladus.Tensor, a "
        "NumPy array, a torch tensor on the mps device, or an MLX array."
    )


def new_empty(like: Any, shape: Any = None, dtype: Any = None) -> Any:
    """Returns an array of the same kind and on the same device as `like`.

    The result is a NumPy array, an `enceladus.Tensor`, a PyTorch tensor, or an MLX
    array, so a host wrapper can allocate its outputs without knowing the framework.
    NumPy, `enceladus.Tensor`, and PyTorch results are uninitialized. MLX results are
    zero-filled and evaluated, because Enceladus writes to MLX arrays in place and
    each output needs a buffer of its own.

    Args:
        like: The array whose kind, device, and (by default) shape and dtype to use.
        shape: The shape of the result. Defaults to `like.shape`.
        dtype: The element type in `like`'s framework. Defaults to `like.dtype`.
    """
    shape = tuple(like.shape) if shape is None else tuple(shape)
    dtype = like.dtype if dtype is None else dtype
    if isinstance(like, np.ndarray):
        return np.empty(shape, dtype)
    if isinstance(like, Tensor):
        from enceladus.runtime.tensor import empty

        return empty(shape, like.np_dtype if dtype is like.dtype else dtype)
    fw = framework_of(like)
    if fw == KIND_TORCH:
        import torch

        return torch.empty(shape, dtype=dtype, device=like.device)
    if fw == KIND_MLX:
        import mlx.core as mx

        out = mx.zeros(shape, dtype=dtype)
        mx.eval(out)
        return out
    raise TypeError(
        f"{type(like).__name__} isn't a supported array type. Pass an enceladus.Tensor, a "
        "NumPy array, a torch tensor on the mps device, or an MLX array."
    )


def as_tensor(obj: Any) -> Tensor:
    """Returns an `enceladus.Tensor` that shares the memory of a PyTorch or MLX array.

    The call waits for the framework's pending work first. Later launches on the
    tensor run on Enceladus's stream, which doesn't order against the framework's own
    work: call `enceladus.synchronize()` before the framework reads the results.

    Raises:
        TypeError: `obj` isn't a PyTorch MPS tensor or an MLX array.
    """
    fw = framework_of(obj)
    if fw == KIND_TORCH:
        torch_synchronize()
        ba = _from_torch(obj)
    elif fw == KIND_MLX:
        ba = _from_mlx(obj)
    else:
        raise TypeError(f"expected a torch MPS tensor or an MLX array, not {type(obj).__name__}")
    if ba.byte_offset % ba.dtype.itemsize:
        raise ValueError("the array's byte offset isn't a multiple of its element size")
    if not ba.buffer.ptr:
        raise TypeError("the array's buffer uses private storage, which the host can't map")
    return Tensor(ba.buffer, ba.shape, ba.dtype, ba.strides, ba.byte_offset // ba.dtype.itemsize)


def host_view(obj: Any) -> Any:
    """Returns a NumPy view of a PyTorch or MLX array's shared memory, or `obj` itself.

    The interpreter runs on these views. The call waits for the framework's pending
    work first; the framework sees the interpreter's writes directly.
    """
    if framework_of(obj) is None:
        return obj
    return as_tensor(obj)._view()


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
        raise ValueError(
            "NumPy argument strides must be multiples of the element size. Pass "
            "np.ascontiguousarray(x) instead."
        )
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
    fw = framework_of(obj)
    if fw == KIND_TORCH:
        return _from_torch(obj)
    if fw == KIND_MLX:
        return _from_mlx(obj)
    raise TypeError(
        f"unsupported kernel argument of type {type(obj).__name__}; pass an enceladus.Tensor, "
        "a NumPy array, a torch tensor on the mps device, or an MLX array"
    )
