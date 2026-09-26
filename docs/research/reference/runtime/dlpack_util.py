"""Reads the DLTensor inside a DLPack capsule with ctypes (no copy, no consumption)."""

import ctypes
from ctypes import POINTER, Structure, c_char_p, c_int, c_int32, c_int64, c_uint8, c_uint16, c_uint64, c_void_p


class DLDevice(Structure):
    _fields_ = [("device_type", c_int32), ("device_id", c_int32)]


class DLDataType(Structure):
    _fields_ = [("code", c_uint8), ("bits", c_uint8), ("lanes", c_uint16)]


class DLTensor(Structure):
    _fields_ = [
        ("data", c_void_p),
        ("device", DLDevice),
        ("ndim", c_int32),
        ("dtype", DLDataType),
        ("shape", POINTER(c_int64)),
        ("strides", POINTER(c_int64)),
        ("byte_offset", c_uint64),
    ]


class DLManagedTensor(Structure):
    _fields_ = [("dl_tensor", DLTensor), ("manager_ctx", c_void_p), ("deleter", c_void_p)]


# DLPack >= 1.0 "dltensor_versioned": version + manager_ctx + deleter + flags, then DLTensor.
class DLManagedTensorVersioned(Structure):
    _fields_ = [("major", ctypes.c_uint32), ("minor", ctypes.c_uint32), ("manager_ctx", c_void_p),
                ("deleter", c_void_p), ("flags", c_uint64), ("dl_tensor", DLTensor)]


_api = ctypes.pythonapi
_api.PyCapsule_GetPointer.restype = c_void_p
_api.PyCapsule_GetPointer.argtypes = [ctypes.py_object, c_char_p]
_api.PyCapsule_GetName.restype = c_char_p
_api.PyCapsule_GetName.argtypes = [ctypes.py_object]


def inspect_capsule(cap):
    name = _api.PyCapsule_GetName(cap)
    p = _api.PyCapsule_GetPointer(cap, name)
    if name == b"dltensor_versioned":
        t = DLManagedTensorVersioned.from_address(p).dl_tensor
    else:
        t = DLManagedTensor.from_address(p).dl_tensor
    return {
        "capsule_name": name.decode(),
        "data": t.data,
        "device_type": t.device.device_type,
        "ndim": t.ndim,
        "shape": [t.shape[i] for i in range(t.ndim)],
        "strides": [t.strides[i] for i in range(t.ndim)] if t.strides else None,
        "byte_offset": t.byte_offset,
        "dtype": (t.dtype.code, t.dtype.bits, t.dtype.lanes),
    }
