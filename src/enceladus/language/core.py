"""Core language objects: dtypes, `constexpr`, and the builtin registry.

This module has no dependencies on the rest of Enceladus, so the runtime, the compiler, and
the interpreter can all import it.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "dtype",
    "constexpr",
    "Builtin",
    "builtin",
    "dtype_from_numpy",
    "dtype_from_name",
    "int1",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "bfloat16",
    "float32",
]


# ---------------------------------------------------------------------------
# dtypes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class dtype:  # noqa: N801 - matches `triton.language.dtype`
    """A Enceladus element type, such as `tl.float32` or `tl.int32`.

    Attributes:
        name: The Triton-style short name, for example `"fp32"`, `"bf16"`, or `"i32"`.
        kind: One of `"bool"`, `"int"`, `"uint"`, or `"float"`.
        primitive_bitwidth: The width in bits. `int1` reports 1.
        ir_name: The name of the matching IR scalar type, for example `"f32"`.
    """

    name: str
    kind: str
    primitive_bitwidth: int
    ir_name: str
    long_name: str

    @property
    def itemsize(self) -> int:
        """The storage size in bytes. `int1` is stored as one byte."""
        return max(1, self.primitive_bitwidth // 8)

    def is_floating(self) -> bool:
        return self.kind == "float"

    def is_int(self) -> bool:
        """Returns whether this is an integer type, including `int1`."""
        return self.kind in ("int", "uint", "bool")

    def is_bool(self) -> bool:
        return self.kind == "bool"

    def is_signed(self) -> bool:
        """Returns whether the type is signed. Floats are signed, `int1` isn't."""
        return self.kind in ("int", "float")

    def is_unsigned(self) -> bool:
        return self.kind in ("uint", "bool")

    @property
    def element_ty(self) -> dtype:
        """Returns this dtype.

        Inside a kernel, `x_ptr.dtype` is the pointee dtype, so both `x_ptr.dtype` and the
        Triton spelling `x_ptr.dtype.element_ty` give the element dtype.
        """
        return self

    def to_numpy(self) -> np.dtype:
        """Returns the NumPy storage dtype. `bfloat16` uses `ml_dtypes.bfloat16`."""
        return _NUMPY[self.name]

    def __repr__(self) -> str:
        return f"tl.{self.long_name}"

    __str__ = __repr__

    def __reduce__(self):
        return (dtype_from_name, (self.name,))


int1 = dtype("i1", "bool", 1, "i1", "int1")
int8 = dtype("i8", "int", 8, "i8", "int8")
int16 = dtype("i16", "int", 16, "i16", "int16")
int32 = dtype("i32", "int", 32, "i32", "int32")
int64 = dtype("i64", "int", 64, "i64", "int64")
uint8 = dtype("u8", "uint", 8, "u8", "uint8")
uint16 = dtype("u16", "uint", 16, "u16", "uint16")
uint32 = dtype("u32", "uint", 32, "u32", "uint32")
uint64 = dtype("u64", "uint", 64, "u64", "uint64")
float16 = dtype("fp16", "float", 16, "f16", "float16")
bfloat16 = dtype("bf16", "float", 16, "bf16", "bfloat16")
float32 = dtype("fp32", "float", 32, "f32", "float32")

ALL_DTYPES: tuple[dtype, ...] = (
    int1, int8, int16, int32, int64, uint8, uint16, uint32, uint64,
    float16, bfloat16, float32,
)  # fmt: skip


def _numpy_table() -> dict[str, np.dtype]:
    import ml_dtypes

    return {
        "i1": np.dtype(np.bool_),
        "i8": np.dtype(np.int8),
        "i16": np.dtype(np.int16),
        "i32": np.dtype(np.int32),
        "i64": np.dtype(np.int64),
        "u8": np.dtype(np.uint8),
        "u16": np.dtype(np.uint16),
        "u32": np.dtype(np.uint32),
        "u64": np.dtype(np.uint64),
        "fp16": np.dtype(np.float16),
        "bf16": np.dtype(ml_dtypes.bfloat16),
        "fp32": np.dtype(np.float32),
    }


_NUMPY = _numpy_table()
_FROM_NUMPY = {v: k for k, v in _NUMPY.items()}
_BY_NAME: dict[str, dtype] = {}
for _d in ALL_DTYPES:
    _BY_NAME[_d.name] = _d
    _BY_NAME[_d.ir_name] = _d
    _BY_NAME[_d.long_name] = _d
_BY_NAME["bool"] = int1


def dtype_from_numpy(np_dtype: Any) -> dtype:
    """Returns the Enceladus dtype for a NumPy dtype.

    Raises:
        TypeError: The dtype has no Enceladus equivalent, for example `float64`.
    """
    d = np.dtype(np_dtype)
    name = _FROM_NUMPY.get(d)
    if name is not None:
        return _BY_NAME[name]
    if d == np.float64:
        raise TypeError(
            "Enceladus has no float64 type. Cast the array to float32 first, for example "
            "with `x.astype(np.float32)`."
        )
    raise TypeError(
        f"NumPy dtype {d} has no Enceladus equivalent. Convert the array to a supported dtype, "
        "such as float32 or int32."
    )


def dtype_from_name(name: str) -> dtype:
    """Returns the dtype for a name such as `"fp32"`, `"f32"`, or `"float32"`.

    Raises:
        KeyError: The name isn't a Enceladus dtype.
    """
    try:
        return _BY_NAME[name]
    except KeyError:
        if name in ("fp64", "f64", "float64", "double"):
            raise KeyError("Enceladus has no float64 type. Use float32 instead.") from None
        raise KeyError(f"unknown dtype name {name!r}") from None


# ---------------------------------------------------------------------------
# constexpr
# ---------------------------------------------------------------------------


class constexpr:  # noqa: N801 - matches `triton.language.constexpr`
    """Marks a kernel parameter as a compile-time constant, or wraps a constant value.

    Annotate a parameter as `BLOCK: tl.constexpr` to make its value part of the kernel's
    specialization. The frontend evaluates any expression over constexprs and Python
    literals in Python.
    """

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value.value if isinstance(value, constexpr) else value

    def __repr__(self) -> str:
        return f"constexpr({self.value!r})"

    def __eq__(self, other: object) -> bool:
        return unwrap(self) == unwrap(other)

    def __hash__(self) -> int:
        return hash(self.value)

    def __bool__(self) -> bool:
        return bool(self.value)

    def __index__(self) -> int:
        return self.value.__index__()

    def __int__(self) -> int:
        return int(self.value)

    def __float__(self) -> float:
        return float(self.value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.value, name)


def unwrap(value: Any) -> Any:
    """Returns the Python value inside a `constexpr`, or `value` itself."""
    return value.value if isinstance(value, constexpr) else value


def is_constexpr_annotation(annotation: Any) -> bool:
    """Returns whether a parameter annotation means `tl.constexpr`."""
    if annotation is constexpr:
        return True
    if isinstance(annotation, str):
        return annotation.split(".")[-1] == "constexpr"
    return False


# ---------------------------------------------------------------------------
# builtin registry
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ENCELADUS_DEBUG
# ---------------------------------------------------------------------------

_OFF = (b"", b"0", b"false", b"False")
_ENV_DATA = getattr(os.environ, "_data", None)
if type(_ENV_DATA) is dict:
    # `os.environ` keeps its contents in `_data`, a dict of bytes that assignments and
    # deletions through `os.environ` update. Its bound `get` costs one C call, several
    # times less than `os.environ.get`, which matters because every launch reads it.
    env_lookup = _ENV_DATA.get
else:  # pragma: no cover - CPython always has `_data`

    def env_lookup(key: bytes, default: bytes | None = None) -> bytes | None:
        """Returns the raw value of environment variable `key`, or `default`."""
        v = os.environ.get(key.decode())
        return default if v is None else v.encode()


def debug_enabled() -> bool:
    """Returns whether `ENCELADUS_DEBUG` is set, which turns on `tl.device_assert`.

    The flag is part of every kernel's specialization, so changing it recompiles.
    """
    return env_lookup(b"ENCELADUS_DEBUG", b"") not in _OFF


class _InterpState:
    """Tracks whether the interpreter is executing a kernel, and the current program."""

    def __init__(self) -> None:
        self.depth = 0
        self.program_id: tuple[int, int, int] = (0, 0, 0)
        self.grid: tuple[int, int, int] = (1, 1, 1)


INTERP = _InterpState()


class Builtin:
    """A `tl` function with a frontend handler and an interpreter handler.

    The frontend handler takes the code generator context first, followed by the user's
    arguments as IR values or compile-time Python values, and emits IR. The interpreter
    handler takes the user's arguments as interpreter tiles, pointers, or Python values
    and computes on NumPy.
    """

    def __init__(self, name: str, frontend: Callable[..., Any], interp: Callable[..., Any]):
        self.name = name
        self.frontend = frontend
        self.interp = interp
        functools.update_wrapper(self, frontend)
        self.__name__ = name
        self.__qualname__ = name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if INTERP.depth > 0:
            return self.interp(*args, **kwargs)
        raise RuntimeError(
            f"tl.{self.name} can only be called inside a @enceladus.jit function. To run a "
            "kernel on the CPU, set ENCELADUS_INTERPRET=1."
        )

    def __repr__(self) -> str:
        return f"<tl.{self.name}>"


def builtin(interp: Callable[..., Any], name: str | None = None):
    """Registers a `tl` builtin from its frontend handler and interpreter handler.

    Example:

        @builtin(interp=_interp_load)
        def load(ctx, pointer, mask=None, other=None): ...
    """

    def deco(frontend: Callable[..., Any]) -> Builtin:
        n = name or frontend.__name__.rstrip("_")
        b = Builtin(n, frontend, interp)
        BUILTINS[n] = b
        return b

    return deco


BUILTINS: dict[str, Builtin] = {}
"""Every registered builtin, by `tl` name."""

TILE_METHODS: dict[str, Builtin] = {}
"""Tile methods such as `x.to(...)`; each receives the tile as its first argument."""

DESC_METHODS: dict[str, Builtin] = {}
"""Tensor descriptor methods, `desc.load` and `desc.store`."""
