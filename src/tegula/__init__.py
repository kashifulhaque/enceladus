"""Tegula: a Triton-like tile language for Apple GPUs."""

__version__ = "0.1.0.dev0"

# --- runtime (M0) ---
from tegula._C import MetalError
from tegula.runtime.device import Capabilities, get_device
from tegula.runtime.interop import async_numpy
from tegula.runtime.raw import metal_kernel
from tegula.runtime.stream import synchronize
from tegula.runtime.tensor import (
    Tensor,
    arange,
    empty,
    empty_like,
    from_numpy,
    full,
    ones,
    rand,
    randn,
    zeros,
    zeros_like,
)

# --- language and compiler (M1) ---
# isort: split
from tegula import language
from tegula.compiler.errors import CompilationError
from tegula.language.core import constexpr
from tegula.runtime.jit import JITFunction, cdiv, jit, next_power_of_2
