"""Enceladus: a Triton-like tile language for Apple GPUs."""

__version__ = "0.1.0.dev0"

# --- runtime (M0) ---
from enceladus._C import MetalError
from enceladus.runtime.autotuner import Config, autotune, heuristics
from enceladus.runtime.device import Capabilities, get_device
from enceladus.runtime.interop import async_numpy, element_strides, new_empty
from enceladus.runtime.raw import metal_kernel
from enceladus.runtime.stream import synchronize
from enceladus.runtime.tensor import (
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
from enceladus import language
from enceladus.compiler.errors import CompilationError
from enceladus.language.core import constexpr
from enceladus.runtime.jit import JITFunction, cdiv, jit, next_power_of_2
