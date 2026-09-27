"""Enceladus: a Triton-like tile language for Apple GPUs."""

from importlib.metadata import PackageNotFoundError, version

try:
    # CI sets the published version in pyproject.toml, so read it from the metadata.
    __version__ = version("enceladus")
except PackageNotFoundError:  # running from a source tree that isn't installed
    __version__ = "0.0.0"

# --- runtime (M0) ---
from enceladus._C import MetalError
from enceladus.runtime.autotuner import Config, autotune, heuristics
from enceladus.runtime.debug import capture
from enceladus.runtime.device import Capabilities, get_device
from enceladus.runtime.interop import (
    async_numpy,
    element_dtype,
    element_strides,
    new_empty,
    new_zeros,
)
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
from enceladus.compiler.errors import CompilationError, DeviceAssertionError
from enceladus.language.core import constexpr
from enceladus.runtime.jit import JITFunction, cdiv, jit, next_power_of_2

_LAZY_SUBMODULES = ("configs", "testing")


def __getattr__(name: str):
    # `enceladus.configs` and `enceladus.testing` import on first use, so `import enceladus`
    # doesn't pay for them.
    if name in _LAZY_SUBMODULES:
        import importlib

        return importlib.import_module(f"enceladus.{name}")
    raise AttributeError(f"module 'enceladus' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY_SUBMODULES])
