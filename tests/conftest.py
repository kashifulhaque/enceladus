"""Shared test helpers: example loading and the `check_kernel` differential test."""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import ml_dtypes
import numpy as np
import pytest

# The interpreter builds and verifies the IR of every kernel it runs when TEGULA_VERIFY is
# set, so every interpreter test is also a frontend and verifier test.
os.environ.setdefault("TEGULA_VERIFY", "1")

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
MODES = ("interpret", "compiled")

_examples: dict[str, ModuleType] = {}


def load_example(stem: str) -> ModuleType:
    """Imports `examples/<stem>.py`, for example `load_example("01_vector_add")`."""
    mod = _examples.get(stem)
    if mod is None:
        name = f"tegula_example_{stem}"
        spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{stem}.py")
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        _examples[stem] = mod
    return mod


def mode_available(mode: str) -> bool:
    if mode == "interpret":
        return True
    from tegula.runtime import jit

    return jit.COMPILED_AVAILABLE


@pytest.fixture(params=MODES)
def mode(request: pytest.FixtureRequest) -> str:
    """Parametrizes a test over execution modes, skipping modes that don't exist yet."""
    if not mode_available(request.param):
        pytest.skip(f"{request.param} execution isn't available yet")
    return request.param


@contextlib.contextmanager
def execution_mode(mode: str) -> Iterator[None]:
    """Runs launches inside the block in `mode` ("interpret" or "compiled")."""
    old = os.environ.get("TEGULA_INTERPRET")
    os.environ["TEGULA_INTERPRET"] = "1" if mode == "interpret" else "0"
    try:
        yield
    finally:
        if old is None:
            del os.environ["TEGULA_INTERPRET"]
        else:
            os.environ["TEGULA_INTERPRET"] = old


# (atol, rtol) by output dtype. Kernels compute in fp32 and round once to the output dtype,
# so a few units in the last place of the output type is the expected error.
TOLERANCES: dict[Any, tuple[float, float]] = {
    np.dtype(np.float32): (1e-5, 1e-5),
    np.dtype(np.float16): (1e-3, 2e-3),
    np.dtype(ml_dtypes.bfloat16): (1e-2, 1.6e-2),
}


def assert_close(actual: Any, expected: Any, atol: float | None = None,
                 rtol: float | None = None, what: str = "output") -> None:  # fmt: skip
    """Asserts equal shape and dtype, then closeness with dtype-aware tolerances.

    Integer and boolean outputs must match exactly.
    """
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, f"{what}: shape {actual.shape} != {expected.shape}"
    assert actual.dtype == expected.dtype, f"{what}: dtype {actual.dtype} != {expected.dtype}"
    if actual.dtype.kind in "iub":
        np.testing.assert_array_equal(actual, expected, err_msg=what)
        return
    d_atol, d_rtol = TOLERANCES.get(actual.dtype, (1e-5, 1e-5))
    np.testing.assert_allclose(
        actual.astype(np.float64), expected.astype(np.float64),
        atol=d_atol if atol is None else atol, rtol=d_rtol if rtol is None else rtol,
        equal_nan=True, err_msg=what,
    )  # fmt: skip


def check_kernel(
    run: Callable[..., Any],
    args: Sequence[Any],
    reference: Callable[..., Any],
    *,
    kwargs: Mapping[str, Any] | None = None,
    modes: Sequence[str] = MODES,
    atol: float | None = None,
    rtol: float | None = None,
) -> None:
    """Runs a kernel wrapper in each mode and compares it with a NumPy reference.

    `run(*args, **kwargs)` launches the kernel and returns an output array or a tuple of
    them. `reference(*args, **kwargs)` returns the expected outputs. When several modes run,
    their outputs are also compared with each other. Modes that aren't available yet are
    skipped; if none is available, the test is skipped.

    Args:
        run: The host wrapper, such as `add` from `examples/01_vector_add.py`.
        args: Positional inputs, shared by `run` and `reference`.
        reference: The NumPy reference.
        kwargs: Keyword arguments for both `run` and `reference`.
        modes: The execution modes to check: "interpret", "compiled", or both.
        atol: An absolute tolerance that overrides the dtype default.
        rtol: A relative tolerance that overrides the dtype default.
    """
    kwargs = dict(kwargs or {})
    active = [m for m in modes if mode_available(m)]
    if not active:
        pytest.skip(f"none of the modes {tuple(modes)} is available yet")
    expected = reference(*args, **kwargs)
    expected = expected if isinstance(expected, tuple) else (expected,)
    outputs: dict[str, tuple[Any, ...]] = {}
    for m in active:
        with execution_mode(m):
            out = run(*args, **kwargs)
        out = out if isinstance(out, tuple) else (out,)
        assert len(out) == len(expected), f"{m}: {len(out)} outputs, expected {len(expected)}"
        for i, (a, e) in enumerate(zip(out, expected, strict=True)):
            assert_close(a, e, atol, rtol, f"{m} output {i} vs NumPy")
        outputs[m] = out
    if len(outputs) == 2:
        for i, (a, b) in enumerate(zip(outputs["compiled"], outputs["interpret"], strict=True)):
            assert_close(a, b, atol, rtol, f"compiled vs interpreted output {i}")
