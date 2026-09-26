"""Benchmarking and comparison helpers."""

from __future__ import annotations

import statistics
from collections.abc import Callable
from typing import Any

import numpy as np

from tegula.runtime.device import get_device


def do_bench(
    fn: Callable[[], Any],
    warmup_ms: float = 50.0,
    rep: int = 20,
    return_mode: str = "median",
) -> float | list[float]:
    """Times `fn` on the GPU and returns milliseconds.

    Each repetition runs in its own command buffer, and the time comes from the
    command buffer's GPU timestamps, not the wall clock. `fn` must launch its work
    on Tegula's stream with `tegula.Tensor` arguments; a launch that synchronizes
    internally (for example, one with NumPy arguments) can't be timed.

    Args:
        fn: A function that launches one or more kernels.
        warmup_ms: GPU time to spend warming up before measuring. The GPU clocks
            down when idle, so cold measurements run slow.
        rep: The number of timed repetitions.
        return_mode: "median", "min", "mean", or "all" (a list of every sample).

    Returns:
        The GPU time in milliseconds, summarized by `return_mode`.
    """
    stream = get_device().stream
    old_flush = stream.flush_every
    stream.flush_every = 1 << 30  # keep each repetition in one command buffer
    try:
        stream.synchronize()
        spent = 0.0
        while spent < warmup_ms:
            fn()
            t0, t1 = stream.native.flush_timed()
            if t1 <= t0:
                raise RuntimeError(
                    "do_bench measured no GPU work; fn must launch kernels with tegula.Tensor "
                    "arguments and must not synchronize"
                )
            spent += (t1 - t0) * 1e3
        samples = []
        for _ in range(rep):
            fn()
            t0, t1 = stream.native.flush_timed()
            samples.append((t1 - t0) * 1e3)
    finally:
        stream.flush_every = old_flush
    if return_mode == "all":
        return samples
    if return_mode == "min":
        return min(samples)
    if return_mode == "mean":
        return statistics.fmean(samples)
    return statistics.median(samples)


def default_tolerance(dtype: Any) -> tuple[float, float]:
    """Returns (atol, rtol) suited to comparing results of `dtype`."""
    name = str(np.dtype(dtype.to_numpy() if hasattr(dtype, "to_numpy") else dtype))
    if name == "bfloat16":
        return 2e-2, 2e-2
    if name == "float16":
        return 2e-3, 2e-3
    if name == "float32":
        return 1e-5, 1e-5
    return 0.0, 0.0


def assert_close(actual: Any, expected: Any, atol: float | None = None, rtol: float | None = None):
    """Asserts that two arrays match within dtype-aware tolerances."""
    a = np.asarray(actual)
    e = np.asarray(expected)
    d_atol, d_rtol = default_tolerance(a.dtype)
    np.testing.assert_allclose(
        a.astype(np.float32) if a.dtype.kind == "V" or a.dtype.itemsize == 2 else a,
        e.astype(np.float32) if e.dtype.kind == "V" or e.dtype.itemsize == 2 else e,
        atol=d_atol if atol is None else atol,
        rtol=d_rtol if rtol is None else rtol,
    )
