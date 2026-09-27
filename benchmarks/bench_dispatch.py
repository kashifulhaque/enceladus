"""Measures launch overhead: sustained batched launches and synchronous round trips.

The PyTorch rows launch through `torch.mps.compile_shader` on PyTorch's MPS stream. The
MLX rows measure the synchronous MLX path, and lazy launches (`enceladus.lazy_mlx`)
against MLX's own ops. Rows for a framework that isn't installed are skipped.

Run with `uv run python benchmarks/bench_dispatch.py`.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import enceladus
import enceladus.language as tl

SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void vadd(device const float* x [[buffer(0)]], device const float* y [[buffer(1)]],
                 device float* o [[buffer(2)]], constant uint& n [[buffer(3)]],
                 uint i [[thread_position_in_grid]]) {
    if (i < n) o[i] = x[i] + y[i];
}
"""


def sustained(launch, n: int = 10_000, reps: int = 5, sync=enceladus.synchronize) -> list[float]:
    """Returns µs per launch for `reps` runs of `n` back-to-back launches.

    `sync` waits for the stream that `launch` uses.
    """
    out = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        for _ in range(n):
            launch()
        sync()
        out.append((time.perf_counter() - t0) / n * 1e6)
    return out


def round_trip(launch, n: int = 300, sync=enceladus.synchronize) -> list[float]:
    """Returns µs for `n` launch-and-wait round trips."""
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        launch()
        sync()
        out.append((time.perf_counter() - t0) * 1e6)
    return out


def report(name: str, samples: list[float]) -> None:
    print(f"{name:<40} min {min(samples):8.2f} µs   median {statistics.median(samples):8.2f} µs")


def main() -> None:
    dev = enceladus.get_device()
    print(f"Device: {dev.caps.name} ({dev.caps.architecture}), flush every "
          f"{dev.stream.flush_every} dispatches")
    k = enceladus.metal_kernel(SRC, "vadd")
    n = 1024
    x, y, o = enceladus.randn(n), enceladus.randn(n), enceladus.empty(n)
    launcher = k[(n // 256,), (256,)]

    def launch() -> None:
        launcher(x, y, o, n)

    # Warm up the GPU clocks: the GPU runs slowly for the first ~50 ms of work.
    sustained(launch, n=20_000, reps=1)
    report("raw metal_kernel, sustained", sustained(launch))
    report("raw metal_kernel, sync round trip", round_trip(launch))

    # Native floor: the stream's dispatch call with prebuilt arguments.
    stream = dev.stream.native
    plan = k._plan((True, True, True, False))
    bufs, offs, scal = [x.buffer, y.buffer, o.buffer], [0, 0, 0], n.to_bytes(4, "little")

    def native() -> None:
        stream.dispatch(k.pipeline, plan, bufs, offs, scal, (n // 256, 1, 1), (256, 1, 1))

    report("native Stream.dispatch, sustained", sustained(native))

    # @enceladus.jit hit path: binding, specialization key, packing, and dispatch.
    @enceladus.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs,
                 mask=mask), mask=mask)  # fmt: skip

    def jit_launch() -> None:
        add_kernel[(1,)](x, y, o, n, BLOCK=1024)

    jit_launch()
    report("@enceladus.jit launch, sustained", sustained(jit_launch))
    bench_torch(add_kernel, n)
    bench_mlx(add_kernel, n)


def bench_torch(add_kernel, n: int) -> None:
    """Measures launches through the PyTorch path against PyTorch's own floors."""
    try:
        import torch
    except ImportError:
        return
    if not torch.backends.mps.is_available():
        return
    sync = torch.mps.synchronize
    x, y, o = (torch.randn(n, device="mps") for _ in range(3))

    def jit_launch() -> None:
        add_kernel[(1,)](x, y, o, n, BLOCK=1024)

    jit_launch()
    sustained(jit_launch, n=20_000, reps=1, sync=sync)  # warm up
    report("@enceladus.jit on torch MPS, sustained", sustained(jit_launch, sync=sync))
    report("@enceladus.jit on torch MPS, sync round trip", round_trip(jit_launch, sync=sync))

    # Floors: the same compile_shader kernel called directly, and a built-in torch op.
    lib = torch.mps.compile_shader(SRC)

    def direct() -> None:
        lib.vadd(x, y, o, n, threads=(n,), group_size=(256,), arg_casts={3: "int32"})

    report("torch compile_shader call, sustained", sustained(direct, sync=sync))
    report("torch.add(out=), sustained", sustained(lambda: torch.add(x, y, out=o), sync=sync))


def bench_mlx(add_kernel, n: int) -> None:
    """Measures the synchronous MLX path, and lazy launches against MLX's own ops."""
    try:
        import mlx.core as mx
    except ImportError:
        return
    x, y, o = mx.random.normal((n,)), mx.random.normal((n,)), mx.zeros((n,))
    mx.eval(x, y, o)

    def jit_launch() -> None:
        add_kernel[(1,)](x, y, o, n, BLOCK=1024)

    jit_launch()
    report("@enceladus.jit on MLX arrays, sync round trip", round_trip(jit_launch))

    # Lazy launches: a chain of 1,000 dependent adds, each into a fresh output, then one
    # evaluation. MLX's own `x + y` chain is the floor.
    def lazy_chain() -> Any:
        acc = y
        for _ in range(1000):
            out = enceladus.new_empty(x)
            add_kernel[(1,)](x, acc, out, n, BLOCK=1024)
            acc = out
        return acc

    def mlx_chain() -> Any:
        acc = y
        for _ in range(1000):
            acc = x + acc
        return acc

    enceladus.lazy_mlx(True)
    try:
        for name, chain in [("@enceladus.jit on MLX arrays, lazy chain", lazy_chain),
                            ("MLX x + y chain", mlx_chain)]:  # fmt: skip
            mx.eval(chain())  # warm up
            samples = []
            for _ in range(10):
                t0 = time.perf_counter()
                mx.eval(chain())
                samples.append((time.perf_counter() - t0) / 1000 * 1e6)
            report(name, samples)
    finally:
        enceladus.lazy_mlx(False)


if __name__ == "__main__":
    main()
