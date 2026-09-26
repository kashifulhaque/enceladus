"""Measures launch overhead: sustained batched launches and synchronous round trips.

Run with `uv run python benchmarks/bench_dispatch.py`.
"""

from __future__ import annotations

import statistics
import time

import tegula
import tegula.language as tl

SRC = """
#include <metal_stdlib>
using namespace metal;
kernel void vadd(device const float* x [[buffer(0)]], device const float* y [[buffer(1)]],
                 device float* o [[buffer(2)]], constant uint& n [[buffer(3)]],
                 uint i [[thread_position_in_grid]]) {
    if (i < n) o[i] = x[i] + y[i];
}
"""


def sustained(launch, n: int = 10_000, reps: int = 5) -> list[float]:
    """Returns µs per launch for `reps` runs of `n` back-to-back launches."""
    out = []
    for _ in range(reps):
        tegula.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            launch()
        tegula.synchronize()
        out.append((time.perf_counter() - t0) / n * 1e6)
    return out


def round_trip(launch, n: int = 300) -> list[float]:
    """Returns µs for `n` launch-and-wait round trips."""
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        launch()
        tegula.synchronize()
        out.append((time.perf_counter() - t0) * 1e6)
    return out


def report(name: str, samples: list[float]) -> None:
    print(f"{name:<40} min {min(samples):8.2f} µs   median {statistics.median(samples):8.2f} µs")


def main() -> None:
    dev = tegula.get_device()
    print(f"Device: {dev.caps.name} ({dev.caps.architecture}), flush every "
          f"{dev.stream.flush_every} dispatches")
    k = tegula.metal_kernel(SRC, "vadd")
    n = 1024
    x, y, o = tegula.randn(n), tegula.randn(n), tegula.empty(n)
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

    # @tegula.jit hit path: binding, specialization key, packing, and dispatch.
    @tegula.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs,
                 mask=mask), mask=mask)  # fmt: skip

    def jit_launch() -> None:
        add_kernel[(1,)](x, y, o, n, BLOCK=1024)

    jit_launch()
    report("@tegula.jit launch, sustained", sustained(jit_launch))


if __name__ == "__main__":
    main()
