"""Row cumsum and histogram throughput, with MLX's cumsum for comparison.

GB/s counts one read and one write of the matrix for cumsum, and one read of the input
for the histogram. Enceladus uses GPU timestamps; MLX uses wall clock around a
synchronized call, so it includes about 0.1 ms of launch and sync overhead.

Run with `uv run python benchmarks/bench_scan_atomics.py`.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import enceladus
from enceladus.testing import do_bench

M = N = 4096
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _example(stem: str):
    spec = importlib.util.spec_from_file_location(stem, EXAMPLES / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gbps(nbytes: int, ms: float) -> float:
    return nbytes / (ms * 1e-3) / 1e9


def mlx_cumsum_ms() -> float | None:
    try:
        import mlx.core as mx
    except ImportError:
        return None
    x = mx.random.normal((M, N))
    mx.eval(x)
    for _ in range(10):
        mx.eval(mx.cumsum(x, axis=1))
    out = []
    for _ in range(20):
        t0 = time.perf_counter()
        mx.eval(mx.cumsum(x, axis=1))
        out.append((time.perf_counter() - t0) * 1e3)
    return min(out)


def main() -> None:
    print(f"Device: {enceladus.get_device().caps.name}")
    cs = _example("10_cumsum")
    x = enceladus.randn(M, N)
    out = enceladus.empty_like(x)
    ref = mlx_cumsum_ms()
    print(f"Row cumsum {M} x {N} fp32 (MLX: "
          f"{f'{gbps(2 * M * N * 4, ref):.1f} GB/s' if ref else 'n/a'})")  # fmt: skip
    for nw in (4, 8, 16):
        def run(nw=nw):
            cs.cumsum_kernel[(M,)](out, x, M, N, N, N, REVERSE=False, BLOCK_M=1, BLOCK_N=N,
                                   num_warps=nw)  # fmt: skip

        ms = do_bench(run, rep=20, return_mode="min")
        print(f"  num_warps={nw:<3} {ms:7.3f} ms {gbps(2 * M * N * 4, ms):7.1f} GB/s")

    hist = _example("09_histogram")
    n = 1 << 24
    data = enceladus.randn(n)
    for bins in (64, 4096):
        h = enceladus.zeros(bins, dtype="int32")

        def run(h=h, bins=bins):
            hist.histogram_kernel[(enceladus.cdiv(n, 1024),)](
                data, h, n, -4.0, bins / 8.0, NUM_BINS=bins, BLOCK=1024)

        ms = do_bench(run, rep=20, return_mode="min")
        print(f"Histogram of {n} fp32 values, {bins} bins: {ms:7.3f} ms "
              f"{gbps(n * 4, ms):7.1f} GB/s")  # fmt: skip


if __name__ == "__main__":
    main()
