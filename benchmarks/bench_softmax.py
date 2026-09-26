"""Row softmax bandwidth at 4096 x 4096 against MLX and PyTorch MPS, plus a reduction sweep.

GB/s counts one read and one write of the matrix. Enceladus uses GPU timestamps; MLX and
PyTorch use wall clock around a synchronized call, so they include about 0.1 ms of
launch and sync overhead.

Run with `uv run python benchmarks/bench_softmax.py`.
"""

from __future__ import annotations

import time

import enceladus
import enceladus.language as tl
from enceladus.testing import do_bench

M = N = 4096


@enceladus.jit
def softmax_kernel(out_ptr, in_ptr, stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * stride + cols, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)
    tl.store(out_ptr + row * stride + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


@enceladus.jit
def rowsum_kernel(out_ptr, in_ptr, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    x = tl.load(in_ptr + row * stride + tl.arange(0, BLOCK))
    tl.store(out_ptr + row, tl.sum(x, axis=0))


def gbps(nbytes: int, ms: float) -> float:
    return nbytes / (ms * 1e-3) / 1e9


def wall_min(fn, sync, reps: int = 20) -> float:
    for _ in range(10):
        fn()
    sync()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync()
        out.append((time.perf_counter() - t0) * 1e3)
    return min(out)


def mlx_ms(dtype: str) -> float | None:
    try:
        import mlx.core as mx
    except ImportError:
        return None
    x = mx.random.normal((M, N)).astype(getattr(mx, dtype))
    mx.eval(x)
    return wall_min(lambda: mx.eval(mx.softmax(x, axis=-1)), lambda: None)


def torch_ms(dtype: str) -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.backends.mps.is_available():
        return None
    x = torch.randn(M, N, device="mps", dtype=getattr(torch, dtype))
    return wall_min(lambda: torch.softmax(x, dim=-1), torch.mps.synchronize)


def main() -> None:
    print(f"Device: {enceladus.get_device().caps.name}; softmax {M} x {N}")
    print(f"{'dtype':<10}{'num_warps':>10}{'ms':>9}{'GB/s':>8}{'MLX':>8}{'torch':>8}")
    for dtype in ("float32", "float16"):
        x = enceladus.randn(M, N, dtype=dtype)
        out = enceladus.empty_like(x)
        nbytes = 2 * M * N * x.itemsize
        ref = [mlx_ms(dtype), torch_ms(dtype)]
        refs = "".join(f"{gbps(nbytes, r):8.1f}" if r else f"{'n/a':>8}" for r in ref)
        for nw in (4, 8, 16):
            def run(x=x, out=out, nw=nw):
                softmax_kernel[(M,)](out, x, N, N, BLOCK=N, num_warps=nw)

            ms = do_bench(run, rep=20, return_mode="min")
            print(f"{dtype:<10}{nw:>10}{ms:9.3f}{gbps(nbytes, ms):8.1f}{refs}")

    # Reduction microbenchmark: row sums across threadgroup sizes.
    x = enceladus.randn(M, N)
    out = enceladus.empty(M)
    print("\nRow sum 4096 x 4096 fp32, by threads per threadgroup:")
    results = {}
    for nw in (4, 8, 16, 32):
        def run(nw=nw):
            rowsum_kernel[(M,)](out, x, N, BLOCK=N, num_warps=nw)

        results[nw * 32] = gbps(M * N * 4, do_bench(run, rep=20, return_mode="min"))
        print(f"  {nw * 32:>5} threads: {results[nw * 32]:6.1f} GB/s")
    spread = (max(results.values()) - min(results.values())) / max(results.values())
    print(f"  spread between 128 and 1024 threads: {spread:.1%}")


if __name__ == "__main__":
    main()
