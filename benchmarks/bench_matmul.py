"""Matmul TFLOPS for both `tl.dot` backends, against MLX and PyTorch MPS.

It also reports exact `int8` matmul (`int32` accumulation) in TOPS. MLX's matmul takes
only floating-point types, and PyTorch MPS `int8` matmul returns `int8`, so neither has a
comparable `int8` number.

Enceladus uses GPU timestamps. MLX and PyTorch use wall clock around a synchronized call
(overhead is under 1% at 4096^3). Operands come from memory, never constants.

Run with `uv run python benchmarks/bench_matmul.py [--quick]`.
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import enceladus
import enceladus.language as tl
from enceladus.testing import do_bench

ROOT = Path(__file__).resolve().parents[1]
SHAPES = [(4096, 4096, 4096), (2000, 2000, 2000), (513, 513, 513), (1024, 4096, 1024)]


def load(stem: str):
    spec = importlib.util.spec_from_file_location(stem, ROOT / "examples" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tflops(m: int, n: int, k: int, ms: float) -> float:
    return 2 * m * n * k / (ms * 1e-3) / 1e12


def wall_min(fn, sync, reps: int = 10) -> float:
    for _ in range(3):
        fn()
    sync()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync()
        out.append((time.perf_counter() - t0) * 1e3)
    return min(out)


def mlx_ms(m, n, k, dtype):
    try:
        import mlx.core as mx
    except ImportError:
        return None
    a = mx.random.normal((m, k)).astype(getattr(mx, dtype))
    b = mx.random.normal((k, n)).astype(getattr(mx, dtype))
    mx.eval(a, b)
    return wall_min(lambda: mx.eval(a @ b), lambda: None)


def torch_ms(m, n, k, dtype):
    try:
        import torch
    except ImportError:
        return None
    a = torch.randn(m, k, device="mps", dtype=getattr(torch, dtype))
    b = torch.randn(k, n, device="mps", dtype=getattr(torch, dtype))
    return wall_min(lambda: a @ b, torch.mps.synchronize)


def interleaved(variants: dict, rounds: int = 3, rep: int = 10) -> dict[str, list[float]]:
    """Times each variant `rounds` times in turn and returns all samples per variant.

    Alternating the variants spreads GPU clock changes and other GPU users across them.
    """
    samples: dict[str, list[float]] = {name: [] for name in variants}
    for _ in range(rounds):
        for name, fn in variants.items():
            samples[name] += do_bench(fn, rep=rep, return_mode="all")
    return samples


def row(label: str, dtype: str, cfg: str, flops: float, ms: list[float], extra: str = "") -> None:
    best, med = min(ms), statistics.median(ms)
    print(f"{label:<18}{dtype:<10}{cfg:<24}{flops / best / 1e9:8.2f}{flops / med / 1e9:8.2f}"
          f"{extra}")  # fmt: skip


@enceladus.jit
def int8_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr,
                       BK: tl.constexpr):  # fmt: skip
    pid_n, pid_m = tl.program_id(0), tl.program_id(1)
    a = tl.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [BM, BK])
    b = tl.make_tensor_descriptor(b_ptr, [K, N], [N, 1], [BK, BN])
    c = tl.make_tensor_descriptor(c_ptr, [M, N], [N, 1], [BM, BN])
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k in range(0, K, BK):
        acc = tl.dot(a.load([pid_m * BM, k]), b.load([k, pid_n * BN]), acc)
    c.store([pid_m * BM, pid_n * BN], acc)


def int8_variants(m: int, n: int, k: int) -> dict:
    """Returns exact int8 matmul launches, after checking one against NumPy."""
    rng = np.random.default_rng(0)
    a_np = rng.integers(-128, 128, (m, k), dtype=np.int8)
    b_np = rng.integers(-128, 128, (k, n), dtype=np.int8)
    a, b = enceladus.from_numpy(a_np), enceladus.from_numpy(b_np)
    c = enceladus.empty((m, n), "int32")
    out = {}
    for bm, bn, bk, nw in ((64, 64, 32, 4), (64, 64, 64, 4), (128, 64, 32, 8)):
        grid = (enceladus.cdiv(n, bn), enceladus.cdiv(m, bm))
        out[f"exact {bm}x{bn}x{bk} w{nw}"] = (
            lambda grid=grid, bm=bm, bn=bn, bk=bk, nw=nw: int8_matmul_kernel[grid](
                a, b, c, m, n, k, BM=bm, BN=bn, BK=bk, num_warps=nw))
    if m * n * k <= 1 << 33:
        next(iter(out.values()))()
        want = (a_np.astype(np.int64) @ b_np.astype(np.int64)).astype(np.int32)
        if not np.array_equal(c.numpy(), want):
            raise AssertionError(f"int8 matmul {m}x{n}x{k} differs from NumPy")
    return out


def main() -> None:
    ex = load("04_matmul")
    fused = load("07_matmul_fused")
    shapes = SHAPES[:1] if "--quick" in sys.argv else SHAPES
    print(f"Device: {enceladus.get_device().caps.name}")
    print("TFLOPS as the minimum and median time over 3 interleaved rounds of 10 runs.")
    print(f"{'shape':<18}{'dtype':<10}{'config':<24}{'TFLOPS':>8}{'median':>8}{'MLX':>8}"
          f"{'torch':>8}")  # fmt: skip
    for m, n, k in shapes:
        flops = 2 * m * n * k
        for dtype in ("float32", "float16", "bfloat16"):
            a, b = enceladus.randn(m, k, dtype=dtype), enceladus.randn(k, n, dtype=dtype)
            c = enceladus.empty((m, n), dtype)
            refs = [mlx_ms(m, n, k, dtype), torch_ms(m, n, k, dtype)]
            ref = "".join(f"{tflops(m, n, k, r):8.2f}" if r else f"{'n/a':>8}" for r in refs)
            ex.matmul_tuned(a, b, c)  # tunes on first use
            best = ex.matmul_desc_tuned.config_for(a, b, c, m, n, k, k, n, n)
            variants = {
                "simdgroup 64x64x32 w4": lambda: ex.matmul_desc(a, b, c, 64, 64, 32, 4,
                                                                 "simdgroup"),
                "mpp 64x64 w4": lambda: ex.matmul_desc(a, b, c, 64, 64, 32, 4, "mpp"),
                f"tuned {best.kwargs['BM']}x{best.kwargs['BN']} "
                f"{'mpp' if best.dot_backend == 'mpp' else 'sg'} w{best.num_warps}":
                    lambda: ex.matmul_tuned(a, b, c),
            }  # fmt: skip
            if (m, n, k) == SHAPES[0] and dtype != "bfloat16":
                bias = enceladus.randn(n, dtype=dtype)
                for backend in ("simdgroup", "mpp"):
                    variants[f"{backend} + bias + GELU"] = (
                        lambda backend=backend: fused.matmul_bias_gelu(a, b, bias, c,
                                                                       dot_backend=backend))
            res = interleaved(variants)
            label = f"{m}x{n}x{k}"
            for i, (cfg, ms) in enumerate(res.items()):
                row(label if i == 0 else "", dtype, cfg, flops, ms, ref if i == 0 else "")
        res = interleaved(int8_variants(m, n, k))
        for cfg, ms in res.items():
            row("", "int8", cfg, flops, ms, f"{'n/a':>8}{'n/a':>8}")


if __name__ == "__main__":
    main()
