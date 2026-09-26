"""Matmul TFLOPS for both `tl.dot` backends, against MLX and PyTorch MPS.

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

import enceladus
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


if __name__ == "__main__":
    main()
