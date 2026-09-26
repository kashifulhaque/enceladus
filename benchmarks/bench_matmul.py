"""Matmul TFLOPS against MLX and PyTorch MPS.

Enceladus uses GPU timestamps. MLX and PyTorch use wall clock around a synchronized call
(overhead is under 1% at 4096^3). Operands come from memory, never constants.

Run with `uv run python benchmarks/bench_matmul.py [--quick]`.
"""

from __future__ import annotations

import importlib.util
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


def main() -> None:
    ex = load("04_matmul")
    fused = load("07_matmul_fused")
    shapes = SHAPES[:1] if "--quick" in sys.argv else SHAPES
    print(f"Device: {enceladus.get_device().caps.name}")
    print(f"{'shape':<18}{'dtype':<10}{'config':<22}{'TFLOPS':>8}{'MLX':>8}{'torch':>8}")
    for m, n, k in shapes:
        for dtype in ("float32", "float16", "bfloat16"):
            a, b = enceladus.randn(m, k, dtype=dtype), enceladus.randn(k, n, dtype=dtype)
            c = enceladus.empty((m, n), dtype)
            refs = [mlx_ms(m, n, k, dtype), torch_ms(m, n, k, dtype)]
            ref = "".join(f"{tflops(m, n, k, r):8.2f}" if r else f"{'n/a':>8}" for r in refs)
            for bm, bn, bk, nw in ((64, 64, 32, 4),):
                def run(bm=bm, bn=bn, bk=bk, nw=nw):
                    ex.matmul_desc(a, b, c, bm, bn, bk, nw)

                ms = do_bench(run, rep=10, return_mode="min")
                cfg = f"{bm}x{bn}x{bk} w{nw}"
                print(f"{f'{m}x{n}x{k}':<18}{dtype:<10}{cfg:<22}{tflops(m, n, k, ms):8.2f}{ref}")
            ex.matmul_tuned(a, b, c)  # tunes on first use

            def run_tuned():
                ex.matmul_tuned(a, b, c)

            tms = do_bench(run_tuned, rep=10, return_mode="min")
            best = ex.matmul_desc_tuned.config_for(a, b, c, m, n, k, k, n, n)
            tcfg = f"tuned {best.kwargs['BM']}x{best.kwargs['BN']} w{best.dot_warps}"
            print(f"{'':<18}{dtype:<10}{tcfg:<22}{tflops(m, n, k, tms):8.2f}")
            if (m, n, k) == SHAPES[0] and dtype != "bfloat16":
                bias = enceladus.randn(n, dtype=dtype)

                def run_fused():
                    fused.matmul_bias_gelu(a, b, bias, c)

                fms = do_bench(run_fused, rep=10, return_mode="min")
                print(f"{'':<18}{dtype:<10}{'+ bias + GELU':<22}{tflops(m, n, k, fms):8.2f}"
                      f"  ({fms / ms - 1:+.1%} vs plain)")


if __name__ == "__main__":
    main()
