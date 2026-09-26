"""Flash attention forward TFLOPS against MLX `mx.fast.scaled_dot_product_attention`.

Enceladus uses GPU timestamps. MLX uses the wall clock around a synchronized call, which
adds about 0.1 ms, under 2% at these sizes. The two run interleaved in one process, round
after round, so clock and thermal drift affect both alike. Each round reports the minimum
and the median of 10 runs; the table shows the best round and the median of the round
medians.

FLOPs count the two matmuls, `4 * batch * heads * seq_len^2 * head_dim`, halved for a
causal mask.

Run with `uv run python benchmarks/bench_attention.py [--quick]`.
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import enceladus
from enceladus.testing import do_bench

ROOT = Path(__file__).resolve().parents[1]
# (batch, heads, seq_len, head_dim)
SHAPES = [(1, 16, 2048, 64), (1, 16, 2048, 128), (1, 16, 4096, 64), (1, 16, 4096, 128)]
ROUNDS, REPS = 3, 10


def load(stem: str):
    spec = importlib.util.spec_from_file_location(stem, ROOT / "examples" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tflops(shape, causal: bool, ms: float) -> float:
    z, h, n, d = shape
    flops = 4 * z * h * n * n * d / (2 if causal else 1)
    return flops / (ms * 1e-3) / 1e12


def wall(fn, reps: int = REPS) -> list[float]:
    for _ in range(3):
        fn()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def main() -> None:
    import mlx.core as mx

    ex = load("08_flash_attention")
    shapes = SHAPES[:2] if "--quick" in sys.argv else SHAPES
    print(f"Device: {enceladus.get_device().caps.name}; float16; TFLOPS as min (median)")
    print(f"{'shape':<20}{'causal':<8}{'config':<14}{'Enceladus':>16}{'MLX':>16}{'ratio':>8}")
    for shape in shapes:
        z, h, n, d = shape
        q, k, v = (enceladus.randn(*shape, dtype="float16", seed=i) for i in range(3))
        o = enceladus.empty(shape, "float16")
        mq, mk, mv = (mx.array(np.array(t.numpy())) for t in (q, k, v))
        mx.eval(mq, mk, mv)
        scale = 1.0 / d**0.5
        for causal in (False, True):
            ex.attention_tuned(q, k, v, causal, o=o)  # tunes on first use
            best = ex.attention_tuned_kernel.config_for(q, k, v, o, scale, n, n * d, d,
                                                        HEAD_DIM=d, CAUSAL=causal)  # fmt: skip
            mask = "causal" if causal else None

            def run_ours():
                ex.attention_tuned(q, k, v, causal, o=o)

            def run_mlx():
                mx.eval(mx.fast.scaled_dot_product_attention(mq, mk, mv, scale=scale,
                                                             mask=mask))  # fmt: skip

            ours, theirs = [], []
            for _ in range(ROUNDS):
                ours.append(do_bench(run_ours, rep=REPS, return_mode="all"))
                theirs.append(wall(run_mlx))
            o_min = min(min(r) for r in ours)
            o_med = statistics.median(statistics.median(r) for r in ours)
            m_min = min(min(r) for r in theirs)
            m_med = statistics.median(statistics.median(r) for r in theirs)
            cfg = f"{best.kwargs['BLOCK_M']}x{best.kwargs['BLOCK_N']} w{best.num_warps}"
            ot = f"{tflops(shape, causal, o_min):.2f} ({tflops(shape, causal, o_med):.2f})"
            mt = f"{tflops(shape, causal, m_min):.2f} ({tflops(shape, causal, m_med):.2f})"
            print(f"{str(shape):<20}{str(causal):<8}{cfg:<14}{ot:>16}{mt:>16}"
                  f"{o_min / m_min:>7.2f}x")  # fmt: skip


if __name__ == "__main__":
    main()
