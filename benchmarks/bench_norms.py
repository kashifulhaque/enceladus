"""LayerNorm and RMSNorm forward at 4096 x 4096, using the kernels in `examples/`.

GB/s counts the minimum traffic: one read and one write of the matrix. The LayerNorm
example reads each row three times (Triton-tutorial style); the rereads mostly hit the
cache.

Run with `uv run python benchmarks/bench_norms.py`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import enceladus
from enceladus.testing import do_bench

ROOT = Path(__file__).resolve().parents[1]
M = N = 4096


def load(stem: str):
    spec = importlib.util.spec_from_file_location(stem, ROOT / "examples" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ln, rms = load("03_layernorm"), load("06_rmsnorm")
    print(f"Device: {enceladus.get_device().caps.name}; {M} x {N}")
    for dtype in ("float32", "float16"):
        x = enceladus.randn(M, N, dtype=dtype)
        y = enceladus.empty_like(x)
        w, b = enceladus.randn(N, dtype=dtype), enceladus.randn(N, dtype=dtype)
        mean, rstd = enceladus.empty(M), enceladus.empty(M)
        nbytes = 2 * M * N * x.itemsize
        for block in (1024, 4096):
            def run_ln(block=block):
                ln.layernorm_kernel[(M,)](x, y, w, b, mean, rstd, N, N, N, 1e-5, BLOCK=block,
                                          num_warps=8)  # fmt: skip

            ms = do_bench(run_ln, rep=20, return_mode="min")
            print(f"LayerNorm {dtype} BLOCK={block:<5} {ms:7.3f} ms {nbytes / ms / 1e6:7.1f} GB/s")

        def run_rms():
            rms.rmsnorm_kernel[(M,)](x, w, y, N, N, N, 1e-6, BLOCK=N, num_warps=8)

        ms = do_bench(run_rms, rep=20, return_mode="min")
        print(f"RMSNorm   {dtype} BLOCK={N:<5} {ms:7.3f} ms {nbytes / ms / 1e6:7.1f} GB/s")


if __name__ == "__main__":
    main()
