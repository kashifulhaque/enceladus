"""Vector add bandwidth at 256 MB per array, compared with MLX, plus Enceladus compile time.

Run with `uv run python benchmarks/bench_elementwise.py`.
"""

from __future__ import annotations

import importlib.util
import statistics
import time
from pathlib import Path

import numpy as np

import enceladus
from enceladus.compiler.frontend import build_ir
from enceladus.compiler.pipeline import compile_module
from enceladus.testing import do_bench

ROOT = Path(__file__).resolve().parents[1]


def load(stem: str):
    spec = importlib.util.spec_from_file_location(stem, ROOT / "examples" / f"{stem}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gbps(nbytes: int, ms: float) -> float:
    return nbytes / (ms * 1e-3) / 1e9


def bench_mlx(n: int, dtype: str) -> float | None:
    try:
        import mlx.core as mx
    except ImportError:
        return None
    x = mx.random.normal((n,)).astype(getattr(mx, dtype))
    y = mx.random.normal((n,)).astype(getattr(mx, dtype))
    mx.eval(x, y)
    for _ in range(10):
        mx.eval(x + y)
    samples = []
    for _ in range(20):
        t0 = time.perf_counter()
        mx.eval(x + y)
        samples.append((time.perf_counter() - t0) * 1e3)
    return min(samples)


def compile_time_ms(kernel, args, consts, reps: int = 50) -> float:
    spec = kernel.specialize(kernel.bind(args, consts))
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        m = build_ir(kernel, spec.arg_types, spec.arg_facts, spec.constexprs, 4, "relaxed")
        compile_module(m)
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def main() -> None:
    ex = load("01_vector_add")
    print(f"Device: {enceladus.get_device().caps.name}")
    print(f"{'kernel':<34}{'min ms':>9}{'GB/s':>9}{'MLX GB/s':>10}")
    # 256 MB per float32 and int8 array. int8 shows the effect of vector loads most.
    for dtype, n in (("float32", 1 << 26), ("float16", 1 << 26), ("int8", 1 << 28)):
        if dtype == "int8":
            x = enceladus.from_numpy(np.ones(n, np.int8))
            y = enceladus.from_numpy(np.ones(n, np.int8))
        else:
            x, y = enceladus.randn(n, dtype=dtype), enceladus.randn(n, dtype=dtype)
        out = enceladus.empty_like(x)
        nbytes = 3 * n * x.itemsize
        for block in (1024, 4096):
            grid = (enceladus.cdiv(n, block),)

            def run(grid=grid, x=x, y=y, out=out, block=block):
                ex.add_kernel[grid](x, y, out, n, BLOCK=block)

            ms = do_bench(run, rep=20, return_mode="min")
            mlx_ms = bench_mlx(n, dtype)
            mlx = f"{gbps(nbytes, mlx_ms):10.1f}" if mlx_ms else f"{'n/a':>10}"
            print(f"{f'add {dtype} BLOCK={block}':<34}{ms:9.3f}{gbps(nbytes, ms):9.1f}{mlx}")
        del x, y, out
    x = enceladus.randn(1024)
    t = compile_time_ms(ex.add_kernel, (x, x, x, 1024), {"BLOCK": 1024})
    print(f"Enceladus compile time (frontend to MSL), vector add: {t:.2f} ms")


if __name__ == "__main__":
    main()
