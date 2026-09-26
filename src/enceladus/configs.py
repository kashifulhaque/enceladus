"""Pre-validated autotuning configurations.

The matmul list comes from the tile-shape sweep in `docs/research/05-kernel-bench.md`.
Each SIMD group computes an SM x SN strip, where SM = BM / WM and SN = BN / WN. The
valid strips are 16x32, 16x64, and 32x16 for every dtype, plus 32x32 for float16 and
bfloat16. Strips that measured 7-15x slower from register spilling are excluded: FP32
with 32x32 or 16x128, and any dtype with 64x32.
"""

from __future__ import annotations

from typing import Any

from enceladus.runtime.autotuner import Config

# (BM, BN, BK, num_warps, (WM, WN)) -> per-SIMD-group strip in the comment.
_COMMON = [
    (64, 64, 32, 4, (4, 1)),  # 16x64, the measured best
    (32, 64, 32, 2, (2, 1)),  # 16x64
    (128, 64, 32, 8, (8, 1)),  # 16x64
    (64, 32, 32, 4, (4, 1)),  # 16x32
    (32, 64, 32, 4, (2, 2)),  # 16x32
    (64, 32, 32, 4, (2, 2)),  # 32x16
]
_HALF_ONLY = [
    (64, 64, 32, 4, (2, 2)),  # 32x32
    (128, 64, 32, 8, (4, 2)),  # 32x32
]


def matmul_configs(dtype: Any = "float16") -> list[Config]:
    """Returns the matmul configurations to autotune over for `dtype`."""
    name = str(getattr(dtype, "name", dtype))
    shapes = _COMMON + ([] if name in ("float32", "fp32", "f32") else _HALF_ONLY)
    return [Config({"BM": bm, "BN": bn, "BK": bk}, num_warps=nw, dot_warps=dw)
            for bm, bn, bk, nw, dw in shapes]  # fmt: skip
