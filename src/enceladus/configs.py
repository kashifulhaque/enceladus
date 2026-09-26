"""Pre-validated autotuning configurations.

The matmul list comes from the tile-shape sweep in `docs/research/05-kernel-bench.md`.
Each SIMD group computes an SM x SN strip, where SM = BM / WM and SN = BN / WN. The
valid strips are 16x32, 16x64, and 32x16 for every dtype, plus 32x32 for float16 and
bfloat16. Strips that measured 7-15x slower from register spilling are excluded: FP32
with 32x32 or 16x128, and any dtype with 64x32.
"""

from __future__ import annotations

from typing import Any

import numpy as np

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


_FP32_NAMES = ("float32", "fp32", "f32")
_HALF_NAMES = ("float16", "fp16", "f16", "half", "bfloat16", "bf16")


def _dtype_name(dtype: Any) -> str:
    """Returns the bare name of a tl, NumPy, ml_dtypes, torch, or MLX dtype, or of a string."""
    if isinstance(dtype, str):
        name = dtype
    elif hasattr(dtype, "long_name"):  # a tl dtype, whose `name` is the short form
        name = dtype.long_name
    else:
        try:  # NumPy dtypes and scalar classes, including ml_dtypes.bfloat16
            name = np.dtype(dtype).name
        except (TypeError, ValueError):  # torch.float16 and mlx.core.float16 print dotted
            name = str(dtype)
    return name.rsplit(".", 1)[-1]


def matmul_configs(dtype: Any = "float16") -> list[Config]:
    """Returns the matmul configurations to autotune over for `dtype`.

    Args:
        dtype: The operand dtype: float32, float16, or bfloat16, as a tl dtype, a NumPy
            dtype or scalar type, a torch or MLX dtype, or a name such as `"float32"`.

    Raises:
        ValueError: `dtype` isn't float32, float16, or bfloat16.
    """
    name = _dtype_name(dtype)
    if name in _FP32_NAMES:
        shapes = _COMMON
    elif name in _HALF_NAMES:
        shapes = _COMMON + _HALF_ONLY
    else:
        raise ValueError(f"matmul_configs supports float32, float16, and bfloat16, but got "
                         f"{dtype!r}")  # fmt: skip
    return [Config({"BM": bm, "BN": bn, "BK": bk}, num_warps=nw, dot_warps=dw)
            for bm, bn, bk, nw, dw in shapes]  # fmt: skip
