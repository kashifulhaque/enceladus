"""Fused elementwise op: `gelu(x * scale + bias)`, with a bias per column.

`fused_gelu` accepts NumPy arrays, `enceladus.Tensor` objects, PyTorch tensors on the
`mps` device, and MLX arrays, and returns an array of the same kind.

Run the demo with `ENCELADUS_INTERPRET=1 uv run python examples/05_fused_gelu.py`.
"""

import math

import numpy as np

import enceladus
import enceladus.language as tl


@enceladus.jit
def fused_gelu_kernel(x_ptr, bias_ptr, out_ptr, n, n_cols, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    b = tl.load(bias_ptr + offs % n_cols, mask=mask).to(tl.float32)
    y = x * scale + b
    tl.store(out_ptr + offs, 0.5 * y * (1.0 + tl.erf(y * 0.7071067811865476)), mask=mask)


def fused_gelu(x, bias, scale: float, block: int = 1024):
    """Returns `gelu(x * scale + bias)` for a contiguous 2D `x` and a bias per column."""
    out = enceladus.new_empty(x)
    n = int(np.prod(x.shape))
    fused_gelu_kernel[(enceladus.cdiv(n, block),)](x, bias, out, n, x.shape[-1], scale, BLOCK=block)
    return out


def reference(x: np.ndarray, bias: np.ndarray, scale: float, **_) -> np.ndarray:
    y = x.astype(np.float64) * np.float32(scale) + bias.astype(np.float64)
    erf = np.vectorize(math.erf, otypes=[np.float64])
    return (0.5 * y * (1.0 + erf(y / math.sqrt(2.0)))).astype(x.dtype)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 300)).astype(np.float32)
    bias = rng.standard_normal(300).astype(np.float32)
    print("max abs error:", np.abs(fused_gelu(x, bias, 0.5) - reference(x, bias, 0.5)).max())
