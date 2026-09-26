"""RMSNorm forward, one row per program: `x * rsqrt(mean(x^2) + eps) * w`.

Run the demo with `TEGULA_INTERPRET=1 uv run python examples/06_rmsnorm.py`.
"""

import numpy as np

import tegula
import tegula.language as tl


@tegula.jit
def rmsnorm_kernel(x_ptr, w_ptr, out_ptr, stride_x, stride_out, n_cols, eps,
                   BLOCK: tl.constexpr):  # fmt: skip
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / n_cols + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * stride_out + cols, x * rstd * w, mask=mask)


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Returns RMSNorm of each row of a 2D array, in the input dtype."""
    m, n = x.shape
    out = np.empty_like(x)
    rmsnorm_kernel[(m,)](x, w, out, x.strides[0] // x.itemsize, out.strides[0] // out.itemsize,
                         n, eps, BLOCK=tegula.next_power_of_2(n))  # fmt: skip
    return out


def reference(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    z = x.astype(np.float64)
    rstd = 1 / np.sqrt((z * z).mean(axis=1, keepdims=True) + eps)
    return (z * rstd * w.astype(np.float64)).astype(x.dtype)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.standard_normal((32, 1000)).astype(np.float32)
    w = rng.standard_normal(1000).astype(np.float32)
    print("max abs error:", np.abs(rmsnorm(x, w) - reference(x, w)).max())
