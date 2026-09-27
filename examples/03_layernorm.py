"""LayerNorm forward, one row per program, looping over the row in blocks.

`layernorm` accepts NumPy arrays, `enceladus.Tensor` objects, PyTorch tensors on the
`mps` device, and MLX arrays, and returns arrays of the same kind.

Run the demo with `ENCELADUS_INTERPRET=1 uv run python examples/03_layernorm.py`.
"""

import numpy as np

import enceladus
import enceladus.language as tl


@enceladus.jit
def layernorm_kernel(x_ptr, y_ptr, w_ptr, b_ptr, mean_ptr, rstd_ptr, stride_x, stride_y,
                     n_cols, eps, BLOCK: tl.constexpr):  # fmt: skip
    row = tl.program_id(0)
    x_ptr += row * stride_x
    y_ptr += row * stride_y
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        acc += tl.load(x_ptr + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
    mean = tl.sum(acc, axis=0) / n_cols
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        a = tl.load(x_ptr + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
        d = tl.where(cols < n_cols, a - mean, 0.0)
        acc += d * d
    var = tl.sum(acc, axis=0) / n_cols
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)
    for off in range(0, n_cols, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < n_cols
        w = tl.load(w_ptr + cols, mask=mask)
        b = tl.load(b_ptr + cols, mask=mask)
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, (x - mean) * rstd * w + b, mask=mask)


def layernorm(x, w, b, eps: float = 1e-5, block: int | None = None):
    """Returns `(y, mean, rstd)` for a 2D input normalized over its last dimension.

    The rows can be strided, but the elements within a row must be contiguous.
    """
    m, n = x.shape
    y = enceladus.new_empty(x)
    mean = enceladus.new_empty(x, (m,), np.float32)
    rstd = enceladus.new_empty(x, (m,), np.float32)
    block = block or min(1024, enceladus.next_power_of_2(n))
    stride_x, stride_y = enceladus.element_strides(x)[0], enceladus.element_strides(y)[0]
    layernorm_kernel[(m,)](x, y, w, b, mean, rstd, stride_x, stride_y, n, eps, BLOCK=block)
    return y, mean, rstd


def reference(x, w, b, eps=1e-5, block=None):
    z = x.astype(np.float64)
    mean = z.mean(axis=1)
    var = z.var(axis=1)
    rstd = 1 / np.sqrt(var + eps)
    y = (z - mean[:, None]) * rstd[:, None] * w.astype(np.float64) + b.astype(np.float64)
    return y.astype(x.dtype), mean.astype(np.float32), rstd.astype(np.float32)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.standard_normal((32, 1000)).astype(np.float32)
    w = rng.standard_normal(1000).astype(np.float32)
    b = rng.standard_normal(1000).astype(np.float32)
    y, _, _ = layernorm(x, w, b, block=256)
    print("max abs error:", np.abs(y - reference(x, w, b)[0]).max())
