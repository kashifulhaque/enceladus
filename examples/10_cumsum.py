"""Row-wise cumulative sum with `tl.cumsum`, one block of rows per program.

Each program loads a `BLOCK_M x BLOCK_N` tile that covers `BLOCK_M` whole rows and scans
it along the rows. Set `reverse=True` for a suffix sum. `cumsum` accepts NumPy arrays,
`enceladus.Tensor` objects, PyTorch tensors on the `mps` device, and MLX arrays, and
returns an array of the same kind.

Run the demo with `uv run python examples/10_cumsum.py`.
"""

import numpy as np

import enceladus
import enceladus.language as tl


@enceladus.jit
def cumsum_kernel(out_ptr, in_ptr, n_rows, n_cols, stride_in, stride_out,
                  REVERSE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):  # fmt: skip
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)
    x = tl.load(in_ptr + rows[:, None] * stride_in + cols[None, :], mask=mask)
    y = tl.cumsum(x, axis=1, reverse=REVERSE)
    tl.store(out_ptr + rows[:, None] * stride_out + cols[None, :], y, mask=mask)


def cumsum(x, reverse: bool = False):
    """Returns the cumulative sum of each row of a 2D array.

    Integers narrower than 32 bits sum in `int32`, as in `tl.cumsum`.
    """
    m, n = x.shape
    dt = enceladus.element_dtype(x)
    out = enceladus.new_empty(x, dtype=np.int32 if dt.kind in "iub" and dt.itemsize < 4 else dt)
    block_n = enceladus.next_power_of_2(n)
    block_m = max(1, min(16, 4096 // block_n))
    cumsum_kernel[(enceladus.cdiv(m, block_m),)](
        out, x, m, n, enceladus.element_strides(x)[0], enceladus.element_strides(out)[0],
        REVERSE=reverse, BLOCK_M=block_m, BLOCK_N=block_n,
    )  # fmt: skip
    return out


def reference(x: np.ndarray, reverse: bool = False) -> np.ndarray:
    out_dtype = np.int32 if x.dtype.kind in "iub" and x.dtype.itemsize < 4 else x.dtype
    acc = out_dtype if x.dtype.kind in "iub" else np.float32  # bfloat16 has kind "V"
    z = x.astype(acc)
    if reverse:
        return np.flip(np.cumsum(np.flip(z, 1), axis=1, dtype=acc), 1).astype(out_dtype)
    return np.cumsum(z, axis=1, dtype=acc).astype(out_dtype)


if __name__ == "__main__":
    x = np.random.default_rng(0).standard_normal((300, 1000)).astype(np.float32)
    print("max abs error:", np.abs(cumsum(x) - reference(x)).max())
