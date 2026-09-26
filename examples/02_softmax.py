"""Row softmax, one row per program.

Run the demo with `ENCELADUS_INTERPRET=1 uv run python examples/02_softmax.py`.
"""

import numpy as np

import enceladus
import enceladus.language as tl


@enceladus.jit
def softmax_kernel(out_ptr, in_ptr, stride_in, stride_out, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * stride_in + cols, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    tl.store(out_ptr + row * stride_out + cols, num / tl.sum(num, axis=0), mask=mask)


def softmax(x: np.ndarray) -> np.ndarray:
    """Returns the softmax of each row of a 2D array, in the input dtype."""
    m, n = x.shape
    out = np.empty_like(x)
    item = x.itemsize
    softmax_kernel[(m,)](
        out, x, x.strides[0] // item, out.strides[0] // item, n,
        BLOCK=enceladus.next_power_of_2(n),
    )  # fmt: skip
    return out


def reference(x: np.ndarray) -> np.ndarray:
    z = x.astype(np.float64)
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return (z / z.sum(axis=1, keepdims=True)).astype(x.dtype)


if __name__ == "__main__":
    x = np.random.default_rng(0).standard_normal((64, 1000)).astype(np.float32)
    print("max abs error:", np.abs(softmax(x) - reference(x)).max())
