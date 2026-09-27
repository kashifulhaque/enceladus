"""Histogram of float data with `tl.atomic_add`.

Each program bins `BLOCK` values and adds one count per value to a shared histogram in
device memory. Values outside `[lo, hi)` are skipped. `histogram` accepts NumPy arrays,
`enceladus.Tensor` objects, PyTorch tensors on the `mps` device, and MLX arrays, and
returns an array of the same kind.

Run the demo with `uv run python examples/09_histogram.py`.
"""

import numpy as np

import enceladus
import enceladus.language as tl


@enceladus.jit
def histogram_kernel(x_ptr, hist_ptr, n, lo, inv_width, NUM_BINS: tl.constexpr,
                     BLOCK: tl.constexpr):  # fmt: skip
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=0.0).to(tl.float32)
    b = tl.floor((x - lo) * inv_width).to(tl.int32)
    ok = (offs < n) & (b >= 0) & (b < NUM_BINS)
    tl.atomic_add(hist_ptr + b, 1, mask=ok)


def histogram(x, num_bins: int, lo: float, hi: float, block: int = 1024):
    """Returns the `int32` counts of `x` in `num_bins` equal bins over `[lo, hi)`."""
    hist = enceladus.new_zeros(x, (num_bins,), np.int32)
    n = int(np.prod(x.shape))
    histogram_kernel[(enceladus.cdiv(n, block),)](
        x, hist, n, float(lo), num_bins / (hi - lo), NUM_BINS=num_bins, BLOCK=block,
    )  # fmt: skip
    return hist


def reference(x: np.ndarray, num_bins: int, lo: float, hi: float) -> np.ndarray:
    xs = x.astype(np.float32)
    b = np.floor((xs - np.float32(lo)) * np.float32(num_bins / (hi - lo))).astype(np.int64)
    b = b[(b >= 0) & (b < num_bins)]
    return np.bincount(b, minlength=num_bins).astype(np.int32)


if __name__ == "__main__":
    x = np.random.default_rng(0).standard_normal(1_000_000).astype(np.float32)
    got = histogram(x, 64, -4.0, 4.0)
    print("counts match:", np.array_equal(got, reference(x, 64, -4.0, 4.0)))
