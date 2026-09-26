"""Vector add: `out = x + y`.

`add` accepts NumPy arrays, `enceladus.Tensor` objects, PyTorch tensors on the `mps`
device, and MLX arrays, and returns an array of the same kind.

Run the demo with `uv run python examples/01_vector_add.py`.
"""

import numpy as np

import enceladus
import enceladus.language as tl


@enceladus.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def add(x, y, block: int = 1024):
    """Returns `x + y` for two contiguous arrays of the same shape and dtype."""
    out = enceladus.new_empty(x)
    n = int(np.prod(x.shape))
    add_kernel[(enceladus.cdiv(n, block),)](x, y, out, n, BLOCK=block)
    return out


def reference(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x + y


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100_003).astype(np.float32)
    y = rng.standard_normal(100_003).astype(np.float32)
    out = add(x, y)
    print("max abs error:", np.abs(out - reference(x, y)).max())
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.backends.mps.is_available():
        tx, ty = torch.from_numpy(x).to("mps"), torch.from_numpy(y).to("mps")
        print("max abs error, torch:", (add(tx, ty) - (tx + ty)).abs().max().item())
