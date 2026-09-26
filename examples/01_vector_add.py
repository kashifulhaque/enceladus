"""Vector add: `out = x + y`.

Run the demo with `ENCELADUS_INTERPRET=1 uv run python examples/01_vector_add.py`.
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


def add(x: np.ndarray, y: np.ndarray, block: int = 1024) -> np.ndarray:
    """Returns `x + y` for two contiguous arrays of the same shape and dtype."""
    out = np.empty_like(x)
    n = x.size
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
