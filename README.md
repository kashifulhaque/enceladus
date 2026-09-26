# Enceladus

Enceladus is a Triton-like Python language for writing GPU kernels for Apple silicon
Macs. You write a kernel as a Python function that operates on tiles, and Enceladus
compiles it to Metal Shading Language and runs it on the GPU. You don't need Xcode or
the Metal toolchain. Kernels accept NumPy arrays, PyTorch `mps` tensors, and MLX arrays.

## Install

Enceladus needs an Apple silicon Mac with macOS 15 or later and Python 3.11 or later.
To install it, run the following command:

```bash
uv add enceladus
```

For PyTorch or MLX support, add the `torch` or `mlx` extra: `uv add "enceladus[torch]"`.

## Example

The following program adds two vectors on the GPU:

```python
import numpy as np

import enceladus
import enceladus.language as tl


@enceladus.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


x = np.random.default_rng(0).standard_normal(100_000, dtype=np.float32)
y = np.ones_like(x)
out = np.empty_like(x)
add_kernel[(enceladus.cdiv(x.size, 1024),)](x, y, out, x.size, BLOCK=1024)
np.testing.assert_allclose(out, x + y)
```

To run a kernel on the CPU for debugging, set `ENCELADUS_INTERPRET=1`.

## Learn more

- [User guide](https://github.com/kashifulhaque/enceladus/blob/main/docs/guide/index.md)
- [Examples](https://github.com/kashifulhaque/enceladus/tree/main/examples), including
  a fused matmul, flash attention, and a histogram
- [Porting from Triton](https://github.com/kashifulhaque/enceladus/blob/main/docs/guide/porting-from-triton.md)

To build from source, run `uv sync`, and then run the tests with `uv run pytest -q`.
