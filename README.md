# Enceladus

Enceladus is a Triton-like Python language for writing GPU kernels for Apple silicon
Macs. You write a kernel as a Python function that operates on tiles, and Enceladus
compiles it to Metal Shading Language (MSL) and runs it on the GPU through Metal.

Enceladus has the following properties:

- **Triton's programming model.** Kernels use `@enceladus.jit`, `kernel[grid](...)`,
  `tl.constexpr`, and a `tl` namespace that mirrors `triton.language`, so most Triton
  kernels port by changing the imports.
- **A light install.** The compiler is pure Python with a small native runtime. You
  don't need Xcode, the Metal toolchain, or an LLVM build.
- **Framework interop.** Kernels accept NumPy arrays, PyTorch tensors on the `mps`
  device, and MLX arrays without copying them, and PyTorch launches run in order with
  PyTorch's own operations.
- **Correctness tools.** A NumPy interpreter runs any kernel on the CPU, unsupported
  code fails with an error at the Python source line, and `tl.device_print` and
  `tl.device_assert` work on the GPU.

## Install

Enceladus needs a Mac with Apple silicon, macOS 15 or later, and Python 3.11 or later.
To add it to a [uv](https://docs.astral.sh/uv/) project, run the following command:

```bash
uv add enceladus
```

To install PyTorch or MLX support too, use the `torch` or `mlx` extra, for example
`uv add "enceladus[torch]"`.

## Example

The following program adds two vectors on the GPU:

```python
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


x = np.random.default_rng(0).standard_normal(100_000, dtype=np.float32)
y = np.ones_like(x)
out = np.empty_like(x)
add_kernel[(enceladus.cdiv(x.size, 1024),)](x, y, out, x.size, BLOCK=1024)
np.testing.assert_allclose(out, x + y)
```

Each program of the grid adds one block of 1,024 elements, and the mask keeps the last
block inside the arrays. To run the same kernel on the CPU with `print()` and `pdb`
support, set `ENCELADUS_INTERPRET=1`.

## Documentation

The [user guide](docs/guide/index.md) has the following pages:

- [Quickstart](docs/guide/quickstart.md)
- [Programming model](docs/guide/programming-model.md)
- [Language reference](docs/guide/language-reference.md)
- [Debugging](docs/guide/debugging.md)
- [Framework interop](docs/guide/interop.md)
- [Porting from Triton](docs/guide/porting-from-triton.md)
- [Performance tips](docs/guide/performance.md)

The [`examples/`](examples/) directory has complete kernels, including LayerNorm, a
fused matmul epilogue, flash attention, a histogram, and a cumulative sum.

## Build from source

To build Enceladus from a clone of the repository and run the tests, run the following
commands:

```bash
uv sync
uv run pytest -q
```

After you change the native code in `src/enceladus/_C/`, rebuild it with
`uv sync --reinstall-package enceladus`. To build release wheels, run
`scripts/build_wheels.sh`.

The design and the milestone plan are in [`PLAN.md`](PLAN.md), and the results of each
milestone are in [`docs/progress.md`](docs/progress.md).
