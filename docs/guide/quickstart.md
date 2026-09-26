# Quickstart

This page shows you how to install Enceladus and run three kernels on your Mac's GPU: a
vector add, a row softmax, and a matrix multiplication. Each example is a complete
program that you can copy into a file and run.

## Before you begin

To use Enceladus, you need the following:

- A Mac with Apple silicon (M1 or later).
- macOS 15 or later.
- Python 3.11 or later.
- [uv](https://docs.astral.sh/uv/), the Python package and project manager.

You don't need Xcode or the Metal toolchain. Enceladus compiles each kernel to Metal
Shading Language (MSL) and has the Metal framework compile it while your program runs.

## Install Enceladus

To add Enceladus to your uv project, run the following command:

```bash
uv add enceladus
```

If you plan to pass PyTorch tensors or MLX arrays to kernels, install the matching extra
instead, for example `uv add "enceladus[torch]"` or `uv add "enceladus[mlx]"`.

To check that Enceladus finds your GPU, run the following command:

```bash
uv run python -c "import enceladus; print(enceladus.get_device())"
```

The output is similar to the following:

```text
<enceladus.Device Apple M4 Pro (applegpu_g16s)>
```

## Add two vectors

A kernel is a Python function decorated with `@enceladus.jit`. When you launch it, the
GPU runs many copies of it, called *programs*, and each program handles one block of
the data. The following program adds two vectors of 98,432 elements in blocks of 1,024:

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


n = 98_432
x = enceladus.randn(n, seed=0)
y = enceladus.randn(n, seed=1)
out = enceladus.empty_like(x)
grid = (enceladus.cdiv(n, 1024),)
add_kernel[grid](x, y, out, n, BLOCK=1024)

np.testing.assert_allclose(out.numpy(), x.numpy() + y.numpy())
print("vector add matches NumPy")
```

Save the program as `add.py` and run it with `uv run python add.py`.

The kernel works as follows:

- `tl.program_id(0)` returns the index of the current program. The grid,
  `(enceladus.cdiv(n, 1024),)`, launches enough programs to cover all `n` elements.
- `tl.arange(0, BLOCK)` creates a *tile*: a block of 1,024 values that the program
  processes at once. `offs` holds the element indices of this program's block.
- `x_ptr + offs` is a tile of pointers. `tl.load` reads through them, and `mask` keeps
  the last program from reading or writing past the end of the arrays.
- `BLOCK: tl.constexpr` makes the block size a compile-time constant. Enceladus compiles
  one version of the kernel for each value you pass.

`enceladus.randn` and `enceladus.empty_like` allocate `enceladus.Tensor` objects in
memory that the CPU and GPU share. The launch returns before the GPU finishes, and
`out.numpy()` waits for the result. Kernels also accept NumPy arrays, PyTorch tensors on
the `mps` device, and MLX arrays.

## Compute a row softmax

A kernel can reduce a tile to a single value. The following program computes the softmax
of each row of a 512 x 1,000 NumPy array, one row per program:

```python
import numpy as np

import enceladus
import enceladus.language as tl


@enceladus.jit
def softmax_kernel(out_ptr, in_ptr, stride_in, stride_out, n_cols,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * stride_in + cols, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)
    x = x - tl.max(x, axis=0)
    num = tl.exp(x)
    tl.store(out_ptr + row * stride_out + cols, num / tl.sum(num, axis=0), mask=mask)


x = np.random.default_rng(0).standard_normal((512, 1000), dtype=np.float32)
out = np.empty_like(x)
n_rows, n_cols = x.shape
softmax_kernel[(n_rows,)](
    out, x, enceladus.element_strides(x)[0], enceladus.element_strides(out)[0], n_cols,
    BLOCK=enceladus.next_power_of_2(n_cols),
)

expected = np.exp(x - x.max(axis=1, keepdims=True))
expected /= expected.sum(axis=1, keepdims=True)
np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)
print("softmax matches NumPy")
```

Tile dimensions must be powers of two, so the kernel uses a block of 1,024 columns for
1,000-column rows and masks the rest. Masked lanes load `-inf`, which doesn't change the
row maximum and contributes 0 to the sum. `tl.max` and `tl.sum` reduce the tile across
all the threads of the program.

This launch uses NumPy arrays, so it waits for the GPU before it returns, and `out`
holds the result right away. `enceladus.element_strides` returns an array's strides in
elements for every supported array type.

## Multiply matrices

A *tensor descriptor* describes a 2D array by its shape and strides. It loads and stores
whole blocks and handles blocks that extend past the edges of the array. The following
program multiplies a 1,000 x 600 matrix by a 600 x 800 matrix, one 64 x 64 output block
per program:

```python
import numpy as np

import enceladus
import enceladus.language as tl


@enceladus.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_bk, stride_cm,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n, pid_m = tl.program_id(0), tl.program_id(1)
    a = tl.make_tensor_descriptor(a_ptr, [M, K], [stride_am, 1], [BM, BK])
    b = tl.make_tensor_descriptor(b_ptr, [K, N], [stride_bk, 1], [BK, BN])
    c = tl.make_tensor_descriptor(c_ptr, [M, N], [stride_cm, 1], [BM, BN])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc = tl.dot(a.load([pid_m * BM, k]), b.load([k, pid_n * BN]), acc)
    c.store([pid_m * BM, pid_n * BN], acc.to(c.dtype))


M, N, K = 1000, 800, 600
rng = np.random.default_rng(0)
a = rng.standard_normal((M, K), dtype=np.float32)
b = rng.standard_normal((K, N), dtype=np.float32)
c = np.empty((M, N), np.float32)
grid = (enceladus.cdiv(N, 64), enceladus.cdiv(M, 64))
matmul_kernel[grid](a, b, c, M, N, K, K, N, N, BM=64, BN=64, BK=32, num_warps=4)

np.testing.assert_allclose(c, a @ b, rtol=1e-4, atol=1e-3)
print("matmul matches NumPy")
```

The grid is two-dimensional: `tl.program_id(0)` selects the block column and
`tl.program_id(1)` the block row. `tl.dot` multiplies two tiles with the GPU's
SIMD-group matrix instructions and accumulates in `float32`. `num_warps=4` runs each program on 4 SIMD groups of
32 threads. Because the loads go through descriptors, `tl.dot` reads its operands
straight from device memory, which is the fastest way to write a matmul in Enceladus.

## Run a kernel on the CPU

Enceladus includes an interpreter that runs kernels on the CPU with NumPy. In the
interpreter, you can use `print()` and `pdb` inside a kernel. To run the vector add in
the interpreter, run the following command:

```bash
ENCELADUS_INTERPRET=1 uv run python add.py
```

The interpreter runs one program at a time, so it's much slower than the GPU. Use it to
debug a kernel and to check the compiled kernel's results.

## What's next

- To learn how programs, tiles, and synchronization work, see
  [Programming model](programming-model.md).
- To look up a `tl` function, see [Language reference](language-reference.md).
- To make kernels faster, see [Performance tips](performance.md).
- To use Enceladus with PyTorch or MLX, see [Framework interop](interop.md).
- To port an existing Triton kernel, see [Porting from Triton](porting-from-triton.md).
