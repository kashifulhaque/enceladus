# Framework interop

Enceladus kernels accept NumPy arrays, `enceladus.Tensor` objects, PyTorch tensors on
the `mps` device, and MLX arrays as array arguments, without copying them. This page
explains how each kind of array binds to a kernel, how launches order with the
framework's own work, and how to share Enceladus tensors with other frameworks.

Importing Enceladus doesn't import PyTorch or MLX. Enceladus detects their arrays by
type. To install a framework together with Enceladus, use the `torch` or `mlx` extra,
for example `uv add "enceladus[torch]"`.

## Write framework-neutral host code

The following helpers let one host function serve every kind of array:

- `enceladus.new_empty(like, shape=None, dtype=None)` allocates an array of the same
  kind and device as `like`: a NumPy array, an `enceladus.Tensor`, a PyTorch tensor, or
  an MLX array. `shape` and `dtype` default to those of `like`. `dtype` can be a NumPy
  dtype, such as `np.float32`, or a dtype of `like`'s framework.
- `enceladus.new_zeros(like, shape=None, dtype=None)` does the same, and fills the
  array with zeros.
- `enceladus.element_strides(x)` returns the strides of any supported array in elements.
  NumPy reports strides in bytes and MLX doesn't expose them, so pass this function's
  result to kernels instead.
- `enceladus.element_dtype(x)` returns the element type of any supported array as a
  NumPy dtype, for host code that chooses an output type.

The following program defines a function that adds two arrays of any supported kind and
returns an array of the same kind, and calls it on each kind:

```python
import mlx.core as mx
import numpy as np
import torch

import enceladus
import enceladus.language as tl


@enceladus.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def add(x, y):
    out = enceladus.new_empty(x)
    n = int(np.prod(x.shape))
    add_kernel[(enceladus.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
    return out


a = np.arange(3000, dtype=np.float32)
for x in (a, enceladus.from_numpy(a), torch.from_numpy(a).to("mps"), mx.array(a)):
    out = add(x, x)
    print(type(out).__name__, out[:3])
```

The examples in the repository's `examples/` directory follow this pattern.

## PyTorch

A PyTorch tensor must be on the `mps` device. A CPU tensor raises a `TypeError` that
suggests `.to("mps")`, and `float64` tensors aren't supported. Views work, including
views with a nonzero storage offset: Enceladus binds the tensor's storage buffer at the
view's byte offset.

### Ordering with PyTorch

Enceladus's stream and PyTorch's MPS stream are separate queues, and Metal doesn't order
work between them. So when every array argument of a launch is an MPS tensor, Enceladus
compiles the kernel a second time with `torch.mps.compile_shader` and runs it on
PyTorch's own stream. The kernel then runs in order with the PyTorch operations before
and after it, and you don't need to synchronize anything yourself:

```python
import torch

import enceladus
import enceladus.language as tl


@enceladus.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


n = 100_000
x = torch.randn(n, device="mps")
y = torch.randn(n, device="mps")
out = torch.empty_like(x)
x.mul_(2)  # a PyTorch op before the kernel
add_kernel[(enceladus.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
result = out - y  # a PyTorch op after the kernel sees its output
torch.testing.assert_close(result, x)
print("PyTorch ordering OK")
```

In a loop of small launches, a launch through `compile_shader` costs about 4.4 µs,
against about 3.2 µs on Enceladus's own stream. Both paths compile with the same math
settings, so a kernel gives bit-identical results on PyTorch tensors and on other arrays.

### Fallback to synchronized launches

Some launches can't run on PyTorch's stream:

- The launch mixes PyTorch tensors with other kinds of arrays.
- `compile_shader` rejects the kernel, for example because it needs a later MSL version
  than `compile_shader` uses.
- The kernel calls `tl.device_print` or, with `ENCELADUS_DEBUG=1`, `tl.device_assert`.

These launches take a synchronized path instead. Enceladus waits for PyTorch's stream,
launches on its own stream, and waits for the kernel to finish. The results are
correct, but each launch costs about 100 µs. The `enceladus` logger records a warning the
first time each kernel takes this path, with the reason:

```text
WARNING:enceladus:enceladus: kernel `add_kernel` runs on Enceladus's own queue with a
sync before and after each launch (about 100 µs per launch): it mixes PyTorch tensors
with other array types
```

## MLX

By default, MLX launches are synchronous. MLX arrays are lazy, so Enceladus evaluates
each MLX argument with `mx.eval()` before it binds the array's buffer. It also waits for
MLX's queued work, which might still read an array that the kernel writes. Each launch
then waits for the GPU, which costs about 100 µs. To add launches to MLX's lazy graph
instead, see [Lazy MLX launches](#lazy-mlx-launches).

MLX treats arrays as immutable, but an Enceladus kernel writes to its output arguments
in place. An output must therefore be an array that you allocate for that purpose, such
as `mx.zeros(shape)` followed by `mx.eval()`, or `enceladus.new_empty(like)`. Don't
write to an array that MLX might share with other arrays, such as a lazily computed
result or a view. Writing to a broadcast array, such as one from `mx.broadcast_to`,
raises a `ValueError`.

The following program runs a kernel on MLX arrays:

```python
import mlx.core as mx

import enceladus
import enceladus.language as tl


@enceladus.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


n = 100_000
x = mx.random.normal((n,))
y = x * 3  # lazy; Enceladus evaluates it before the launch
out = mx.zeros((n,))
mx.eval(out)  # allocate the output's own buffer
add_kernel[(enceladus.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
assert mx.allclose(out, x * 4).item()
print("MLX OK")
```

### Lazy MLX launches

With `enceladus.lazy_mlx(True)`, a launch on MLX arrays can join MLX's lazy graph through
`mx.fast.metal_kernel`. The launch returns right away, and MLX runs the kernel when
something evaluates its results, in order with MLX's own operations. MLX arrays are
immutable, so a lazy launch doesn't write in place: each array that the kernel writes
takes the value of a new array, as `out[...] = result` would.

A launch is lazy when it meets the following conditions:

- Every array argument is an MLX array.
- Every array that the kernel writes is a *fresh output*: an array from
  `enceladus.new_empty` or `enceladus.new_zeros` that no launch has written yet. In
  lazy mode, both functions return a lazy zero-filled array, and the kernel sees a
  zero-filled output. A kernel can read a fresh output, for example to accumulate into
  it with atomics.
- The kernel doesn't call `tl.device_print` or, with `ENCELADUS_DEBUG=1`,
  `tl.device_assert`.

Any other launch on MLX arrays, such as a kernel that updates an array in place, takes
the synchronized path. The `enceladus` logger records a warning the first time each
kernel does, with the reason. Don't modify a fresh output yourself, for example with
`out[0] = 1`, before you pass it to a kernel: the lazy launch ignores its contents.

The following program chains 100 lazy launches and evaluates the result once:

```python
import mlx.core as mx

import enceladus
import enceladus.language as tl


@enceladus.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def add(x, y):
    out = enceladus.new_empty(x)  # a fresh output
    add_kernel[(enceladus.cdiv(x.size, 1024),)](x, y, out, x.size, BLOCK=1024)
    return out


enceladus.lazy_mlx(True)
x = mx.random.normal((100_000,))
y = x
for _ in range(100):
    y = add(x, y)  # returns without waiting for the GPU
assert mx.allclose(y, x * 101).item()  # evaluates the whole chain
enceladus.lazy_mlx(False)
print("lazy MLX OK")
```

Lazy launches give the same results as synchronized ones, bit for bit. In lazy mode,
Enceladus doesn't evaluate MLX arrays to learn their offsets, so kernels don't
specialize on the alignment of MLX arguments other than fresh outputs.
`enceladus.element_strides(x)` still evaluates `x`, except for a fresh output. A lazy
launch doesn't check that a strided view spans fewer than 2^31 elements, because MLX
reports strides only for evaluated arrays.

## NumPy

Enceladus binds a NumPy array's memory directly: it wraps the memory pages that the
array occupies, without copying. If Metal can't wrap the memory, Enceladus copies the
array to the GPU and copies it back after the launch. Arrays with negative strides
raise an error; pass `np.ascontiguousarray(x)` instead. `float64` arrays aren't
supported.

Because NumPy has no stream, a launch with a NumPy argument waits for the GPU before it
returns. To batch many launches, call `enceladus.async_numpy(True)`, and then call
`enceladus.synchronize()` before you read the arrays. Enceladus keeps the arrays alive
until that synchronization. For an example, see
[Streams and synchronization](programming-model.md#streams-and-synchronization).

## Share Enceladus tensors through DLPack

An `enceladus.Tensor` exports itself through DLPack with the Metal device type, so
PyTorch and MLX can import it without copying. Both sides then share the same memory:

```python
import mlx.core as mx
import torch

import enceladus

t = enceladus.arange(8, dtype="float32")
tt = torch.from_dlpack(t)  # a torch tensor on the mps device
tt.mul_(10)
torch.mps.synchronize()
print(t.numpy())  # [ 0. 10. 20. 30. 40. 50. 60. 70.]
m = mx.from_dlpack(t)  # an MLX array over the same buffer
print(m)
```

The export waits for Enceladus's pending launches, so the importer sees their results.
It orders nothing after that: if you launch Enceladus kernels on the original tensor
later, call `enceladus.synchronize()` before the other framework reads it. Launches on
the imported PyTorch tensor run on PyTorch's stream and order themselves.

## What's next

- For how launches order with the CPU, see
  [Streams and synchronization](programming-model.md#streams-and-synchronization).
- To reduce launch overhead, see [Performance tips](performance.md).
