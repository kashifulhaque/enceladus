# Programming model

This page explains how Enceladus runs a kernel: programs and the launch grid, tiles and
masks, the two ways to address memory, compile-time constants and specialization, and
the ordering rules between your kernels, the CPU, and other GPU frameworks.

If you know Triton, most of this page is familiar. For the differences, see
[Porting from Triton](porting-from-triton.md).

## Programs and the grid

A kernel is a Python function decorated with `@enceladus.jit`. You launch it by
indexing it with a *grid* and calling the result:

<!-- snippet: skip -->
```python
kernel[grid](arg0, arg1, BLOCK=1024, num_warps=4)
```

The grid is a tuple of one to three non-negative integers. The GPU runs one *program*
for each point of the grid, and each program runs the whole kernel function. Inside the
kernel, `tl.program_id(axis)` returns the program's index along `axis`, and
`tl.num_programs(axis)` returns the grid's size along it. The GPU runs programs in no
particular order, so a program must not depend on the results of another program in the
same launch.

The grid can also be a function. Enceladus calls it with a dictionary of every argument
by name, including compile-time constants, and uses the tuple it returns. A function
grid is useful when an autotuner chooses the block size:

<!-- snippet: skip -->
```python
grid = lambda meta: (enceladus.cdiv(n, meta["BLOCK"]),)
```

Each program runs as one Metal threadgroup of `num_warps` SIMD groups, and each SIMD
group has 32 threads. `num_warps` is a launch option that must be a power of two from 1
to 32; the default is 4. You don't write code for individual threads. The compiler
decides which thread holds which element of each tile.

## Tiles

A *tile* is an n-dimensional block of values that one program holds and computes on at
once, like a small NumPy array. Every operation in a kernel works on whole tiles:
`x + y` adds two tiles elementwise, and `tl.sum(x, axis=0)` reduces a tile along an
axis. Tiles follow NumPy's broadcasting rules, and `x[:, None]` inserts a dimension.

Tile shapes have the following rules:

- Every dimension must be a compile-time power of two from 1 to 65,536. To process a
  length that isn't a power of two, round the block up with
  `enceladus.next_power_of_2` and mask the extra elements.
- A tile must fit in registers. If a tile needs more than 256 32-bit registers per
  thread, compilation fails with an error that suggests smaller blocks or a larger
  `num_warps`. For more than 128 registers, Enceladus prints a warning, because the
  kernel might spill and run slowly.
- Tiles are immutable. You can't assign to an element of a tile. Build another tile
  with `tl.where` instead.

A scalar is a value without a shape, such as a kernel argument `n` or the result of
`tl.program_id`. Scalars broadcast to any tile shape.

### Types

Enceladus supports boolean, signed and unsigned integer types from 8 to 64 bits, and
the floating-point types `float16`, `bfloat16`, and `float32`. It has no `float64`. For
the full list, see [Data types](language-reference.md#data-types).

Arithmetic follows Triton's promotion rules:

- A Python integer literal takes the other operand's type if it fits in that type. A
  Python float literal takes the other operand's floating-point type, or `float32`.
- Mixing an integer and a floating-point operand gives the floating-point type. Mixing
  `float16` and `bfloat16` gives `float32`.
- `/` on integers gives `float32`, and `//` needs integer operands.
- Operations on `float16` and `bfloat16` compute in `float32` and round each result
  back to the 16-bit type.

## Masks

A *mask* is a boolean tile, usually a comparison such as `offs < n`. Memory operations
take a mask to skip elements:

- `tl.load(ptrs, mask=mask, other=0.0)` reads only where `mask` is true, and returns
  `other` elsewhere.
- `tl.store(ptrs, value, mask=mask)` writes only where `mask` is true.
- Atomics take a mask too, and return 0 where it's false.

Masks keep a program from reading or writing out of bounds, and they let one block
size serve any problem size. Enceladus doesn't check bounds for you on pointer
accesses. To find an out-of-bounds access, see
[Check conditions on the GPU](debugging.md#check-conditions-on-the-gpu).

## Pointers and tensor descriptors

Enceladus gives you two ways to address memory.

### Pointers

An array argument arrives in the kernel as a pointer to its first element.
`x_ptr.dtype` is the element type. Adding an integer tile to a pointer gives a tile of
pointers, and you load and store through it with a mask:

<!-- snippet: skip -->
```python
offs = pid * BLOCK + tl.arange(0, BLOCK)
x = tl.load(x_ptr + offs, mask=offs < n, other=0.0)
```

Pointer tiles can express any access pattern: strided, gathered, or transposed. You
compute every address yourself and mask every access.

### Tensor descriptors

A *tensor descriptor* describes a strided array by its base pointer, its shape, and its
strides in elements. It loads and stores whole blocks at element offsets:

<!-- snippet: skip -->
```python
a = tl.make_tensor_descriptor(a_ptr, [M, K], [stride_am, 1], [BM, BK])
tile = a.load([pid_m * BM, k])        # zeros where the block leaves the array
c.store([pid_m * BM, pid_n * BN], acc)  # skips elements outside the array
```

A descriptor checks its own bounds, so you don't write masks. The innermost stride must
be 1. Descriptors also tell the compiler that a block is a dense 2D region, which lets
`tl.dot` load matrix fragments straight from device memory. Use descriptors for matrix
multiplication and attention, and pointer tiles for everything else.

## Compile-time constants and specialization

Enceladus compiles a separate version of a kernel, called a *specialization*, for each
combination of the following properties of the launch arguments:

- The value of each parameter annotated as `tl.constexpr`.
- The type of each runtime argument: the element type of each array, and `int32`,
  `int64`, `float32`, or `int1` for each Python scalar. A Python `int` is `int32` if it
  fits in 32 bits and `int64` otherwise. A Python `float` is `float32`.
- Facts about runtime arguments: whether an integer is divisible by 16, whether it
  equals 1, and whether an array's data is 16-byte aligned. These facts help the
  compiler, and they never remove an argument from the kernel's signature.
- The launch options `num_warps`, `dot_warps`, and `dot_backend`, and the
  `ENCELADUS_DEBUG` environment variable.

### Constexprs

A parameter annotated as `tl.constexpr` is a compile-time constant. Its value can be
an integer, a float, a boolean, a `tl` dtype, or `None`. The compiler
evaluates any expression over constexprs in Python, so a constexpr can set a tile shape,
choose a branch, or unroll a loop:

<!-- snippet: skip -->
```python
@enceladus.jit
def scale_kernel(x_ptr, out_ptr, n, HAS_SCALE: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n)
    if HAS_SCALE:  # decided at compile time; the other branch isn't compiled
        x = x * 2.0
    tl.store(out_ptr + offs, x, mask=offs < n)
```

Tile shapes, `tl.arange` bounds, and `tl.static_range` bounds must be compile-time
values. If you pass a runtime value where the compiler needs a constant, the error
message names the parameter to annotate.

### Recompilation

Each specialization costs a compile of several milliseconds, and Enceladus caches
compiled kernels in memory and on disk. If a kernel compiles more than 16 times in one
process, Enceladus warns and names the argument that changed most often. If that
argument is a size that varies from launch to launch, add it to `do_not_specialize` so
that its divisibility doesn't create more specializations:

<!-- snippet: skip -->
```python
@enceladus.jit(do_not_specialize=["n"])
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    ...
```

### Other `@enceladus.jit` options

`@enceladus.jit` also takes the following options:

- `interpret=True` runs every launch in the NumPy interpreter. By default, the
  `ENCELADUS_INTERPRET` environment variable decides.
- `math_mode="fast"` lets the Metal compiler assume that no value is infinite or NaN.
  The default, `"relaxed"`, keeps infinities and NaNs.

## Control flow and helper functions

A kernel can use the following Python control flow:

- `if` and `else` on a scalar condition. A condition on constexprs is decided at
  compile time. To choose between tile values elementwise, use `tl.where`.
- `for` loops over `range(...)`, `tl.range(...)`, or `tl.static_range(...)`.
  `tl.static_range` unrolls the loop at compile time. A variable that a loop updates
  must keep its type and shape across iterations.
- `and`, `or`, `not`, and conditional expressions on scalars. On tiles, use `&`, `|`,
  and `~`.

`while`, `break`, `continue`, `return` inside a runtime `if` or loop, `try`, `with`,
comprehensions, lambdas, and nested functions aren't supported. The compiler refuses
them with an `enceladus.CompilationError` that points at the source line.

A kernel can call other `@enceladus.jit` functions, which the compiler inlines. Unlike
kernels, these helper functions can return values, including tuples. A kernel can also
call plain Python functions with compile-time arguments, for example `math.log2(BLOCK)`.

The following program uses a helper function:

```python
import numpy as np

import enceladus
import enceladus.language as tl


@enceladus.jit
def gelu(x):
    return 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))


@enceladus.jit
def gelu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, gelu(x), mask=mask)


x = np.linspace(-3, 3, 5000, dtype=np.float32)
out = np.empty_like(x)
gelu_kernel[(enceladus.cdiv(x.size, 1024),)](x, out, x.size, BLOCK=1024)
print(out[:4])
```

## Streams and synchronization

Enceladus launches are asynchronous where it's safe. Launches go to a *stream*, an
ordered queue of GPU work like a CUDA stream, and Enceladus has one default stream per
process. Kernels on the stream run in launch order, so a kernel always sees the writes
of the kernels launched before it. The stream batches launches into Metal command
buffers and submits a batch every 64 launches (set `ENCELADUS_FLUSH_EVERY` to change
the size) or when the CPU needs results.

To wait for all launched work, call `enceladus.synchronize()`. Errors from the GPU,
such as a failed `tl.device_assert` or a failed command buffer, surface at the next
synchronization as `enceladus.DeviceAssertionError` or `enceladus.MetalError`.
Launch kernels from one thread at a time; the stream isn't thread-safe.

What a launch waits for depends on the kinds of arrays that you pass:

| Array arguments | Where the kernel runs | When results are visible |
|---|---|---|
| `enceladus.Tensor` only | Enceladus's stream, asynchronously | After a sync. `Tensor.numpy()`, `tolist()`, `print()`, and `np.asarray()` sync for you. |
| NumPy arrays, alone or with tensors | Enceladus's stream; the launch waits | When the launch returns. With `enceladus.async_numpy(True)`, after `enceladus.synchronize()`. |
| PyTorch MPS tensors only | PyTorch's MPS stream, through `torch.mps.compile_shader` | In order with surrounding PyTorch operations; no Enceladus sync needed. |
| MLX arrays | Enceladus's stream; the launch evaluates its inputs first and waits | When the launch returns. |
| PyTorch tensors with other kinds | Enceladus's stream, between syncs of both streams | When the launch returns. |

The following sections describe each case.

### `enceladus.Tensor`

An `enceladus.Tensor` lives in memory that the CPU and GPU share. Launches with only
tensor arguments return right away, which makes them the fastest way to run many small
kernels. Enceladus allocates tensors with `enceladus.empty`, `zeros`, `ones`, `full`,
`randn`, `rand`, `arange`, `empty_like`, and `zeros_like`, and wraps NumPy arrays with
`enceladus.from_numpy`. Slicing a tensor, as in `t[1:, ::2]`, returns a view.

### NumPy arrays

Enceladus binds a NumPy array's memory directly, without copying it. Because NumPy has
no notion of a stream, a launch with a NumPy argument waits for the GPU, so the array
holds the result when the launch returns. That wait also completes every earlier
launch on the stream, and it costs about 70-100 µs.

To batch launches over NumPy arrays, turn off the wait with
`enceladus.async_numpy(True)`. You must then call `enceladus.synchronize()` before you
read the arrays:

```python
import numpy as np

import enceladus
import enceladus.language as tl


@enceladus.jit
def inc_kernel(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(x_ptr + offs, tl.load(x_ptr + offs, mask=mask) + 1, mask=mask)


x = np.zeros(10_000, np.float32)
enceladus.async_numpy(True)
for _ in range(100):
    inc_kernel[(enceladus.cdiv(x.size, 1024),)](x, x.size, BLOCK=1024)
enceladus.synchronize()  # required before reading x
enceladus.async_numpy(False)
assert (x == 100).all()
```

### PyTorch and MLX

When every array argument is a PyTorch tensor on the `mps` device, the kernel runs on
PyTorch's own MPS stream, so it's ordered with the PyTorch operations before and after
it, and you synchronize the way you do for any PyTorch code. MLX launches are
synchronous. For details, see [Framework interop](interop.md).

## What's next

- To debug a kernel, see [Debugging](debugging.md).
- To look up a `tl` function, see [Language reference](language-reference.md).
