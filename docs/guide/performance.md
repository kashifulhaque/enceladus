# Performance tips

This page describes how to make Enceladus kernels fast: how to measure them, how to feed
`tl.dot` efficiently, how to choose block sizes with the autotuner, how to keep launch
overhead low, and how to choose a `tl.dot` backend.

## Measure with GPU timestamps

To time a kernel, use `enceladus.testing.do_bench`. It warms up the GPU, runs your
function repeatedly, and times each run with the command buffer's GPU timestamps, so
Python overhead doesn't distort the result. The function must launch kernels on
`enceladus.Tensor` arguments, because a launch that synchronizes can't be timed:

```python
import enceladus
import enceladus.language as tl
from enceladus.testing import do_bench


@enceladus.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


n = 1 << 24
x, y = enceladus.randn(n), enceladus.randn(n)
out = enceladus.empty_like(x)
grid = (enceladus.cdiv(n, 1024),)
ms = do_bench(lambda: add_kernel[grid](x, y, out, n, BLOCK=1024))
print(f"{ms:.3f} ms, {3 * 4 * n / ms / 1e6:.0f} GB/s")
```

`do_bench` returns the median in milliseconds. Pass `return_mode="min"` or `"all"` for
other summaries. The GPU runs slowly for about the first 50 ms of work after it idles,
and other GPU work on your Mac, including the display, adds noise. Compare kernels in
the same process, and repeat a measurement before you trust a small difference.

## Choose block sizes and `num_warps`

Most memory-bound kernels reach full bandwidth with blocks of 1,024 to 4,096 elements
and 4 to 8 SIMD groups. On an M4 Pro, the vector add with `BLOCK=1024` and the default
`num_warps=4` reaches about 235 GB/s, and a row softmax with 8 SIMD groups about
240 GB/s, against a measured copy bandwidth of 238 GB/s.

Keep each tile within about 128 registers per thread. Registers per thread are roughly
the tile's element count, times its element size in 32-bit words, divided by
`32 * num_warps`. For a tile of more than 128 registers, Enceladus prints a warning,
because the kernel might spill registers to memory and slow down by several times. For
a tile of more than 256, compilation fails. `kernel.explain` lists the register count of every tile.

## Feed `tl.dot` from tensor descriptors

`tl.dot` can get its operands in two ways:

- When an operand is a tensor descriptor load, such as `tl.dot(a_desc.load([m, k]),
  ...)`, and nothing else uses the loaded tile, `tl.dot` loads its matrix fragments
  straight from device memory. This path needs no threadgroup memory and no barriers.
  The GPU's caches supply the reuse between SIMD groups.
- Any other operand, such as a pointer-tile load, is staged through threadgroup memory
  first.

The direct path is the fastest way to write a matmul on Apple GPUs. In the research
behind Enceladus, direct loads ran 7-15% faster than staging through threadgroup
memory. To check which path a kernel takes, run `kernel.explain`: a direct operand shows
as "read by tl.dot straight from device memory; 0 registers."

Epilogues fuse for free. Operations on the accumulator after the loop, such as adding a
bias, applying an activation, or casting, work on the accumulator's registers in place.
A matmul with a fused bias and GELU runs within about 2-3% of a plain matmul.

## Autotune kernels

The fastest block sizes depend on the problem shape, the dtype, and the GPU. Some
configurations also spill registers and run 7-15 times slower, in ways that no simple
model predicts. The autotuner measures instead of guessing.

`@enceladus.autotune(configs, key)` compiles every candidate `enceladus.Config` in
parallel, times each one with `do_bench`, and uses the fastest for each combination
of the `key` arguments and the argument dtypes. It rejects configurations that run more
than 3 times slower than the median, and it saves each result on disk, in
`~/.cache/enceladus/autotune/`, keyed by the kernel's source, the compiler version, and
the GPU. Later processes reuse saved results without benchmarking.

Enceladus ships pre-validated configuration lists for the two kernels that need them
most:

- `enceladus.configs.matmul_configs(dtype)` returns the matmul configurations for
  `float32`, `float16`, or `bfloat16`, excluding the shapes that spill. They set `BM`,
  `BN`, and `BK`, `num_warps`, and `dot_warps`, and three of them use
  `dot_backend="mpp"`.
- `enceladus.configs.attention_configs(dtype, head_dim)` returns the flash attention
  configurations, which set `BLOCK_M` and `BLOCK_N`, for `examples/08_flash_attention.py`.

The following program autotunes a descriptor matmul. With `ENCELADUS_PRINT_AUTOTUNING=1`,
it prints each result as a `Config` that you can paste into your code:

<!-- snippet: env ENCELADUS_PRINT_AUTOTUNING=1 -->
```python
import numpy as np

import enceladus
import enceladus.language as tl
from enceladus.configs import matmul_configs


@enceladus.autotune(configs=matmul_configs("float16"), key=["M", "N", "K"])
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


M, N, K = 1024, 1024, 512
rng = np.random.default_rng(0)
a = rng.standard_normal((M, K)).astype(np.float16)
b = rng.standard_normal((K, N)).astype(np.float16)
c = np.empty((M, N), np.float16)
grid = lambda meta: (enceladus.cdiv(N, meta["BN"]), enceladus.cdiv(M, meta["BM"]))
matmul_kernel[grid](a, b, c, M, N, K, K, N, N)
print("chose", matmul_kernel.config_for(a, b, c, M, N, K, K, N, N))
```

The grid must be a function of `meta`, because the block sizes come from the chosen
configuration. The output is similar to the following:

```text
enceladus: autotuning matmul_kernel [1024|1024|512|float16,float16,float16] chose enceladus.Config({'BM': 64, 'BN': 32, 'BK': 32}, num_warps=4, dot_warps=(4, 1), dot_backend='mpp') (0.186 ms)
```

The autotuner has the following options, which match Triton's:

- `prune_configs_by={"early_config_prune": fn}` filters the configurations before
  benchmarking. `fn(configs, named_args)` returns the configurations to keep, for
  example the ones for the dtype of the inputs, as `examples/04_matmul.py` does.
- `prune_configs_by={"perf_model": fn, "top_k": n}` benchmarks only the `n`
  configurations that a performance model ranks best.
- `reset_to_zero=["out_ptr"]` zeroes an argument before each benchmark run, for kernels
  that accumulate into their output. `restore_value` restores arguments after tuning.
- `enceladus.heuristics({"BLOCK": fn})` computes constexprs from the other arguments
  instead of benchmarking them.

Set `ENCELADUS_ALWAYS_COMPILE=1` to ignore saved results and tune again.

## Keep launches asynchronous

Each launch that waits for the GPU costs about 70-100 µs, most of it in the GPU driver.
An asynchronous launch costs about 3.3 µs of host time for a `@enceladus.jit` kernel,
and the stream batches up to 64 launches into one command buffer, which brings the
GPU-side cost of a small launch to about 1 µs. For code that runs many small kernels,
asynchronous launches are 20-30 times faster.

To keep launches asynchronous, follow these guidelines:

- Allocate your data as `enceladus.Tensor` objects, with `enceladus.empty`,
  `enceladus.zeros`, `enceladus.randn`, and the other allocation functions, or wrap
  existing NumPy arrays with `enceladus.from_numpy`. A launch with only tensor arguments
  never waits.
- Read results on the CPU once, at the end. `Tensor.numpy()`, `print(tensor)`, and
  `np.asarray(tensor)` each wait for all pending work.
- If you must pass NumPy arrays, call `enceladus.async_numpy(True)` and call
  `enceladus.synchronize()` before you read them.
- With PyTorch, pass only MPS tensors to a kernel. A launch that mixes PyTorch tensors
  with other arrays takes the synchronized path.
- MLX launches always wait. Batch work into fewer, larger kernels where you can.

`enceladus.from_numpy` shares the array's memory only when the array is C-contiguous
and starts on a 16 KB page boundary. Otherwise, it copies the array, and later changes
to the NumPy array don't reach the tensor.

The stream submits a command buffer every 64 launches. To change the batch size, set
`ENCELADUS_FLUSH_EVERY`. Larger batches rarely help, because the GPU becomes the
bottleneck at about 1 µs per dependent launch.

## Reduce compile time

The first launch of each specialization compiles the kernel, which takes several
milliseconds for a simple kernel and about 200 ms for a kernel that uses the `mpp`
`tl.dot` backend. Enceladus caches the generated MSL on disk, and Metal caches the
compiled binary, so later processes load a kernel in well under 10 ms. To compile
before the first launch, for example at startup, call `kernel.warmup(*args, **constexprs)`
with the same arguments as the launch.

Avoid specializations you don't need. If a size argument changes on every launch, add
it to `do_not_specialize` in `@enceladus.jit`.

## Use fast math

By default, kernels compile with Metal's relaxed math mode, which keeps infinities and
NaNs. `@enceladus.jit(math_mode="fast")` lets the Metal compiler assume that no value is
infinite or NaN, which can speed up math-heavy kernels. Don't use it for kernels that
rely on `-inf`, such as a softmax that masks with `-float("inf")`.

In both modes, `tl.sin` and `tl.cos` use Metal's precise variants, because the fast ones
return 0 for inputs above about 1e7. In compute-bound code, they take up to 3.3 times as
long as the fast ones. `tl.tanh` takes about 1.7 times as long as Metal's fast `tanh`,
which is inaccurate near 0 and returns NaN for inputs of 45 and more.

For exponentials, prefer `tl.exp2` and fold the factor `log2(e)` into a scale that you
compute once, as `examples/08_flash_attention.py` does.

## Choose a `tl.dot` backend

Enceladus can compile `tl.dot` in two ways. The `dot_backend` option selects one, as a
launch option, as an argument of `warmup` and `explain`, or as a field of
`enceladus.Config`:

- `"simdgroup"` compiles every `tl.dot` to `simdgroup_matrix` instructions, which every
  Apple GPU from M1 on supports. It works with any tile and any epilogue.
- `"mpp"` compiles eligible `tl.dot` loops to Metal 4's `matmul2d` operation from Metal
  Performance Primitives (MPP). It needs a GPU that supports the Metal 4 family and
  macOS 26 or later.
- `"auto"`, the default, picks `"mpp"` on Apple10 (M5) and later GPUs, where `matmul2d`
  runs on the GPU's neural accelerators, and `"simdgroup"` on earlier GPUs.

The following launch requests the `mpp` backend:

<!-- snippet: skip -->
```python
matmul_kernel[grid](a, b, c, M, N, K, K, N, N, BM=64, BN=64, BK=32, dot_backend="mpp")
```

A `tl.dot` loop is eligible for `mpp` when it meets the following conditions:

- The accumulator starts from `tl.zeros` and the loop updates it only with
  `acc = tl.dot(a, b, acc)`.
- Both operands are tensor descriptor loads, optionally through `tl.trans`, from
  descriptors created before the loop, at offsets built from program IDs, loop
  counters, and non-negative constants.
- The operands are `float16` accumulating in `float32` or `float16`, `bfloat16`
  accumulating in `float32` or `bfloat16`, or `float32` accumulating in `float32`.
  Integer `tl.dot` always uses `"simdgroup"`: on an M4 Pro, `int8` `matmul2d` sums in
  floating point and returned wrong `int32` results once a sum passed 2^24.
- The output tile is 16 to 128 in each dimension, and the kernel uses at most 8 SIMD
  groups.
- After the loop, the result reaches a single descriptor store through elementwise
  operations only. A bias add and an activation qualify; a reduction such as `tl.sum`
  doesn't.

A kernel that isn't eligible, or a GPU that doesn't support `matmul2d`, falls back to
`"simdgroup"`, so a kernel that compiles with `"simdgroup"` also compiles when you
request `"mpp"`. The `enceladus` logger records each fallback and its reason at debug
level, and `kernel.explain` lists the reasons. For an example, see
[Explain the compiler's decisions](debugging.md#explain-the-compilers-decisions).

On an M4 Pro, which runs `matmul2d` on its regular shader cores, the `mpp` backend
measured the following in `benchmarks/bench_matmul.py`:

- `float16` at 4096³: 3-6% faster than `simdgroup`, about 5.8-6.0 TFLOPS.
- `float16` and `float32` at 2000³: 15-25% faster than `simdgroup`.
- Small and ragged shapes, such as 513³: about 2 times faster than `simdgroup`.
- `float32` at 4096³: its best runs match `simdgroup`, but its median runs about 20%
  slower.

Because the winner depends on the shape and dtype, let the autotuner choose:
`matmul_configs` includes `mpp` configurations, and on a GPU without `matmul2d` they
compile with `simdgroup`. A kernel that uses `mpp` takes about 200 ms to compile the
first time; the disk caches make later processes fast.
