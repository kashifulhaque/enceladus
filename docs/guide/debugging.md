# Debugging

This page describes the tools that help you find bugs in Enceladus kernels: compilation
errors, the CPU interpreter, IR and MSL dumps, `kernel.explain`, printing and asserts on
the GPU, and GPU capture.

## Read compilation errors

When the compiler can't compile a kernel, it raises `enceladus.CompilationError` with
the file, line, and column of the problem, a description, and a suggested fix. For
example, a kernel whose block size parameter isn't annotated as `tl.constexpr` fails as
follows:

```text
enceladus.CompilationError: add.py:9:12: the end of tl.arange must be a compile-time
integer, but got a runtime value `BLOCK` of type tl.int32. If `BLOCK` comes from a kernel
parameter, annotate the parameter as tl.constexpr, for example `BLOCK: tl.constexpr`.
    offs = tl.arange(0, BLOCK)
           ^
```

Enceladus refuses every construct that it doesn't support, such as a `while` loop or an
unsupported dtype, with an error of this form. It doesn't compile a kernel that it can't
compile correctly.

## Run kernels in the interpreter

The interpreter runs a kernel on the CPU with NumPy, one program at a time. Each `tl`
operation is one NumPy operation on a whole tile, so you can use `print()`, `pdb`, and
your debugger's breakpoints inside the kernel and see real tile values.

To run every kernel in the interpreter, set `ENCELADUS_INTERPRET=1`:

```bash
ENCELADUS_INTERPRET=1 uv run python my_script.py
```

To interpret a single kernel, pass `interpret=True` to the decorator:

<!-- snippet: skip -->
```python
@enceladus.jit(interpret=True)
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    ...
```

The interpreter behaves as follows:

- It accepts the same arguments as the compiled kernel: NumPy arrays,
  `enceladus.Tensor` objects, PyTorch MPS tensors, and MLX arrays. It works on NumPy
  views of their memory, so its writes are visible to the caller.
- An unmasked load or store out of bounds raises `IndexError` with the kernel's line.
  The GPU doesn't check bounds, so run a suspect kernel in the interpreter first.
- It's slow. Use small grids when you debug.
- It also runs the compiler's checks on each kernel, once per specialization. Python
  accepts some code that compiled mode refuses, such as a list of tiles or a `tl.dot`
  with a K block of 4. For such a kernel, the interpreter warns once, names the line
  that compiled mode refuses, and runs the kernel anyway. To raise the error instead,
  set `ENCELADUS_VERIFY=1`. To skip the check, set `ENCELADUS_VERIFY=0`. The check
  skips kernels whose source can't be read, such as kernels defined in an interactive
  prompt.

The interpreter computes the same results as the compiled kernel, with a few
exceptions: the GPU flushes `float32` denormals to zero and the interpreter doesn't,
math functions such as `exp` differ in the last bits, and colliding atomics can apply
in a different order. The test suite uses the interpreter as the reference for every
compiled kernel.

## Inspect the generated IR and MSL

To see what the compiler generates, set `ENCELADUS_DUMP=1`. For a compiled launch,
Enceladus writes the IR and the MSL to the kernel's disk-cache entry and prints its
path to `stderr`:

```text
enceladus: add_kernel: IR and MSL in /Users/you/.cache/enceladus/70ddc3341d173d35df885ec2bd54181e
```

The directory contains `ir.txt` (the Enceladus IR), `kernel.metal` (the generated MSL),
and `meta.json` (the argument layout and compile options). With `ENCELADUS_INTERPRET=1`,
`ENCELADUS_DUMP=1` prints the IR to `stdout` instead.

The following environment variables control the compiler's caches:

| Variable | Effect |
|---|---|
| `ENCELADUS_CACHE_DIR` | Moves the disk cache from `~/.cache/enceladus` to another directory. |
| `ENCELADUS_ALWAYS_COMPILE=1` | Ignores the disk cache and saved autotuning results, and compiles every kernel again. |
| `ENCELADUS_OVERRIDE_DIR` | Loads `DIR/KERNEL_NAME.metal` instead of the generated MSL, where `KERNEL_NAME` is the kernel function's name. |

`ENCELADUS_OVERRIDE_DIR` lets you edit the generated MSL by hand to test a change. Copy
`kernel.metal` from a dump, rename it after the kernel, and keep its entry point and
`[[buffer(N)]]` arguments unchanged.

To get the MSL of one specialization from Python, compile it without launching:

<!-- snippet: skip -->
```python
compiled = add_kernel.warmup(x, y, out, n, BLOCK=1024)
print(compiled.msl)
print(compiled.threadgroup_memory_bytes)
```

The returned object also has `ir` and `num_warps` attributes.

## Explain the compiler's decisions

`kernel.explain(*args, grid=..., **constexprs)` prints and returns a report of what the
compiler decided for one specialization, without launching the kernel. It takes the same
arguments and launch options as a launch. The report has the following parts, each with
source lines:

- The MSL language version, and whether the kernel uses logging or asserts.
- The `tl.dot` backend, each `tl.dot` with its shape and SIMD-group grid, and, for each
  dot that couldn't use the backend you asked for, the reason.
- Every tile with its layout (registers, lanes, and SIMD groups per dimension) and its
  register count per thread. Operands that `tl.dot` reads straight from device memory
  show 0 registers.
- Every layout conversion, with whether it moves registers or exchanges data through
  threadgroup memory.
- Peak threadgroup memory, and each request by operation.

The following output comes from a matmul kernel that asks for `dot_backend="mpp"` but
reduces its result with `tl.sum`, which the `mpp` backend doesn't support:

```text
Kernel `rowsum_matmul`: num_warps=4 (128 threads per program), grid (4, 1, 1)
MSL language version 3.2
dot backend: simdgroup
  dbg.py:30  64x64x32 (MxNxK), SIMD-group grid 4x1  `acc = tl.dot(...)`
  uses simdgroup instead of mpp: dbg.py:30:15: the tl.dot result feeds `reduce`,
  which isn't an elementwise op in the loop's block

Tiles (layout; registers per thread):
  dbg.py:29  loop-carried `acc` f32[64, 64]: simdgroup_matrix fragments: regs 2x16,
  lanes 8x4, warps 4x1; 32 registers  `for k in range(0, K, BK):`
  dbg.py:30  desc_load f16[64, 32]: read by tl.dot straight from device memory;
  0 registers  `acc = tl.dot(...)`
  ...

Layout conversions:
  (none)

Threadgroup memory: 0 bytes of 32768 (one arena that each operation reuses)
```

Look for layout conversions through threadgroup memory inside loops, tiles with many
registers, and `tl.dot` fallbacks. Each costs performance. The `enceladus` logger also
records each `tl.dot` fallback at debug level when the kernel compiles:

<!-- snippet: skip -->
```python
import logging

logging.getLogger("enceladus").setLevel(logging.DEBUG)
logging.basicConfig()
```

## Print values from the GPU

`tl.device_print(prefix, *args)` prints runtime values from a compiled kernel to
`sys.stderr`. It prints one line for each element of its tile arguments, and each line
starts with the program ID and the element's index:

```python
import numpy as np

import enceladus
import enceladus.language as tl


@enceladus.jit
def print_kernel(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n)
    tl.device_print("x", x)


print_kernel[(2,)](np.arange(4, dtype=np.float32), 4, BLOCK=2)
```

The output is similar to the following:

```text
pid (0, 0, 0) idx (0) x: 0.000000
pid (0, 0, 0) idx (1) x: 1.000000
pid (1, 0, 0) idx (0) x: 2.000000
pid (1, 0, 0) idx (1) x: 3.000000
```

Printing has the following behavior:

- Lines arrive when the stream synchronizes, not while the kernel runs, and in no
  particular order across programs. Every line that a kernel prints appears before the
  synchronization that waits for the kernel returns.
- The interpreter prints the same lines with the same format.
- Metal handles about 170,000 lines per second, and it silently drops lines when one
  command buffer prints more than about 95,000. Print a few elements, for example under
  `if tl.program_id(0) == 0:`, or synchronize more often.
- A kernel that prints runs on a special command queue for the rest of the process.
  On PyTorch tensors, it takes the slower synchronized path.
- Printing and GPU capture don't work in the same process. For details, see
  [Capture a GPU trace](#capture-a-gpu-trace).

To print compile-time values, such as a tile's shape or a constexpr, while the kernel
compiles, use `tl.static_print`.

## Check conditions on the GPU

`tl.device_assert(cond, msg, mask=None)` checks a condition in a compiled kernel.
Metal's own validation layer doesn't catch out-of-bounds accesses, so asserts are the
main way to find them on the GPU.

Asserts compile only when you set `ENCELADUS_DEBUG=1`. Without it, the compiler removes
them and the kernel runs at full speed. The flag is part of the kernel's cache key, so
changing it recompiles. The following kernel checks its gather indices:

<!-- snippet: env ENCELADUS_DEBUG=1 -->
```python
import numpy as np

import enceladus
import enceladus.language as tl


@enceladus.jit
def gather_kernel(x_ptr, idx_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    idx = tl.load(idx_ptr + offs, mask=mask)
    tl.device_assert((idx >= 0) & (idx < n), "index out of range", mask=mask)
    ok = mask & (idx >= 0) & (idx < n)
    tl.store(out_ptr + offs, tl.load(x_ptr + idx, mask=ok), mask=mask)


x = np.arange(4, dtype=np.float32)
idx = np.array([0, 1, 2, 9], np.int32)
out = np.empty(4, np.float32)
try:
    gather_kernel[(2,)](x, idx, out, 4, BLOCK=2)
except enceladus.DeviceAssertionError as e:
    print(e)
else:
    raise SystemExit("no assertion failed; run with ENCELADUS_DEBUG=1")
```

With `ENCELADUS_DEBUG=1`, the output is similar to the following:

```text
gather.py:12:5: device assertion failed in program (1, 0, 0): index out of range
    tl.device_assert((idx >= 0) & (idx < n), "index out of range", mask=mask)
    ^
```

An assert has the following behavior:

- The error names the first failing program and the assert's source line. It's raised
  at the next synchronization, which for NumPy arguments is the end of the launch.
- The kernel keeps running after an assert fails, because Metal can't stop a running
  kernel. Guard the dangerous access with a mask too, as the example does with `ok`.
- Where `mask` is false, the condition isn't checked.
- The interpreter follows `ENCELADUS_DEBUG` too, and raises at the failing assert.

To check compile-time conditions, use `tl.static_assert`.

## Capture a GPU trace

`enceladus.capture(path)` records every GPU command inside a `with` block into a
`.gputrace` document. You can open the document in Xcode to inspect each dispatch, its
buffers, and its shader.

Metal allows capture only when the process starts with `MTL_CAPTURE_ENABLED=1` in its
environment. Without it, `enceladus.capture` raises an error that explains the
requirement. To capture a launch, run your script as follows:

```bash
MTL_CAPTURE_ENABLED=1 uv run python capture_add.py
```

The script wraps the launches that you want to record:

<!-- snippet: env MTL_CAPTURE_ENABLED=1 -->
```python
import os
import tempfile

import enceladus
import enceladus.language as tl


@enceladus.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


x, y = enceladus.randn(4096), enceladus.randn(4096)
out = enceladus.empty_like(x)
path = os.path.join(tempfile.mkdtemp(), "add.gputrace")
with enceladus.capture(path):
    add_kernel[(4,)](x, y, out, 4096, BLOCK=1024)
print("wrote", path)
```

The path must end in `.gputrace` and must not exist yet. The capture synchronizes the
stream when it starts and when it ends.

Printing and capture are incompatible. Metal refuses to capture once a process has set
up shader logging, which Enceladus does when a kernel that calls `tl.device_print` first
launches. So `enceladus.capture` raises an error if a printing kernel has run in the
process, and a printing kernel that launches inside a capture raises too. Capture in a
process that doesn't print.

## Summary of environment variables

The following table lists every environment variable that Enceladus reads:

| Variable | Effect |
|---|---|
| `ENCELADUS_INTERPRET=1` | Runs kernels in the NumPy interpreter. |
| `ENCELADUS_DEBUG=1` | Compiles `tl.device_assert` checks. |
| `ENCELADUS_DUMP=1` | Writes the IR and MSL of each compiled kernel and prints where. |
| `ENCELADUS_CACHE_DIR` | Sets the disk-cache directory (default `~/.cache/enceladus`). |
| `ENCELADUS_ALWAYS_COMPILE=1` | Ignores the disk cache and saved autotuning results. |
| `ENCELADUS_OVERRIDE_DIR` | Loads hand-edited MSL from `DIR/KERNEL_NAME.metal`. |
| `ENCELADUS_VERIFY` | Controls the compiler's checks on interpreted launches: unset warns about kernels that compiled mode refuses, `1` raises the error, and `0` skips the check. `1` also verifies the IR after every compiler pass. |
| `ENCELADUS_PRINT_AUTOTUNING=1` | Prints each autotuning result. |
| `ENCELADUS_FLUSH_EVERY` | Sets how many launches the stream batches per command buffer (default 64). |
| `MTL_CAPTURE_ENABLED=1` | Lets `enceladus.capture` record GPU traces. Metal reads it at process start. |
