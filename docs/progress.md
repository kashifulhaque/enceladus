# Progress log

This log records each milestone's results, benchmark numbers, known gaps, and deviations
from [PLAN.md](../PLAN.md). All numbers come from the development machine (M4 Pro,
16-core GPU, macOS 27, Xcode 27).

## M0: Runtime and raw kernels

### What was built

- `src/tegula/_C/`: the Objective-C++ runtime (`tegula_rt.h`, `tegula_rt.mm`) and the
  nanobind module (`bindings.mm`), built by scikit-build-core and CMake.
  - Compile options: language version (3.2 by default, always set explicitly), math mode,
    FP32 function precision, invariance, and logging. Diagnostics return in a 64 KB
    buffer.
  - Pipelines are created with binding reflection, so `Pipeline.bindings()` reports each
    buffer index, name, data type, and size.
  - `Stream`: one open command buffer with a serial encoder. It commits every
    `flush_every` dispatches, signals an `MTLSharedEvent`, and on sync spin-waits on the
    event with back-off. Command-buffer errors are collected at sync and raised as
    `tegula.MetalError` with the names of the kernels in the failing batch.
  - `LaunchPlan`: a precomputed binding plan (buffer indices plus a scalar table). Each
    scalar gets its own `setBytes` call at its own index.
  - Timing: `Stream.timed_run` and `Stream.flush_timed` return GPU start and end
    timestamps.
- `src/tegula/runtime/`: `device.py` (singleton and `Capabilities`), `tensor.py`
  (`tegula.Tensor` and allocation helpers), `interop.py` (`as_kernel_arg` for tensors and
  NumPy arrays), `stream.py`, and `raw.py` (`tegula.metal_kernel`).
- `src/tegula/testing.py`: `do_bench` with GPU timestamps and warm-up, plus
  `assert_close`. The plan schedules `do_bench` for M5; it arrived early because the M2
  benchmarks need it.

### Benchmarks

The following numbers come from `benchmarks/bench_dispatch.py` (10,000 launches of a
1,024-element vector add, 5 runs):

| Measurement | Min | Median | Target |
|---|---|---|---|
| Raw `metal_kernel` launch, sustained | 1.05 µs | 1.08 µs | 2 µs or less |
| Native `Stream.dispatch`, sustained | 1.04 µs | 1.06 µs | Not applicable |
| Synchronous round trip | 72 µs | 88 µs | About 100 µs |

Sustained launches are bound by the GPU's cost for a dependent dispatch (about 1.1 µs),
not by Python. A raw one-element-per-thread vector add at 256 MB per array measured
230 GB/s through `do_bench`.

### Tests

`tests/test_runtime.py` has 5 test functions (16 cases after parametrization, 0.2 s).
They cover vector add over tensors, tensor views with offsets, and aligned and unaligned
NumPy arrays at 1, 1,000, and 2^20 + 3 elements; zero-copy aliasing in both directions;
scalar packing by reflected type (`half`, `int`, `bfloat`, `ulong`, `uchar`); error
propagation; and flush-on-threshold with cross-batch dependencies.

### Deviations from the plan

- **NumPy wrapping covers the whole page range.** Instead of wrapping only page-aligned
  arrays, `interop.py` wraps the 16 KB pages that span any NumPy array and binds it with
  a byte offset. This follows the documented `newBufferWithBytesNoCopy` contract (the
  pointer and length are page multiples) and makes every NumPy argument zero-copy. The
  copy-and-write-back path remains as a fallback for when Metal returns `nil`.
- **The wrapped-buffer cache doesn't hold the array.** Cache entries are keyed on the
  array's base object and dropped by a weak-reference callback when that object is
  freed. This prevents a later allocation at the same address from reusing a stale
  wrapper. Asynchronous NumPy launches keep their arrays alive in the stream until the
  next sync.
- **`fr_stream_dispatch` takes a `fr_launch_plan` and a kernel name.** The plan's
  signature lists buffers and a scalar table but no buffer indices, which the ABI needs
  because pointer and scalar arguments interleave. The name feeds error messages.
- **The native stream owns the flush threshold.** `Stream.dispatch` commits when the
  pending count reaches `flush_every`, which keeps the per-launch Python work to one call.
- **Raw kernels bind scalars by reflected type.** `metal_kernel` packs Python numbers
  with the type the kernel declares, and it checks the argument count against the
  reflected bindings. A missing argument raises `TypeError` before anything reaches the
  GPU.

### Known gaps

- **Command-buffer error reporting is untested on this machine.** The GPU tolerated every
  fault tried: reads and writes to unmapped addresses, and a buffer whose purgeable
  state was set to empty. A kernel that spins forever wasn't stopped by a watchdog
  within 60 seconds. The error path exists, but no test exercises it.
- **`Tensor.__dlpack__` and `__dlpack_device__` are deferred to M6**, where they export
  `kDLMetal` as the plan specifies. `Tensor.__array__` covers NumPy interop until then.
- `launcher.py` and `cache.py` start in M2 with the compiled-kernel path.

## M1: Frontend, IR, and interpreter

### What was built

- `language/`: dtypes, `constexpr`, and one builtin registry. Each `tl` function
  registers a frontend handler and a NumPy interpreter handler together. The P0 surface
  is complete, plus `tl.reduce` (including tuple inputs), `tl.dot`,
  `make_tensor_descriptor` with `desc.load` and `desc.store`, `tl.range`, and `tl.cast`.
- `compiler/ir.py`: types, values, ops, regions, a builder, an MLIR-style printer, and a
  verifier with a rule for every op in the plan's table.
- `compiler/frontend.py` and `semantic.py`: the AST-to-IR generator and Triton's
  promotion and broadcasting rules. The frontend and the interpreter share these rules,
  so their result dtypes agree. The M2 entry point is `build_ir(fn, arg_types, arg_facts,
  constexprs, num_warps, math_mode)`.
- `interpreter/interp.py`: `ITile`, `IPointer`, and `IDesc` over NumPy, with a sequential
  grid. An out-of-bounds unmasked access raises `IndexError` with the kernel line.
- `runtime/jit.py`: `@tegula.jit`, specialization facts as function-argument attributes,
  and a SHA-256 dependency hash.
- Examples 01-06.

### Results

- The default suite runs 189 tests and skips 130 in 0.4 s. Every skip is a compiled-mode
  case that M2 enables.
- The frontend plus verifier takes about 0.14 ms for vector add and 0.6 ms for matmul.
- `TEGULA_INTERPRET=1 TEGULA_DUMP=1` prints readable IR for every example.

### Deviations from the plan

- `build_ir` always verifies. `TEGULA_VERIFY=1` makes interpreted launches also build
  and verify IR once per specialization, which is how the tests check every example's IR.
- Triton hints with no effect on Apple GPUs (`num_stages`, `cache_modifier`,
  `eviction_policy`, and `input_precision`) are accepted and ignored.
- A reduction to rank 0 produces a scalar, not a 0-d tile.

### Known gaps

- `multiple_of` and `max_contiguous` (M2), atomics and scans (M7), `device_print` and
  `device_assert` (M9), `join`, and `split` aren't implemented.
- The dependency hash tracks globals referenced by bare names only, not `module.attr`
  chains, and has no test.
- In the interpreter, the loop variable is a Python `int`, so `//`, `%`, and overflow on
  it follow Python rules rather than `i32` rules.
