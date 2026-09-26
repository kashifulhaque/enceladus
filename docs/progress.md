# Progress log

This log records each milestone's results, benchmark numbers, known gaps, and deviations
from [PLAN.md](../PLAN.md). All numbers come from the development machine (M4 Pro,
16-core GPU, macOS 27, Xcode 27).

## M0: Runtime and raw kernels

### What was built

- `src/enceladus/_C/`: the Objective-C++ runtime (`enceladus_rt.h`, `enceladus_rt.mm`) and the
  nanobind module (`bindings.mm`), built by scikit-build-core and CMake.
  - Compile options: language version (3.2 by default, always set explicitly), math mode,
    FP32 function precision, invariance, and logging. Diagnostics return in a 64 KB
    buffer.
  - Pipelines are created with binding reflection, so `Pipeline.bindings()` reports each
    buffer index, name, data type, and size.
  - `Stream`: one open command buffer with a serial encoder. It commits every
    `flush_every` dispatches, signals an `MTLSharedEvent`, and on sync spin-waits on the
    event with back-off. Command-buffer errors are collected at sync and raised as
    `enceladus.MetalError` with the names of the kernels in the failing batch.
  - `LaunchPlan`: a precomputed binding plan (buffer indices plus a scalar table). Each
    scalar gets its own `setBytes` call at its own index.
  - Timing: `Stream.timed_run` and `Stream.flush_timed` return GPU start and end
    timestamps.
- `src/enceladus/runtime/`: `device.py` (singleton and `Capabilities`), `tensor.py`
  (`enceladus.Tensor` and allocation helpers), `interop.py` (`as_kernel_arg` for tensors and
  NumPy arrays), `stream.py`, and `raw.py` (`enceladus.metal_kernel`).
- `src/enceladus/testing.py`: `do_bench` with GPU timestamps and warm-up, plus
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
- `runtime/jit.py`: `@enceladus.jit`, specialization facts as function-argument attributes,
  and a SHA-256 dependency hash.
- Examples 01-06.

### Results

- The default suite runs 189 tests and skips 130 in 0.4 s. Every skip is a compiled-mode
  case that M2 enables.
- The frontend plus verifier takes about 0.14 ms for vector add and 0.6 ms for matmul.
- `ENCELADUS_INTERPRET=1 ENCELADUS_DUMP=1` prints readable IR for every example.

### Deviations from the plan

- `build_ir` always verifies. `ENCELADUS_VERIFY=1` makes interpreted launches also build
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

## M2: Layouts, codegen, and elementwise kernels

### What was built

- `compiler/layout.py`: `BitLayout` with `blocked`, `slice_layout`, `expand`, `broadcast`,
  `permute`, `reshape`, `simd_acc`, and `reg_map`. `reg_map` finds, for each destination
  register, the source register in the same thread that holds the same element, or
  reports that the conversion must cross threads. It handles broadcasting, so one
  function covers register remaps, broadcasts, and the "differ only in register order"
  case.
- `compiler/passes/`: `simplify` (CSE and DCE), `axis_info` (contiguity, divisibility in
  bytes for pointers, and constancy), and `layouts` (layout assignment).
- `compiler/codegen/`: the MSL emitter, prelude (`erf`, `sigmoid`, and shuffles for
  `bfloat`, `bool`, and 64-bit types), `msl.py`, and `reduce.py`.
- `runtime/compile.py`, `launcher.py`, and the compiled path in `jit.py`: a generated
  argument binder, a per-kernel specialization cache, the disk cache,
  `ENCELADUS_ALWAYS_COMPILE`, `ENCELADUS_DUMP`, `ENCELADUS_OVERRIDE_DIR`, `kernel.warmup()`, and
  the recompilation warning.

### Benchmarks

| Measurement | Result | Target |
|---|---|---|
| Vector add, 256 MB per array, FP32 | 235 GB/s (MLX 225, wall clock) | 220 GB/s or more |
| Vector add, 256 MB per array, FP16 | 239 GB/s (MLX 219, wall clock) | 220 GB/s or more |
| Enceladus compile time, vector add (frontend to MSL) | 0.32 ms | Under 5 ms |
| `@enceladus.jit` launch, sustained | 3.8 µs | 5 µs or less |

### Tests

Every M1 differential test now also runs compiled: 323 of 329 cases pass. The 6
failures are the pointer-tile matmul, which needs `tl.dot` (M4). `tests/test_codegen.py`
adds a layout conversion through threadgroup memory (checked against NumPy), the
threadgroup-memory overflow error with its source line, reduce-then-broadcast with no
conversion, deterministic MSL, and AxisInfo contiguity and order. `test_softmax` gained a
case where whole lanes see only `other=-inf`.

### Deviations from the plan

- **Rematerialization happens in codegen instead of IR rewriting.** Layout assignment
  classifies tile values as cheap, view, or anchored and fixes layouts only for anchored
  values. Codegen emits cheap values lazily in whatever layout each use needs, memoized
  per scope, and derives views from their input's registers. No `convert_layout` ops are
  inserted into the IR; a use that needs a different layout gets a register remap or a
  threadgroup exchange at that point.
- **`lower_convert_layout`, `alloc_threadgroup_memory`, and `insert_barriers` live in
  codegen.** All exchanges and reduction scratch share one threadgroup arena at offset
  0, sized to the largest use. Every exchange writes behind a barrier and reads behind a
  second one, which is the conservative scheme the plan allows. Overflow raises a
  source-located `CompilationError`.
- **No `strength_reduce` pass.** Metal's compiler strength-reduces division by constants
  and hoists loop-invariant math. Revisit if profiles show integer division in hot loops.
- **Scalar loads only.** The plan says to add vector loads after the tests pass. Vector
  add already reaches 235 GB/s with scalar loads, so vectorization is deferred.
- **Half-precision arithmetic computes in FP32 and rounds per op**, matching the
  interpreter exactly (FP32 has enough precision that this equals correctly rounded
  native FP16 and BF16 arithmetic).
- **The disk cache key includes a hash of the compiler's source.** Without it, a
  compiler change silently reused stale MSL from `~/.cache/enceladus`, which happened once
  while benchmarking.

### Known gaps

- `idx64`: pointer offsets are always `int`, so tensors of 2^31 bytes or more aren't
  addressed correctly yet. No test covers it.
- `multiple_of` and `max_contiguous` hints aren't implemented.

## M3: Reductions and 2D tiles

### What was built

`codegen/reduce.py` lowers `sum`, `max`, `min`, `argmax`, `argmin`, and `tl.reduce` with a
combine region (including tuple inputs). It reduces registers in-thread, then lanes with
`simd_shuffle_xor` (or `simd_sum`, `simd_max`, and `simd_min` when all five lane bits
reduce), then SIMD groups through one threadgroup exchange. FP16 and BF16 accumulate in
FP32. `argmax` and `argmin` break ties toward the lower index. Only index bits whose
basis points into the reduced axis take part, so broadcast lanes are never counted
twice.

### Benchmarks

The following numbers come from `benchmarks/bench_softmax.py` and `bench_norms.py` at
4096 x 4096. MLX and torch are wall clock, which adds about 0.1 ms.

| Kernel | FP32 | FP16 | Target |
|---|---|---|---|
| Softmax, 8 SIMD groups | 242 GB/s (MLX 196, torch 170) | 247 GB/s (MLX 173, torch 153) | 215 GB/s or more |
| LayerNorm, `examples/03`, BLOCK=1024 | 237 GB/s | 235 GB/s | 200 GB/s or more |
| RMSNorm, `examples/06` | 241 GB/s | 249 GB/s | 200 GB/s or more |

Row sums vary by 0.4% between 128 and 1,024 threads per threadgroup (267-268 GB/s),
against a limit of 5%.

### Tests

The softmax, LayerNorm, and RMSNorm examples pass compiled and interpreted for FP32 and
FP16, including 1,000 columns and fully masked lanes. Reductions over every op and dtype,
Welford through a tuple `tl.reduce`, and argmax ties pass in both modes.

### Known gaps

- LayerNorm with BLOCK=4096 in FP16 reaches only 169 GB/s. The generated code has no
  layout conversions; the kernel rereads each row three times with 16 registers per
  thread. BLOCK=1024 meets the target.
- A layout bug made loop-carried accumulators fall back to the FP32 default layout,
  which cost an exchange per iteration. It's fixed: loop results now share their block
  argument's layout, and elementwise ops prefer anchors that aren't loop-carried.

## M4: Matmul

### What was built

- `codegen/dot.py`: `tl.dot` on `simdgroup_matrix`. The accumulator is a
  `simdgroup_matrix` array whose registers the rest of codegen reads through
  `thread_elements()`, so epilogues (bias, activation, casts) need no conversion.
  - *Direct* operands (a `desc_load`, optionally through `tl.trans`, used only by the
    dot) load fragments straight from device memory, with `transpose_matrix` for
    transposed operands. No threadgroup memory, no barriers.
  - *Staged* operands (any other tile) go through threadgroup memory with rows padded by
    16 bytes.
  - The accumulator updates in place when the dot is its only use.
- Tensor descriptors: `make_desc`, `desc_load` (zero-filled out of bounds), and
  `desc_store` (skips out-of-bounds elements) outside direct dots.
- Layout assignment gives `dot` results `simd_acc(BM, BN, WM, WN)` with WN = 1 by default,
  spreading SIMD groups along N only when BM is too small.
- `runtime/device.py` checks the `simdgroup_matrix` lane layout once per process before
  the first `tl.dot` compiles, and refuses `tl.dot` if the layout differs.
- `examples/04_matmul.py` has both variants; `examples/07_matmul_fused.py` fuses bias
  and GELU.

### Benchmarks

The following numbers come from `benchmarks/bench_matmul.py`, config 64x64x32 with 4 SIMD
groups, in TFLOPS (Enceladus GPU-timed; MLX and torch wall clock):

| Shape | FP32 (MLX, torch) | FP16 (MLX, torch) | BF16 (MLX, torch) |
|---|---|---|---|
| 4096³ | 4.80-5.04 (5.24, 5.25) | 5.61 (5.79, 5.81) | 5.61 (5.82, 5.91) |
| 2000³ | 4.29 (4.79, 4.93) | 4.75 (5.36, 5.27) | 4.77 (5.33, 5.27) |
| 513³ | 1.75 (1.42, 1.26) | 1.87 (1.43, 1.43) | 1.95 (1.30, 1.30) |
| 1024x4096x1024 | 5.34 (4.91, 5.06) | 5.75 (5.43, 5.46) | 5.76 (5.47, 5.44) |

- The FP16 target (5.3 TFLOPS) is met. The FP32 target (4.9) is met in most runs; the
  FP32 number varies 4.8-5.0 between runs. In the same harness, the reference
  `best_matmul.metal` measured 4.95-5.09 FP32 and 5.68 FP16, so the generated kernel is
  within about 2% of the reference.
- The fused bias-plus-GELU epilogue runs within 2.3% of plain matmul (target 5%).
- Ragged 2000³ is about 10% behind MLX, the cost of edge tiles.

### Performance findings

- **Pointer arithmetic shape matters by 17%.** `p + r * ld + (c0 + j * 8)` ran at 4.69
  TFLOPS in FP16, and `p + r * ld + c0 + j * 8` at 5.59. Adding each term to the pointer
  lets Metal fold the constant into the address; grouping the terms into one `int` sum
  first blocks it. Codegen now adds column terms to fragment pointers one at a time.
- Accumulator copies per iteration, the in-loop edge branch, fully unrolled versus looped
  fragment code, `max_total_threads_per_threadgroup`, signed versus unsigned index math,
  and scalars in constant buffers versus locals all measured within noise.

### Tests

386 tests pass and 18 are skipped (1.2 s). The skips are large shapes left to
compiled mode because the interpreter is slow on them. New tests: both matmul variants
at 64³, 513³, 1000x777x300, and 2048³ in FP32, FP16, and BF16; descriptor matmul with
transposed B and ragged shapes; the fused epilogue; and a check that the descriptor
matmul uses no threadgroup memory while the pointer variant does.

### Deviations from the plan

- **No `edge_versioning` pass.** Each direct `dot` takes a threadgroup-uniform branch:
  the whole block in bounds runs unmasked `simdgroup_load`s, and anything else runs a
  checked path where each fragment tests its own bounds (uniform across the SIMD group)
  and only straddling fragments use per-lane masked loads. One branch per K step
  measured within noise of hoisting it outside the loop, so neither loop versioning nor
  peeling the K tail is needed.
- **No vector-of-2 epilogue stores.** The epilogue stores elements one at a time with
  masks; replacing it with the reference's `vec<T, 2>` stores measured within noise.
- The register-operand path (an accumulator feeding a second `dot`) is left for M7,
  where attention needs it.

### Known gaps

- Integer `dot` raises an error (the MPP path in M8 covers it).
- `dot_warps` from `enceladus.Config` arrives with the autotuner in M5.

## M5: Autotuning and benchmarking

### What was built

- `runtime/autotuner.py`: `enceladus.Config`, `@enceladus.autotune`, and `@enceladus.heuristics`.
  - Candidates compile in parallel on up to `min(8, maximumConcurrentCompilationTaskCount)`
    threads. A config that fails to compile is skipped with a warning; if every config
    fails, the error names the first failure.
  - Each config is timed with `do_bench` (GPU timestamps, warm-up, median). NumPy
    arguments are wrapped as `enceladus.Tensor` views for timing so launches don't
    synchronize.
  - Configs slower than 3x the median are rejected as probable spill cliffs (logged at
    debug level).
  - `reset_to_zero`, `restore_value`, `pre_hook`, `prune_configs_by` (`early_config_prune`,
    `perf_model` with `top_k`), `ENCELADUS_PRINT_AUTOTUNING=1`, and `config_for()`.
  - Results persist in `~/.cache/enceladus/autotune/<kernel-hash>/<architecture>.json`,
    keyed by the key-argument values and the argument dtypes.
- `dot_warps=(WM, WN)` is a launch option and a `Config` field, and it's part of the
  specialization key.
- `enceladus/configs.py`: `matmul_configs(dtype)`, the pre-validated list from the research
  sweep (6 configs for FP32, 8 for FP16 and BF16).
- `examples/04_matmul.py` gained `matmul_tuned`; `benchmarks/run_all.py` writes
  `benchmarks/results/<date>-<arch>.md`.

### Benchmarks

The autotuned matmul matches the fixed 64x64x32 configuration at 4096³ (FP32 4.77
against 4.76, FP16 5.59 against 5.59) and at 1024x4096x1024 (5.34 against 5.33 FP32, 5.74
against 5.75 FP16). It's 22-30% faster at 513³ (2.26-2.45 TFLOPS against 1.75-1.93). The
full report is in `benchmarks/results/2026-09-26-applegpu_g16s.md`.

### Tests

`tests/test_autotune.py` covers skipping a config that fails to compile (and the error
when all fail), `reset_to_zero` around benchmark runs of an in-place kernel, reuse of a
persisted result by a fresh autotuner with benchmarking disabled, and `@heuristics`.
The whole suite runs 390 tests, with 18 skipped, in 1.3 s.

### Deviations from the plan

- `dot_backend` accepts only "auto" and "simdgroup" until M8 adds "mpp".
- Autotuning compiles don't count toward the recompilation warning.

### Known gaps

- FP32 matmul at 4096³ measured 4.76-5.04 TFLOPS across runs in this session, around
  the 4.9 target. MLX measured 5.23-5.26 in the same runs, so Enceladus is at 91-96% of MLX.
  The reference kernel measured 4.95-5.09 in the same harness.
- The recompilation warning can name a misleading argument when many launch
  configurations are in play; its hint is only a heuristic.

## Fixes after M5

An audit that compared compiled, interpreted, and NumPy results found several silent
miscompiles and autotuning defects. Each fix has a regression test that fails on the
previous code. The suite runs 476 tests, with 18 skipped, in about 1.7 s.

### Miscompiles fixed

- **Generated names shadowed user names.** A kernel argument named `r`, `i`, `j`, `kk`,
  `fa`, `fb`, `buf`, or `tk` captured a local in the generated code. `RESERVED` in
  `codegen/emitter.py` now lists every fixed identifier that codegen and the prelude emit.
  It also lists every object-like macro in the `metal_stdlib` headers (such as `INT_MAX`
  and `M_SQRT2_F`), the `ray_data`, `object_data`, and `threadgroup_imageblock` address
  spaces, and `main`, which MSL forbids as a kernel name. Names that start with `METAL_`
  or `TARGET_OS_` are escaped by prefix, because those macro families grow with the SDK.
- **`tl.dot` overwrote an accumulator that a loop reads again.** `acc0 = tl.dot(a, b)`
  followed by `tl.dot(a, b, acc0)` in a loop accumulated across iterations. In-place
  updates now require that the accumulator owns its storage and that only `if` regions
  sit between its definition and the dot. The loop-carried `acc = tl.dot(a, b, acc)`
  pattern still updates in place.
- **Descriptor accesses at negative offsets weren't masked.** Loads, stores, and direct
  `dot` operands now check `index >= 0` as well as `index < shape`.
- **64-bit pointer offsets were truncated to 32 bits.** A kernel that adds an `int64`,
  `uint64`, or `uint32` offset to a pointer tile now uses `long` offsets for all its
  pointer tiles. Generated MSL for every other kernel is unchanged. The `idx64` gap for
  tensors of 2^31 bytes or more remains.
- **`p - k` wrapped around for the smallest `int8` and `int16` values.** `-(-128)` is
  -128 in `int8`, so the pointer moved the wrong way. Offsets narrower than 32 bits are
  now negated in `int32`, and `uint32` offsets in `int64`.
- **`and` and `or` returned booleans.** They now follow Python's semantics in compiled
  mode, as in the interpreter: `a or b` is `a` when `a` is truthy, and the right side runs
  only when needed. Both operands need the same type. On tiles, `and` and `or` raise an
  error that suggests `&` and `|`. A compile-time right side that decides the result
  folds, as it did before this change: `c and False` is `False` and `c or True` is `True`
  in a test, or when `c` is a comparison. This keeps `if pid > 0 and HAS_BIAS:` with
  `HAS_BIAS=False` from compiling a load through a `None` pointer.
- **Wide integers converted to `half` and `bfloat` without an FP32 step.** `half(long)`
  overflowed from 65505 instead of 65520. These casts now go through `float`, which
  matches the interpreter and NumPy.

### Autotuning fixes

- `matmul_configs` recognizes NumPy scalar types such as `np.float32`, so FP32 tuning no
  longer includes the excluded 32 x 32 strips. Unsupported dtypes raise `ValueError`.
- `examples/04_matmul.py` prunes the config list by the dtype of `a`.
- A result loaded from the disk cache maps back to the matching candidate, so it keeps
  its `pre_hook`. The match happens at the first launch for its key, against the configs
  that `early_config_prune` returns, so a prune function that builds new `Config` objects
  still reuses saved results. A result that matches no candidate is re-tuned.
- A config that fails at run time is skipped with a warning, like a compile failure.
  This catches any exception, which is broader than Triton's list.

### Benchmarks

In an interleaved A/B run in one process, the FP32 4096³ matmul measured 4.83-4.89
TFLOPS with the new bounds checks and 4.84-4.87 without them. The loop body is unchanged;
only the per-step interior test, the edge-fragment helper, and the epilogue masks gained
`>= 0` terms.

### Known gaps

- The GPU flushes FP32 denormals to zero, and the interpreter keeps them, so the two
  modes differ for denormal inputs.
- In the interpreter, `n and tile` returns the tile when `n` is truthy. Compiled mode
  refuses it. `not tile` compiles to an elementwise not, and the interpreter raises.
- `x[:, None] + y[None, :]` at shapes such as 2 x 512 is refused as needing more than 256
  registers, because the broadcast view holds every copy in registers.
- `p - k` for an `int32` `k` of -2^31 still wraps around. Negating it in `int64` would
  move the pointer by 2^31 elements, which the `idx64` gap already rules out.
- Adding a 0-d tile, such as `tl.full((), 4, tl.int32)`, to a scalar pointer is refused
  with "can't broadcast a tile of shape () to a scalar". This predates the fixes.
- If configs share an identity but have different `pre_hook`s, a saved result maps back
  to the first of them, which might not be the one that won the benchmark.

## M6: Framework interop

### What was built

- `runtime/interop.py` accepts PyTorch tensors on the `mps` device and MLX arrays as
  kernel arguments. It detects them by `type(obj).__module__`, so importing Enceladus
  imports neither framework.
  - A PyTorch tensor binds `t.untyped_storage().data_ptr()` (the `id<MTLBuffer>`) at
    the byte offset `t.storage_offset() * t.element_size()`, with its shape and strides.
  - An MLX array is evaluated with `mx.eval()`. Its `id<MTLBuffer>`, byte offset, shape,
    and strides come from its `kDLMetal` DLPack capsule, which `_C.dlpack_inspect` reads
    without consuming it. The launch retains the buffer and holds the array and the
    capsule until the GPU finishes. The dtype comes from `a.dtype`, because MLX's
    capsule reports `bool` as a 32-bit integer.
  - CPU tensors are refused with an error that suggests `.to("mps")`. `float64` is
    refused for NumPy, PyTorch, and MLX.
  - Specialization facts are correct for both frameworks: divisibility by 16 comes from
    the byte offset, not from `data_ptr()`.
  - `enceladus.new_empty(like, shape, dtype)` allocates an output of the caller's kind
    (NumPy, `enceladus.Tensor`, PyTorch, or MLX). `enceladus.element_strides(x)` returns
    strides in elements for every kind. The examples use both, so one host wrapper
    serves all four kinds.
- `runtime/launcher.py` has three launch paths:
  - **PyTorch path.** When every array argument is an MPS tensor, the generated MSL runs
    through `torch.mps.compile_shader` on PyTorch's MPS stream, so it orders with
    surrounding PyTorch operations. Libraries are cached per SHA-256 of the source. The
    launch uses `threads = (grid[0] * tg, grid[1], grid[2])` and
    `group_size = (tg, 1, 1)`. The argument call is generated once per kernel.
  - **Synchronized native path.** Any other launch that involves PyTorch or MLX memory
    (PyTorch mixed with other array kinds, a kernel that `compile_shader` rejects, or MLX
    arrays) calls `torch.mps.synchronize()`, dispatches on Enceladus's stream, and waits
    for it. A PyTorch fallback logs one warning per kernel on the `enceladus` logger.
  - The native path is unchanged for `enceladus.Tensor` and NumPy arguments.
- Raw `enceladus.metal_kernel` launches take the same PyTorch and synchronized paths,
  with scalar types from pipeline reflection.
- `enceladus.Tensor.__dlpack__` and `__dlpack_device__` export `kDLMetal` capsules whose
  `data` is the `id<MTLBuffer>` and whose `byte_offset` locates the view. The export
  waits for Enceladus's stream first. `max_version >= (1, 0)` selects a
  `dltensor_versioned` capsule. `torch.from_dlpack` and `mx.from_dlpack` both import
  these capsules without copying; writes on either side are visible on the other. The
  capsule code is in `_C/bindings.mm` (`dlpack_export`, `dlpack_inspect`).
- The interpreter accepts PyTorch and MLX arguments: it runs on NumPy views of their
  shared memory after waiting for the framework.
- The autotuner benchmarks PyTorch and MLX arguments through `enceladus.Tensor` views
  of their memory, because `do_bench` times Enceladus's stream. The final launch takes
  the PyTorch path.

#### How `compile_shader` binds arguments (torch 2.14)

- Argument `i` binds at `[[buffer(i)]]`, which matches Enceladus's ABI. Tensors bind at
  `storage_offset() * element_size()`, so views with nonzero offsets work.
- A Python `int` binds as `int64`, a `float` as `float32`. `arg_casts` takes a
  `dict[int, str]`, and only `"int8"`, `"int16"`, `"int32"`, and `"uint8"` are accepted
  for ints. There's no half or bfloat cast for floats.
- The adapter maps each ABI scalar type as follows: `i1` to an `int8` cast; `i8`, `i16`,
  `i32`, and `u8` to their casts; `u16` and `u32` to the signed cast of the same width
  with the same bit pattern; `i64` and `u64` (wrapped to signed) to the default `int64`;
  `f32` to the default `float`; and `f16` and `bf16` to a 0-d CPU tensor, which
  `compile_shader` binds by value with `setBytes`.
- `compile_shader` compiles with MSL 4.0, safe math, and precise math functions.
  Enceladus prepends `#pragma METAL fp math_mode(relaxed)` (or `fast`) to match its
  native semantics; without it, `x != x` NaN tests behave differently. MSL has no
  pragma for the math-function precision, so `exp` and similar functions are precise
  (1.4 ULP measured) on the PyTorch path and fast (61 ULP) on the native path.

### Benchmarks

Preliminary: other agents shared the GPU during these runs. `benchmarks/bench_dispatch.py`
measured the following, as the minimum and median of 5 runs of 10,000 launches:

| Launch | Min | Median | Target |
|---|---|---|---|
| `@enceladus.jit` on PyTorch MPS tensors, sustained | 4.89-5.03 µs | 5.00-5.07 µs | 5 µs or less |
| `torch.mps.compile_shader` call made directly, sustained | 2.31 µs | 2.34 µs | Not applicable |
| `torch.add(out=)`, sustained | 2.04 µs | 2.08 µs | Not applicable |
| `@enceladus.jit` on `enceladus.Tensor`, sustained | 3.36 µs | 3.38 µs | 5 µs or less |
| `@enceladus.jit` on PyTorch MPS tensors, sync round trip | 66-75 µs | 116 µs | Not applicable |
| `@enceladus.jit` on MLX arrays, sync round trip | 101 µs | 133 µs | About 100 µs |

The PyTorch path is host-bound. The host enqueues a launch in about 4.0 µs, and
PyTorch's stream adds about 1 µs per launch after the call returns, for direct
`compile_shader` calls too. Two changes to the shared launch path brought the sustained
cost from 7.0 µs to about 5 µs: a fast path for 1-tuple grids, and list comprehensions
instead of generators in the specialization key. They also moved `@enceladus.jit` on
`enceladus.Tensor` from 3.8 µs to 3.4 µs.

### Tests

`tests/test_interop.py` has 42 cases; the module skips when PyTorch or MPS is missing,
and the MLX cases skip when MLX is missing:

- The vector add, softmax, pointer matmul, and descriptor matmul examples run on PyTorch
  and MLX arrays in FP32 and FP16, with and without a 3-row offset (byte offsets that
  aren't 16-byte aligned). Each case runs `check_kernel` in compiled and interpreted
  modes against NumPy, and asserts that PyTorch launches don't take the synchronized
  fallback.
- A raw kernel binds every ABI scalar type at its extreme values on the PyTorch path,
  and a `@enceladus.jit` kernel binds `i1`, `i32`, `i64`, and `f32`, compared bit for bit.
- 200 iterations of "PyTorch `fill_` writes, Enceladus reads and writes, PyTorch reads"
  run without a host sync and see no stale data. With the ordering broken on purpose
  (native dispatch without `torch.mps.synchronize()`), the test fails.
- DLPack export to PyTorch (legacy and versioned capsules) and to MLX shares memory in
  both directions and waits for a pending Enceladus launch.
- A launch that mixes PyTorch tensors with an `enceladus.Tensor` falls back, returns
  correct results without an explicit sync, and logs once.
- CPU tensors, `float64` arrays, and writes to a broadcast MLX array are refused.

The whole suite runs 518 tests, with 18 skipped, in about 2 s.

### Deviations from the plan

- The PyTorch path prepends a math-mode pragma to the source, because `compile_shader`
  compiles with safe math. Math functions stay precise there, so results can differ from
  the native path in the last bits.
- Half and bfloat scalar arguments bind as 0-d CPU tensors, because `arg_casts` has no
  float casts. Each such argument allocates a small CPU tensor per launch.
- A launch that mixes PyTorch tensors with other array kinds takes the synchronized
  native path, as the plan's fallback describes.
- `check_simdgroup_layout` in `runtime/device.py` holds a lock. Autotuning compiles on
  several threads, and concurrent probes raced on the stream, which isn't thread-safe.
  With PyTorch arguments, `matmul_tuned` crashed in one run and deadlocked in the
  next. The race predates M6.

### Known gaps

- The lazy MLX integration through `mx.fast.metal_kernel` isn't implemented. A probe
  shows that it's feasible: the vector add kernel, renamed from `[[kernel]]` to an
  `inline` function in `header`, with its attributes stripped and a body that passes
  MLX's `threadgroup_position_in_grid` and related values, ran correctly, and a chain of
  1,000 dependent launches cost 7.7 µs per launch against about 100 µs for the
  synchronous path. The blockers for general use:
  - MLX allocates outputs, so kernels that read or partially write an output need
    `init_value` or an input copy to keep in-place semantics.
  - Threadgroup memory is declared inside the kernel and must move into the body.
  - Scalars must become 0-d arrays or template constants.
  - mlx#4534 clamps `group_dims` to the grid.
- MLX launches are synchronous (about 100 µs each). Outputs must be arrays allocated for
  the purpose, such as `mx.zeros(shape)` followed by `mx.eval()`, or
  `enceladus.new_empty(like)`. Enceladus writes to them in place, which MLX's immutable
  arrays don't expect: an array that MLX shares or reuses, such as one from
  `mx.broadcast_to` (refused) or a lazily computed result, can't be an output.
- An `enceladus.Tensor` exported through DLPack is ordered with the consumer only at
  export. Launches on the original `enceladus.Tensor` after that need
  `enceladus.synchronize()` before the consumer reads; launches on the imported PyTorch
  tensor take the PyTorch path and order themselves.
- Raw `metal_kernel` launches on PyTorch tensors take about 5.4 µs of host time, because
  their path rebuilds the argument mask on each launch. They weren't optimized.
- The sustained PyTorch launch cost sits at the 5 µs target within noise, not clearly
  under it. Most of the host time is the shared `@enceladus.jit` binding and
  specialization-key work.

## M7: Atomics and scans

This entry covers the atomics and scans parts of M7. Flash attention has its own entry.

### What was built

- Atomics: `tl.atomic_add`, `atomic_max`, `atomic_min`, `atomic_xchg`, `atomic_and`,
  `atomic_or`, `atomic_xor`, and `atomic_cas` on pointers and pointer tiles. They take
  masks (except `atomic_cas`, as in Triton) and return the old values, or 0 where the mask
  is false. `sem` accepts only `None` and `"relaxed"`; `scope` accepts `None`, `"gpu"`, and
  `"cta"`. Anything else raises an error.
- Atomic lowering, in `compiler/codegen/atomic.py`:
  - Native `atomic_int` and `atomic_uint` operations for 32-bit integers, and
    `atomic_float` add and exchange for `float32`.
  - Compare-and-swap loops on the aligned 32-bit word for everything else: `float32` max
    and min, `atomic_cas` of every width, and 8-bit and 16-bit elements (including
    `float16` and `bfloat16` max, min, and exchange). The loops compare bit patterns, so
    they end even for NaNs. Float max and min ignore NaN operands, like `tl.maximum`.
  - `uint64` max and min through `atomic_max_explicit` on `atomic_ulong`, only when the
    result is unused and the device is Apple9 or later.
  - Atomics run only in the owning thread of each element when the layout has broadcast
    lane or SIMD-group bits. The owner then shares the old value, with `simd_shuffle` for
    lane bits or through threadgroup memory for SIMD-group bits. A scalar atomic runs in
    thread 0 and shares its old value through threadgroup memory.
- Scans: `tl.cumsum(x, axis=0, reverse=False, dtype=None)` and
  `tl.associative_scan(x, axis, combine_fn, reverse=False)`, including tuples of tiles.
  `cumsum` sums integers narrower than 32 bits (and `int1`) in 32 bits, like `tl.sum`,
  and accumulates `float16` and `bfloat16` in `float32`. The `combine_fn` needs to be
  associative but not commutative. A reverse scan equals flip, scan, flip.
- Scan lowering, in `compiler/codegen/scan.py`, handles any `BitLayout`. It walks the
  axis bits from least to most significant in runs of one kind (register, lane, or SIMD
  group), because layouts interleave them. For example, a 1D blocked tile of 1,024
  elements has register, lane, SIMD-group, and register bits, in that order. Register
  runs combine in sequence in each thread. Lane runs use `simd_prefix_exclusive_sum` for
  a forward sum over all 32 lanes, and a Hillis-Steele scan with `simd_shuffle`
  otherwise. SIMD-group runs, and any run above a SIMD-group run, exchange block totals
  through threadgroup memory. Each element then applies `combine(prefix, value)`.
- The layout pass gives scan results their first input's layout and gives atomic results
  the layout of their value, else their pointer, like `store`.
- Helper functions that only atomics and scans use, such as `tg_shfl_idx` and the
  compare-and-swap loops, go into the kernel source only when the kernel uses them.
- `examples/09_histogram.py` bins float data with `tl.atomic_add`, and
  `examples/10_cumsum.py` computes row-wise cumulative sums over blocks of rows.

### Benchmarks

These preliminary numbers come from `benchmarks/bench_scan_atomics.py`, taken while other
agents shared the GPU:

- Row cumsum, 4096 x 4096 `float32`, one row per program: 0.56 ms, 239-240 GB/s at 4, 8,
  and 16 SIMD groups. MLX `mx.cumsum(axis=1)` measured 193 GB/s by wall clock.
- Histogram of 2^24 `float32` values: 31 GB/s with 64 bins and 63 GB/s with 4,096 bins.
  Contention on few global counters limits it.

### Tests

`tests/test_atomics_scans.py` adds 96 test cases, and the suite runs 572 tests, with 18
skipped, in about 2 s. The tests cover the following:

- Every atomic operation against the interpreter and NumPy, on aligned and ragged sizes
  with masks, across native paths, compare-and-swap loops (`float32` max, `atomic_cas`),
  and 8-bit and 16-bit elements that neighbor elements that other threads update at the
  same time.
- Broadcast layouts (16 or 64 elements over 32 or 128 threads). The count per element
  catches duplicate atomics, and a `tl.cumsum` of the old values catches non-owner
  threads that hold stale old values. Removing either the predication or the sharing
  fails the test.
- Scalar atomics that hand out unique tickets, `uint64` max with colliding addresses, and
  the errors for `float16` addition and for using a `uint64` max result.
- `tl.cumsum` through the example in five dtypes (including `bfloat16` and `int64`, which
  takes the generic lane scan), three shapes, and both directions.
- `tl.associative_scan` along both axes of 2D tiles, forward and reverse, with `max` and
  with a non-commutative tuple scan (composition of affine maps in `int32`). Swapping the
  combine order fails these tests.
- Scans over a `tl.dot` accumulator, whose lane bits are out of order, and over a
  flattened transpose, which puts a SIMD-group bit below the lane bits.

### Deviations from the plan

- 64-bit atomics are limited to `uint64` max and min with an unused result. Metal on
  this machine accepts only `atomic_max_explicit` and `atomic_min_explicit` on
  `atomic_ulong`, which return nothing. There's no 64-bit compare-and-swap to emulate the
  rest, so the frontend refuses them in both modes.
- `combine_fn` must be a `@enceladus.jit` function, as in `tl.reduce` and Triton. Plain
  Python functions are refused, because the kernel's dependency hash doesn't cover them
  and a disk-cache entry could go stale.
- `tl.atomic_and`, `atomic_or`, and `atomic_xor` were added. They cost nothing on the
  native path.
- `bfloat16` addition is refused with the same error as `float16` addition.
- Codegen reads the device family for `uint64` atomics from `module.attrs["apple_family"]`,
  falling back to the default device.

### Known gaps

- Emulated 8-bit and 16-bit atomics use a compare-and-swap loop on the surrounding
  32-bit word. A plain store by another thread to a neighboring element of the same word
  can be lost while the loop runs. The loop also reads up to 3 bytes past the last element
  of a buffer whose size isn't a multiple of 4 bytes.
- The interpreter applies colliding atomics in row-major order. The GPU order differs, so
  old values at colliding addresses, and `float32` sums, can differ between modes.
- The interpreter doesn't know whether a `uint64` max result is used, so only compiled
  mode refuses using it.
- There are no threadgroup-memory atomics, so a histogram can't privatize its counters
  per threadgroup.
- A scan whose axis bits interleave SIMD-group and register bits exchanges through
  threadgroup memory once per run, with two barriers each. A 1D blocked tile of 1,024
  elements takes two exchanges.

## M7: Flash attention

### What was built

- `examples/08_flash_attention.py`: a flash attention forward kernel in the style of
  Triton's fused-attention tutorial. One program handles `BLOCK_M` query rows of one
  (batch, head) pair and loops over key blocks with an online softmax (running maximum and
  sum, `exp2` with the scale folded in) and an FP32 accumulator. It takes a softmax scale
  and a `CAUSAL` constexpr, and handles sequence lengths that aren't block multiples. The
  host wrappers are `attention` (fixed configuration) and `attention_tuned` (autotuned).
- `enceladus.configs.attention_configs(dtype, head_dim)`: MLX's `steel_attention` shapes
  (32 query rows over 4 SIMD groups, 16 or 32 keys per step), plus 64 rows over 8 SIMD
  groups, plus 16-row strips for head dimensions of 64 or less. Every configuration sets
  `dot_warps=(num_warps, 1)`.
- `tl.dot` register operands (`codegen/dot.py`): a left operand in the dot's register
  operand layout becomes the A fragments through `thread_elements()`.
  `layout.dot_operand_a(BM, BK, WM, WN)` defines that layout; with WN = 1 it equals
  `simd_acc(BM, BK, WM, 1)`, so `tl.dot(p.to(tl.float16), v, acc)` after
  `p = exp2(tl.dot(q, tl.trans(k)) - m[:, None])` moves no data. A cheap operand is
  rematerialized in that layout, and an operand that a register remap reaches is remapped.
  Anything else, including an accumulator from a WN > 1 grid, is staged through
  threadgroup memory as before.
- Hoisted left operands (`passes/layouts.py`): a `load` or `desc_load` whose only uses are
  left operands of dots in a nested loop gets the register operand layout, so the query
  tile loads once, as `simdgroup_matrix` fragments, and stays in registers. The share per
  thread must fit in 32 registers (`MAX_HOISTED_REGS`); larger operands keep the direct
  path. The K loop of a dot with a register operand unrolls in the generated code, because
  a register array indexed by `kk / 8` inside a loop that Metal didn't unroll spilled to
  the stack and halved throughput at head dimension 128.
- Staged right operands for shared B (`dot.stages_b`): when A comes from registers or
  threadgroup memory and WM > 1, the dot stages B through threadgroup memory even when a
  direct load is possible. All SIMD-group rows read the same K and V blocks, and one
  cooperative load per threadgroup beats a device load per SIMD group. A matmul whose
  operands both load directly keeps its direct path.
- One B fragment at a time: when the accumulator and the TN live B fragments of a K step
  exceed 48 registers per thread, the MMA step loads and uses one B fragment at a time.
  Only FP32 dots with TN = 16 and TM = 1, or larger strips, cross the threshold; no matmul
  configuration in `matmul_configs` does.
- The accumulator row reduction needed no change: with WN = 1, `tl.max(s, 1)` and
  `tl.sum(p, 1)` reduce in registers and then shuffle over lane bits 0 and 3, with no
  threadgroup memory. `m[:, None]` broadcasts back into the accumulator layout by register
  copies.
- `benchmarks/bench_attention.py`, hooked into `benchmarks/run_all.py`.
- A runtime fix, found while tuning: see the first known gap.

### Benchmarks

The following table comes from `benchmarks/bench_attention.py` with FP16 inputs of shape
(1, 16, N, D) and the autotuned configuration. Enceladus uses GPU timestamps; MLX
(`mx.fast.scaled_dot_product_attention`) uses the wall clock around a synchronized call.
The two alternate in one process for three rounds of 10 runs. Values are TFLOPS as the
best run (median of round medians), and the ratio is Enceladus time over MLX time.
Other jobs shared the GPU during these runs, so treat the numbers as preliminary.

| N, D | Causal | Config | Enceladus | MLX | Ratio |
|---|---|---|---|---|---|
| 2048, 64 | No | 64x32, 8 SIMD groups | 5.11 (4.88) | 5.17 (4.95) | 1.01x |
| 2048, 64 | Yes | 32x32, 4 | 4.83 (4.66) | 4.77 (4.56) | 0.99x |
| 2048, 128 | No | 32x32, 4 | 4.71 (4.57) | 5.10 (4.90) | 1.08x |
| 2048, 128 | Yes | 64x32, 8 | 4.73 (4.49) | 4.84 (4.62) | 1.02x |
| 4096, 64 | No | 64x32, 4 | 5.08 (5.00) | 5.31 (5.12) | 1.05x |
| 4096, 64 | Yes | 32x32, 4 | 4.76 (4.59) | 5.08 (4.86) | 1.07x |
| 4096, 128 | No | 64x32, 8 | 4.92 (4.74) | 5.09 (5.03) | 1.03x |
| 4096, 128 | Yes | 64x32, 8 | 4.68 (4.53) | 4.95 (4.89) | 1.06x |

Every case is within the 1.3x target. An earlier run gave 0.98-1.09x. The steps, measured
at N = 2048 in FP16 with fixed configurations:

| Step | D = 64 | D = 128 |
|---|---|---|
| M4 paths only (P staged, Q reloaded per iteration) | 4.22 | 3.83 |
| P and Q from registers, K loop indexed by `kk / 8` | 4.34 | 2.47 |
| Unrolled K loop for register operands | 5.09 (16-row strips) | 4.12 |
| K and V staged through threadgroup memory | 5.23 | 4.99 |

FP32 attention at D = 128 ran at 0.35 TFLOPS before the one-B-fragment rule and 4.3
after it (N = 1024). The FP16 and FP32 4096³ matmul benchmarks are unchanged (5.58-5.71
and 4.92-5.26 TFLOPS in one run).

### Tests

The suite runs 505 tests, with 18 skipped, in about 4.5 s. `tests/test_attention.py` adds
the following tests:

- A differential test of the example (compiled, interpreted, and NumPy) over 10 cases:
  aligned and ragged sequence lengths (64, 100, 37), head dimensions 32, 64, and 128,
  causal on and off, a non-default scale, FP16, FP32, and BF16, and configurations that
  reach each operand path (8- and 16-row strips, staged and direct keys, one B fragment at
  a time).
- A register-operand test: an accumulator cast to FP16 feeds a second dot in a loop, with
  a hoisted left operand and ragged sizes in every dimension. With `dot_warps=(4, 1)`, the
  kernel allocates no staging buffer for the left operand; with `(2, 2)`, it stages it.
- A row-reduction test: the row maximum of a dot result uses no threadgroup memory and no
  barrier with WN = 1, and uses threadgroup memory with WN = 2.
- A subprocess test that autotunes the attention kernel as the first `tl.dot` of a fresh
  process, which crashed with a segmentation fault before the runtime fix.

Mutations of the fragment index math, the register operand layout, the fragment loads,
the staging rule, and the one-B-fragment loop each fail at least one test.

### Deviations from the plan

- **K and V are staged, not loaded directly.** The plan says to load K with a transposed
  direct `simdgroup_load`. With A in registers, staging K and V through threadgroup memory
  measured 3-21% faster, so `stages_b` stages them. The transposed direct load still serves
  matmuls. Staging K untransposed and loading its fragments transposed from threadgroup
  memory measured 2-3% slower than writing it transposed, so staging keeps its logical
  orientation.
- **The query tile stays in registers.** The plan doesn't mention it. Reloading Q from
  device memory every iteration measured up to 16% slower, depending on the configuration.
- **Loads and MMAs can interleave.** The one-B-fragment rule departs from the reference
  kernel's loop shape, but only above the register threshold.
- **The attention configurations include more than MLX's.** 64 query rows over 8 SIMD
  groups won most tunings at D = 128.
- **A runtime file changed.** `runtime/device.py` gained a lock (see the first known gap),
  outside the files this milestone was meant to touch.

### Known gaps

- `check_simdgroup_layout` ran its probe kernel without a lock, so autotuning a dot kernel
  as the first `tl.dot` of a process launched it from several compile threads at once and
  crashed with a segmentation fault. It takes a lock. Earlier milestones missed it because
  their tests and benchmarks compiled a dot before tuning.
- Persisted autotuning results are keyed by the kernel's source hash, not the compiler
  version, so a compiler change keeps old winners until you delete
  `~/.cache/enceladus/autotune/<kernel-hash>/`.
- Causal attention masks every key block. Skipping the mask for blocks entirely below the
  diagonal, as Triton's tutorial does with two loops, isn't implemented; causal runs are
  within 1-7% of MLX without it.
- The register-operand path serves only left operands. A right operand in registers
  (`tl.dot(a, acc)`) is staged.
- 16-row strips at head dimension 128 spill (0.3-0.6 TFLOPS) and are left out of
  `attention_configs`. FP32 attention reaches about 4.3 TFLOPS at D = 128 and 4.7 at 64;
  MLX wasn't measured in FP32.
- Staging B measured 5% faster than direct loads for the FP32 4096³ matmul in one run
  (5.12 against 4.86 TFLOPS) and 3% slower in FP16. The matmul path keeps direct loads; a
  dtype-aware rule is worth measuring in a later milestone.
- Only the forward pass exists. Dropout, attention masks other than causal, grouped-query
  attention, and query and key lengths that differ aren't supported by the example.

## M8: Metal 4 matmul2d backend

### What was built

- **`dot_backend` option.** `kernel[grid](..., dot_backend=...)`, `warmup`, `explain`,
  and `enceladus.Config(dot_backend=...)` take `"auto"`, `"simdgroup"`, or `"mpp"`. The
  option is part of the in-process specialization key, and the resolved backend is part
  of the disk-cache key. `runtime/dot_backend.py` resolves it: `"auto"` picks `"mpp"` on
  Apple10 and later GPUs and `"simdgroup"` on earlier ones. `"mpp"` needs
  `supportsFamily(Metal4)`, macOS 26 or later, and a probe kernel that compiles and
  returns the right 16 x 16 product. The probe runs once per process under the lock of
  the `simdgroup_matrix` layout probe, because both launch on the shared stream from
  autotuning's compile threads. On a device that fails the probe, `"mpp"` falls back to
  `"simdgroup"` with a debug log.
- **Eligibility** (`compiler/codegen/mpp.py`, `plan_mpp`). A `tl.dot` lowers to
  `matmul2d` when all of the following hold. Otherwise it falls back to `simdgroup`, and
  the `enceladus` logger records the reason at debug level, such as "the accumulator
  doesn't start from zeros" or "the tl.dot result feeds `reduce`, which isn't an
  elementwise op in the loop's block".
  - It sits directly in a `for` loop, and its accumulator is loop-carried, starts from
    `tl.zeros`, and is used only as `acc = tl.dot(a, b, acc)`.
  - Both operands are descriptor loads, optionally through `tl.trans`, used only by the
    dot, from descriptors created outside the loop, at offsets that a small analysis
    proves non-negative: program IDs, non-negative constants, loop counters with a
    non-negative start and a positive step, and sums and products of those.
  - The operand and accumulator types are a `matmul2d` combination: FP16 to FP32 or
    FP16, FP32 to FP32, or BF16 to FP32 or BF16.
  - The tile is 16 to 128 in both dimensions, and the kernel uses at most 8 SIMD groups
    (see the known gaps).
  - After the loop, the result reaches exactly one `desc_store` through elementwise ops in
    the loop's block. Their other operands must be computable per element: constants,
    scalars, `arange`, `expand_dims`, `broadcast`, `trans`, elementwise ops, pointer-tile
    loads, and descriptor loads in the same block, with no memory write between those
    loads and the store.
- **Codegen.** An eligible kernel gets `#include <metal_tensor>` and the MPP header, and
  it requests MSL 4.0 through M9's `require_language_version`, which takes the maximum
  over features, so `tl.device_print` in an MPP kernel compiles with 4.0 and logging.
  Codegen builds one `tensor_inline` per operand from the descriptor's pointer, with
  extents innermost first (`(K, M)` for a row-major M x K operand) and the descriptor's
  row stride. Transposed operands set the descriptor's `transpose_left` or
  `transpose_right` flag. The op is
  `matmul2d<matmul2d_descriptor(BM, BN, K, ...), execution_simdgroups<num_warps>>`, and
  its destination is a cooperative tensor of the accumulator type.
  - A loop `for k in range(0, K, BK)` whose body is only the dot and scalar offset math,
    where `K` is the K extent of both descriptors, becomes one `run` over the whole K
    range with `tensor_ops::dynamic_length_v<int>` and `mode::multiply`.
  - Any other eligible loop, such as one that carries another value, keeps its structure
    and runs one `mode::multiply_accumulate` step of BK per iteration into a zeroed
    cooperative tensor.
  - Each `run` sits behind a threadgroup-uniform test that its slices start inside their
    tensors; otherwise the tile stays zero. `matmul2d` bounds-checks the rest.
  - The epilogue walks `get_capacity()` elements, skips those that fail
    `is_valid_element(i)`, gets (column, row) from `get_multidimensional_index(i)`,
    computes each elementwise op for that element as MSL locals, and stores with the
    descriptor's bounds check. Epilogue ops that nothing else uses emit no other code, so
    the fused example's bias load and its layout exchange disappear.
  - Generated code refers to library names only through `metal::` and `mpp::`
    qualifiers, which a user variable can't hide, and every local comes from `NameGen`,
    so `RESERVED` needs no new names. Kernels without an MPP dot keep byte-identical MSL.
- **Metadata for `explain`.** `GeneratedKernel` and `CompiledKernel` carry `dot_backend`
  ("mpp", "simdgroup", or None) and `dot_fallbacks` (the reasons), and both persist in
  `meta.json`. `build_module` sets the `dot_backend` module attribute to the resolved
  backend, codegen resets it to "simdgroup" when no dot qualifies, and `kernel.explain`
  lists the fallback reasons.
- **PyTorch path.** `torch.mps.compile_shader` compiles with MSL 4.0, and MPP kernels run
  through it unchanged. A kernel that needs a later version takes the synchronized native
  path instead.
- **Autotuning.** `matmul_configs` adds three `dot_backend="mpp"` configs (64 x 64,
  64 x 32, and 32 x 32, with 4 SIMD groups) for every dtype. On a device without
  `matmul2d`, they compile with `simdgroup`. Autotuning already compiles candidates in
  parallel. A cold MPP compile took 202 ms, and the same kernel in a second process took
  6.9 ms from Enceladus's and Metal's disk caches.
- `examples/04_matmul.py` (`matmul_desc`) and `examples/07_matmul_fused.py` take a
  `dot_backend` argument. `benchmarks/bench_matmul.py` times the simdgroup, MPP, and
  tuned kernels and both fused variants in interleaved rounds.

### Benchmarks

Preliminary: another agent shared the GPU, and MLX's own FP16 numbers varied from 4.4 to
5.8 TFLOPS at 4096³ between runs. The following table comes from one process that
alternated the three kernels for four rounds of 10 runs, with the 64 x 64 x 32
configuration and 4 SIMD groups. Values are TFLOPS as the minimum (median) time;
Enceladus uses GPU timestamps and MLX the wall clock. Each cell lists two runs.

| Shape, dtype | simdgroup | mpp | MLX |
|---|---|---|---|
| 4096³ FP16 | 5.64 (5.50), 5.61 (5.38) | 5.91 (5.74), 5.84 (5.69) | 5.81 (5.55), 5.72 (5.58) |
| 4096³ FP32 | 4.88 (4.75), 4.96 (4.79) | 4.89 (3.88), 4.49 (3.85) | 4.94 (4.86), 5.00 (4.89) |
| 2000³ FP16 | 4.66 (4.44), 4.82 (4.43) | 5.76 (5.26), 5.64 (5.31) | 5.38 (5.09), 5.31 (5.02) |
| 2000³ FP32 | 4.37 (4.05), 4.37 (4.09) | 5.07 (4.78), 5.07 (4.72) | 4.77 (4.50), 4.79 (4.55) |

- **The FP16 4096³ target (5.9 TFLOPS) is at the edge.** Across five interleaved runs, the
  MPP minimum measured 5.84-5.97 TFLOPS (3 of 5 at 5.9 or more) and the median
  5.69-5.89. In the same process, a kernel in the form of the research's
  `matmul_mpp.metal` measured 5.99 (5.91) against 5.97 (5.89) for the generated one, so
  codegen costs nothing measurable. The research's 6.19 didn't reproduce in this session.
- MPP beats `simdgroup` by 3-6% in FP16 at 4096³ and by 15-25% at 2000³. The tuned
  kernels reach 3.7-4.3 TFLOPS at 513³ (32 x 32 MPP) against 1.7-1.9 for the fixed
  simdgroup configuration and 2.3-2.5 for the tuned one in M5.
- **FP32 `matmul2d` is bimodal at 4096³.** Its minimum matches `simdgroup`, but its median
  runs 20% slower, and a sweep of tile shapes and manual K steps (16, 32, and 64) found
  nothing faster than `simdgroup` there. Tuning picks `simdgroup` for FP32 at 4096³ and
  MPP at 2000³ and 513³.
- The fused bias-plus-GELU epilogue on MPP runs within 1% of plain MPP matmul (6.00
  against 5.97 TFLOPS minimum in one interleaved run).

### Tests

The suite runs 676 tests, with 18 skipped, in about 6.3 s. `tests/test_mpp.py` adds 23
cases and skips when the device fails the probe:

- A differential test of the matmul and fused-epilogue examples on `dot_backend="mpp"`
  over an aligned and a ragged shape, FP16 and FP32, and both modes. It asserts that the
  MSL contains `matmul2d` and no `simdgroup_multiply_accumulate`, and that the kernel
  needs MSL 4.0.
- Transposed A, transposed B, and a loop that must stay a loop (manual K steps), with a
  ragged K and an epilogue that reads a descriptor and a masked pointer tile.
- A fallback test: an accumulator that starts from a loaded tile, and a result that
  feeds `tl.sum`. Both give correct results, log the reason, and generate
  `simdgroup_matrix` code without `matmul2d`.
- A disk-cache reload that must restore MSL 4.0, and a launch on PyTorch MPS tensors that
  must take the `compile_shader` path.

Mutations of the slice order, the transpose flag, the epilogue's row and column, the
broadcast index map, the multiply-accumulate mode, the zero-start check, and the cached
language version each fail at least one test.

### Deviations from the plan

- **Explicit `"mpp"` never raises.** An ineligible kernel or an unsupported device falls
  back to `simdgroup` with a debug log, as `"auto"` does.
- **Tiles are limited to 16-128 and 8 SIMD groups.** The plan sets no limit. See the
  known gaps.
- **Epilogue operands are recomputed per element.** Operands such as the bias load are
  computed for each cooperative-tensor element from its coordinates, instead of being
  loaded into a second cooperative tensor with `load()`, which needs a tensor view of the
  operand. Loads that only the epilogue uses emit no other code.
- **FP32 MPP configs stay in `matmul_configs`.** They lose at 4096³ but win at 2000³ and
  513³, and tuning measures each shape.
- **Runtime files changed.** `jit.py`, `compile.py`, `launcher.py`, and `autotuner.py`
  gained the `dot_backend` plumbing, and `compiler/explain.py` gained one line per
  fallback reason.

### Known gaps

- **`matmul2d` returns wrong results for some large tiles.** On macOS 27, a tile with a
  256-row or 256-column dimension over 1 or 2 SIMD groups (256 x 8 over 1, 8 x 256 over
  1, 32 x 256 over 2) gave wrong results, including in the research's `op.run(A, B, C)`
  form, and an 8 x 8 tile didn't compile. Eligibility therefore requires 16-128 in both
  dimensions and at most 8 SIMD groups. A sweep of every power-of-two tile from 16 to 128
  over 1-8 SIMD groups, in FP16 and FP32, on both the whole-K and manual-K paths, gave
  correct results. 16 and 32 SIMD groups weren't tested.
- The 5.9 TFLOPS target is met in some runs only (see the benchmarks). The Apple10
  default (`"auto"` picks `"mpp"`) is untested, because no M5 device was available.
- The layout pass still assigns `simdgroup_matrix` layouts, and checks the SIMD-group
  split, for a dot that lowers to MPP. So an MPP-eligible tile that `simdgroup` can't
  split (for example 16 x 16 over 8 SIMD groups) raises the `simdgroup` error, and
  `kernel.explain` lists `simdgroup_matrix` layouts for those values.
- Integer `tl.dot` still raises in the frontend; `matmul2d`'s `int8` combinations aren't
  wired up.
- An epilogue that needs anything other than elementwise ops, such as a row reduction for
  a fused softmax, falls back to `simdgroup`. Cooperative-tensor reductions
  (`reduce_rows`) and `get_left_input_cooperative_tensor` for attention aren't used.
- `relaxed_precision` stays off; it measured no difference in the research.

## M9: Debugging tools

This entry covers items 1-4 of M9: device printing, device asserts, `kernel.explain`, and
GPU capture. The error-message pass, the user guide, and wheels come later.

### What was built

- **`tl.device_print(prefix, *args)`.** The frontend emits a `print` op; codegen
  (`compiler/codegen/debug.py`) lowers it to `os_log_default.log(...)` from
  `<metal_logging>`.
  - Each line starts with the program ID, then the element's index for tiles, then the
    prefix and the values: `pid (1, 0, 0) idx (3) x: 1.500000 11`. Tile arguments
    broadcast to one shape. Each thread prints the registers it owns, predicated on the
    same owner mask as stores, so broadcast layouts (small tiles, reduction results)
    print each element once.
  - Integers print with `%d`, `%u`, `%ld`, or `%lu`, booleans as 0 or 1, and floats
    (including `half` and `bfloat`) with `%f`. The interpreter formats values the same
    way; compiled and interpreted output matched line for line for every dtype,
    including -inf, NaN, -0.0, and 64-bit extremes.
  - The codegen reports its needs through `GeneratedKernel.language_version` (3.2 for
    printing kernels) and `GeneratedKernel.enable_logging`. `require_language_version`
    combines requirements by taking the maximum. The values flow through `compile.py`,
    `meta.json` in the disk cache, and `cache.get_pipeline`. Kernels that don't print
    keep byte-identical MSL and compile options.
  - The first launch of a printing kernel moves the default stream, after a sync, to a
    command queue whose `MTLLogState` (8 MB buffer, debug level) collects messages. The
    native log handler only appends to a mutex-protected sink; it never calls into
    Python, so the GIL question doesn't arise. `Stream.synchronize` writes the collected
    lines to `sys.stderr` after the GPU finishes.
  - **Ordering guarantee.** Metal delivers messages on its own thread after the command
    buffer completes: none had arrived when `sync` returned, and messages from different
    command buffers interleave. Within one command buffer they arrive in order. So the
    stream ends every command buffer that holds a printing launch with a one-thread
    sentinel kernel that logs a fixed string, and `synchronize` waits (up to 10 s) until
    the sink has counted every sentinel. Every line a kernel prints reaches `sys.stderr`
    before the `synchronize` that waits for the kernel returns; this held for 1 to 500
    launches per sync, including command-buffer splits. Command buffers without printing
    launches get no sentinel and no wait.
  - On the PyTorch path, printing kernels (and kernels with asserts) take the
    synchronized native path, because `torch.mps.compile_shader` can't attach a log
    state. The fallback logs once per kernel.
- **`tl.device_assert(cond, msg, mask=None)`.** Active only when `ENCELADUS_DEBUG=1`;
  otherwise the frontend emits nothing.
  - The flag is part of the disk cache key and of the in-memory specialization key, so
    toggling it recompiles. `core.debug_enabled()` reads `os.environ` through its
    internal bytes dict, and launches put the raw value in the key through the bound
    `dict.get`, which keeps the launch path within noise of the previous build.
  - Codegen adds `device atomic_uint* tg_assert_buf [[buffer(N)]]` after the runtime
    arguments. A failing thread claims the buffer with a compare-and-swap and writes the
    assert index and program ID. `CompiledKernel` owns a zeroed 32-byte buffer and binds
    it on every launch path, including `timed_launch` and the synchronized path.
  - At sync, the stream reads the buffer of every assert kernel that launched since the
    previous sync, resets it, and raises `enceladus.DeviceAssertionError` with the
    source file, line, column, source text, and first failing program ID. The assert
    table (message and location per index) is stored in `meta.json`.
  - The interpreter raises the same error at the failing assert, with the same line.
- **`kernel.explain(*args, grid=..., **meta)`** (`compiler/explain.py`) builds the IR
  through the new `compile.build_module` helper, runs `compile_module`, and prints and
  returns a report with the following parts:
  - The MSL language version, logging, and assert count.
  - The `dot` backend, read as `module.attrs.get("dot_backend", "simdgroup")`, and each
    `dot` with its shape and SIMD-group grid.
  - Every anchored tile with its layout (registers, lanes, and SIMD groups per
    dimension, and how many threads hold each element), its register estimate (the
    codegen's own formula), and its source line. Operands that `tl.dot` reads straight
    from device memory show 0 registers.
  - Every layout conversion that codegen emitted: register moves or a threadgroup
    exchange with its size, both layouts, the use site, and where the value was defined.
  - Peak threadgroup memory and each request by operation and line.

  Codegen records conversions and threadgroup requests in `GeneratedKernel.report`, and
  exposes the layout plan as `GeneratedKernel.plan`.
- **`enceladus.capture(path)`** (`runtime/debug.py`) is a context manager over
  `MTLCaptureManager` that captures the device into a GPU trace document. It raises a
  `RuntimeError` that explains the requirement when `MTL_CAPTURE_ENABLED=1` is missing,
  refuses paths that don't end in `.gputrace` or that exist, syncs before it starts, and
  syncs and stops on exit, even after an exception.
- New public API: `tl.device_print`, `tl.device_assert`, `enceladus.capture`,
  `enceladus.DeviceAssertionError`, and `JITFunction.explain`. New native functions:
  `new_logging_queue`, `Queue.drain_logs`, `Queue.wait_log_sentinels`,
  `Stream.set_log_sentinel`, `Stream.mark_logging`, `capture_start`, and `capture_stop`.

### Manual GPU capture

With `MTL_CAPTURE_ENABLED=1`, capturing a vector add wrote an 84 KB `.gputrace` bundle
(`capture`, `index`, `metadata`, and resource files) in 0.12-0.16 s, and the kernel's
results were correct. Opening the trace in Xcode wasn't checked, because this session had
no GUI. Without the variable, `capture()` raises the explanatory error.

### Benchmarks

The following numbers come from `benchmarks/bench_dispatch.py`, three runs each, against
a build of the previous commit on the same machine:

| Launch, sustained | This build (min) | Previous build (min) |
|---|---|---|
| `@enceladus.jit` on `enceladus.Tensor` | 3.20-3.26 µs | 3.17-3.24 µs |
| `@enceladus.jit` on PyTorch MPS tensors | 5.07-5.23 µs | 4.91-5.23 µs |

Printing is slow by nature: Metal processed about 170,000 messages per second, and a
command buffer that printed 25,600 lines took 160-200 ms to complete.

### Tests

`tests/test_debug.py` adds 4 test functions (9 cases). The suite runs 653 tests, with 18
skipped, in 6.4-7.4 s.

- Device printing in interpreted, compiled, and PyTorch-fallback modes: an 8-element
  tile across 128 threads (16 copies of each element), a row sum (a slice layout held
  by 4 threads), and scalars. The test compares the multiset of `stderr` lines with the
  expected lines, so a duplicate, a missing line, a wrong program ID, or a wrong value
  fails it.
- Device asserts in both modes, with `ENCELADUS_DEBUG` on and off: a gather with an
  out-of-range index in program 1 raises with the assert's line and program ID, and a
  clean launch afterward doesn't raise again. With the flag off, the bad gather runs,
  and the compiled kernel has no error buffer and no assert code.
- `explain` on a kernel that adds a row-major tile to a column-major one and reduces: it
  reports one threadgroup exchange for `bt` at the line of the sum, the reduction's
  threadgroup bytes at that line, the source text, and the same peak as
  `warmup().threadgroup_memory_bytes`.
- `capture` without `MTL_CAPTURE_ENABLED` raises the explanatory error.

Each of these mutations fails at least one test: printing without the owner predicate,
not waiting for log sentinels, not resetting the error buffer, and leaving the debug flag
out of the in-memory specialization key.

### Deviations from the plan

- **Printing blocks capture for the whole process.** The plan says to document that
  printing and capture are incompatible. Metal refuses to start a capture ("Capturing
  Shader logging is not supported") once the process has created any command queue with
  a log state, even after the queue is released and even when the capture targets
  another queue. So `capture()` raises a clear error if a printing kernel has run, and a
  printing kernel that launches during a capture raises too.
- **Printed lines arrive at sync, not while the kernel runs.** Metal's log handler runs
  after the command buffer completes. The stream forwards lines at `synchronize`, which
  `Tensor.numpy()`, NumPy-argument launches, and `enceladus.synchronize()` all call.
- **`device_print` refuses `hex=True`**, and the prefix must be printable ASCII.
- **Asserts don't stop the kernel.** Metal has no way to abort a dispatch, and returning
  early would break barriers, so the kernel keeps running after a failed assert. The
  docstring says to guard the access with a mask too.
- **The interpreter follows `ENCELADUS_DEBUG` too**, so both modes behave the same.
- **`explain` doesn't create a Metal pipeline**, so it has no measured register count;
  the estimate is the codegen's own count of 32-bit registers per tile.

### Known gaps

- Metal drops log messages silently when a command buffer's messages overflow the 8 MB
  log buffer, at about 88 bytes per line. A command buffer that printed 196,608 lines
  delivered 95,324, and the sentinel still arrived, so no warning appeared. Print fewer
  elements, or sync more often.
- If a sentinel never arrives, for example after a command-buffer error, `synchronize`
  waits up to 10 s (0.1 s after an error) and then writes a warning line.
- A printing kernel that `timed_launch` runs right after other pending work can lose its
  sentinel to the earlier command buffer, so its lines might appear at a later sync.
  Only autotuning uses `timed_launch`.
- Once a kernel prints, the default stream stays on the logging queue for the rest of
  the process. Launches that don't print pay no sentinel cost there.
- `explain` lists anchored tiles only. Cheap values (constants, `arange`, and
  elementwise ops over them) are rematerialized at each use and aren't listed.
- The `// file.py:LINE` source comments that the plan's M2 section schedules for
  `ENCELADUS_DEBUG=1` aren't emitted.
