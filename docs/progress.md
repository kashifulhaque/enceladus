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
