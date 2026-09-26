# Metal kernel ceilings on M4 Pro: measurements for the Forge MSL backend

This report measures what hand-written Metal Shading Language (MSL) kernels achieve on an
Apple M4 Pro (16-core GPU, 24 GB unified memory, macOS 27, Metal 4), and which codegen
patterns a Triton-like compiler (Forge) needs to emit to reach those numbers.

All kernels were compiled at runtime with `MTLDevice.newLibraryWithSource`, driven from
Python through PyObjC, timed with `MTLCommandBuffer` `GPUStartTime` and `GPUEndTime`, and
checked against NumPy. Code, kernels, and raw JSON results are in
`scratchpad/kernel_bench/`.

## Summary

- **Compute peak is about 6.5 TFLOPS, not 9.** 16 cores x 128 FP32 lanes x 2 FLOP x
  about 1.59 GHz gives 6.5 TFLOPS (the published figure for the 16-core M4 Pro GPU). Measured
  ceilings are 6.1 TFLOPS for scalar FP32 FMA, 6.2 TFLOPS for `simdgroup_matrix` FP32 MMA,
  6.3 TFLOPS for FP16 MMA, and 5.6 TFLOPS for BF16-accumulator MMA. FP16 runs at the same
  rate as FP32; there's no 2x FP16 rate on this GPU.
- **DRAM bandwidth ceiling is about 238 GB/s** (87% of the 273 GB/s spec). Every coalesced
  elementwise pattern reaches 230-238 GB/s. Vector width, elements per thread, threadgroup
  size, and masking barely matter.
- **Softmax (4096 x 4096) is memory bound** at 230-243 GB/s for both FP32 and FP16, which
  matches MLX (237 and 254 GB/s) and beats torch-mps (194 GB/s).
- **Matmul:** the best hand-written `simdgroup_matrix` kernel reaches 5.46 TFLOPS FP32 and
  5.82 TFLOPS FP16 and BF16 at 4096^3, which is 96-100% of MPS and 98-104% of MLX.
  Metal 4 Metal Performance Primitives (MPP) `matmul2d` compiles at runtime, runs on M4, and
  is the fastest option measured: 5.64 TFLOPS FP32 and 6.19 TFLOPS FP16 (103% of MPS).
- **Biggest codegen surprises:** loading `simdgroup_matrix` fragments straight from device
  memory beats staging through threadgroup memory by 7-15%, and some per-simdgroup tile
  shapes fall off a 10x performance cliff (FP32 32x32 per simdgroup), so tile configs
  must be autotuned and cannot be derived from a register-count model.

## Method and caveats

- **Timing:** GPU timestamps per command buffer, 3 warmup runs, then the minimum of 10-60
  runs. Short kernels (softmax, small matmuls) encode 20-50 back-to-back dispatches in one
  command buffer and report the per-dispatch time.
- **Shared GPU:** another benchmark process and the desktop (WindowServer, browsers) used
  the GPU during this session, with 20-99% background utilization. The harness polls
  `ioreg` "Device Utilization %" and waits for a quiet window before each measurement, and
  reports min-of-N; the median is stored alongside in the JSON. Early runs taken during
  contention showed 2-3x slowdowns and were discarded. Expect about +-3% noise on the
  numbers in this report.
- **Correctness:** every kernel is checked against NumPy (FP32 reference on the
  dtype-rounded inputs). Relative error is max|C - ref| / max|ref|: FP32 0 (bit-identical to
  Accelerate on these inputs), FP16 2.5e-4 to 3.9e-4, BF16 2e-3 to 3e-3. Softmax max
  absolute error is 2e-7 to 9e-7 (FP32) and 2.4e-4 (FP16).
- **GPU clock ramp:** a single short command buffer runs well below peak clocks. For
  example, one 512^3 FP16 matmul takes 261 us alone but 55 us per dispatch when 20 are
  batched (1.0 versus 4.8 TFLOPS). An empty command buffer costs about 5.6 us of GPU time.

## Hardware ceilings

The following table lists the measured compute ceilings (`peak.py`, `fma2.py`, `mma2.py`):

| Microbenchmark | dtype | Result | % of 6.5 TFLOPS spec |
|---|---|---|---|
| Scalar FMA, 32 independent chains per thread | fp32 | 6.11 TFLOPS | 94% |
| Vector FMA (`float4`/`half4`), 8 chains | fp16 | 6.25 TFLOPS | 96% |
| `simdgroup_multiply_accumulate` 8x8x8, 16 chains | fp32 | 6.21 TFLOPS | 96% |
| `simdgroup_multiply_accumulate` 8x8x8, 4-16 chains | fp16 | 6.20-6.33 TFLOPS | 95-97% |
| `simdgroup_multiply_accumulate` 8x8x8, 16 chains | bf16 (bf16 acc) | 5.59 TFLOPS | 86% |
| `exp()` throughput, `fastMathEnabled = true` | fp32 | 740 Gexp/s | n/a |
| `exp()` throughput, `fastMathEnabled = false` | fp32 | 412 Gexp/s | n/a |

`fastMathEnabled` changes `exp` max relative error from 1.6e-7 to 3.7e-6.

A first MMA microbenchmark that used `make_filled_simdgroup_matrix` constants reported
9.7 TFLOPS: the compiler partly folds MMAs on constant operands. Operands loaded from a
buffer give the honest 6.2 TFLOPS. Any Forge microbenchmark suite must load operands from
memory.

## Experiment 1: memory bandwidth

Each array is 256 MB (64M `float`); copy moves 512 MB and add moves 768 MB per call. In
the *coalesced* layout, a threadgroup owns `TG * EPT` vectors and thread `t` handles
`t + j * TG` (Triton-style block). In the *contig* layout, each thread owns `EPT`
consecutive vectors.

| Kernel | Layout | Vector | EPT | TG | Time | GB/s | % of 273 |
|---|---|---|---|---|---|---|---|
| copy | coalesced | float | 1 | 256 | 2.26 ms | 238 | 87% |
| copy | coalesced | float4 | 4 | 256 | 2.31 ms | 232 | 85% |
| copy | coalesced | float4 | 16 | 256 | 2.36 ms | 228 | 83% |
| add | coalesced | float | 1-16 | 256 | 3.40-3.46 ms | 233-237 | 85-87% |
| add | coalesced | float2 | 1-16 | 256 | 3.42-3.45 ms | 233-236 | 85-86% |
| add | coalesced | float4 | 1-16 | 256 | 3.40-3.48 ms | 231-237 | 85-87% |
| add | contig | float | 4 (16 B/thread) | 256 | 3.42 ms | 235 | 86% |
| add | contig | float | 8 (32 B/thread) | 256 | 3.93 ms | 205 | 75% |
| add | contig | float4 | 4 (64 B/thread) | 256 | 4.24 ms | 190 | 70% |
| add | contig | float4 | 8 (128 B/thread) | 256 | 4.59 ms | 175 | 64% |
| add | grid-stride, 64 TGs | float4 | loop | 256 | 3.43 ms | 235 | 86% |
| add | grid-stride, 1024 TGs | float4 | loop | 256 | 3.55 ms | 227 | 83% |

The following table shows the threadgroup-size sweep (add, coalesced):

| Vector, EPT | TG 64 | TG 128 | TG 256 | TG 512 | TG 1024 |
|---|---|---|---|---|---|
| float, 4 | 235 | 236 | 236 | 236 | 235 |
| float4, 1 | 235 | 235 | 235 | 235 | 234 |
| float4, 4 | 234 | 235 | 237 | 234 | 232 |

The following table shows working-set size effects (add, float4, 50 dispatches per
command buffer):

| Size per array | Time per dispatch | GB/s | Note |
|---|---|---|---|
| 1 MB | 18 us | 174 | Dispatch overhead dominates |
| 4 MB | 28 us | 458 | 12 MB working set stays in the system-level cache |
| 16 MB | 210 us | 239 | DRAM |
| 256 MB | 3.44 ms | 234 | DRAM |

## Experiment 2: row softmax (4096 x 4096)

GB/s counts one read and one write of the matrix (128 MB FP32, 64 MB FP16). *Tree* is a
threadgroup-memory tree reduction (log2(TG) barriers). *Simd* is `simd_max`/`simd_sum`
followed by one threadgroup-memory exchange of `TG/32` partials and a second `simd_*`
reduction (two barriers per reduction). *Registers* keeps the row slice in registers
(one read); *reread* reads the row again for each pass.

| Variant | TG | fp32 time | fp32 GB/s | fp16 time | fp16 GB/s |
|---|---|---|---|---|---|
| Tree, registers | 128 | 0.586 ms | 229 | 0.282 ms | 238 |
| Tree, registers | 256 | 0.556 ms | 241 | 0.282 ms | 238 |
| Tree, registers | 512 | 0.580 ms | 231 | 0.284 ms | 236 |
| Tree, registers | 1024 | 0.588 ms | 228 | 0.404 ms | 166 |
| Simd, registers | 32 (1 simdgroup per row) | 0.616 ms | 218 | 0.313 ms | 215 |
| Simd, registers | 128 | 0.588 ms | 228 | 0.277 ms | 242 |
| Simd, registers | 256 | 0.586 ms | 229 | 0.280 ms | 239 |
| Simd, registers | 512 | 0.588 ms | 228 | 0.276 ms | 243 |
| Simd, registers | 1024 | 0.583 ms | 230 | 0.280 ms | 240 |
| Simd, reread | 256 | 0.592 ms | 227 | 0.282 ms | 238 |
| Simd, registers, fast math off | 256 | 0.588 ms | 228 | 0.281 ms | 239 |
| Tree, registers, fast math off | 1024 | 0.587 ms | 229 | 0.426 ms | 158 |
| MLX `mx.softmax` (wall clock) | n/a | 0.567 ms | 237 | 0.264 ms | 254 |
| torch-mps `torch.softmax` (wall clock) | n/a | 0.691 ms | 194 | 0.346 ms | 194 |

Findings:

- Softmax at this shape is bandwidth bound. Reduction style, re-reading, and fast math
  change results by less than the noise, except in the following two cases.
- The tree reduction loses about 30% at TG = 1024 in FP16, where each thread holds only
  one `half4` and 10 barrier rounds dominate. The simd reduction stays flat across
  TG = 128-1024, so it's the safer default.
- One simdgroup per 4096-wide row (TG = 32) loses about 10%.
- Fast math matters only when `exp` is the bottleneck: 1.8x `exp` throughput in the
  compute-bound microbenchmark.

## Experiment 3: matmul

The following table lists TFLOPS for square NN row-major GEMM at 2048^3 and 4096^3. The
last three columns use the 4096^3 result. MPS is `MPSMatrixMultiplication` timed with GPU
timestamps; MLX (`mx.matmul`) and torch-mps are wall clock over 5 calls, so they include
launch overhead (under 1% at these sizes).

| Kernel | Config | dtype | 2048^3 | 4096^3 | % of 6.5 peak | % of MPS | % of MLX |
|---|---|---|---|---|---|---|---|
| (a) naive | 1 thread per output, 16x16 TG | fp32 | 0.53 | 0.52 | 8% | 10% | 10% |
| (b) tiled threadgroup memory | 64x64x16, 4x4 outputs per thread | fp32 | 2.30 | 2.17 | 33% | 40% | 41% |
| (c) simdgroup, threadgroup-staged | 32x64x16, 2x2 simdgroups | fp32 | 4.76 | 4.68 | 72% | 86% | 89% |
| (c) simdgroup, direct loads (`best_matmul.metal`) | 64x64, 4x1 simdgroups, 16x64 per simdgroup | fp32 | 5.39 | 5.46 | 84% | 100% | 104% |
| (d) MPP `matmul2d` | 64x64, 4 simdgroups | fp32 | 5.63 | 5.64 | 87% | 103% | 107% |
| MPS | n/a | fp32 | 5.52 | 5.46 | 84% | 100% | 104% |
| MLX | n/a | fp32 | 5.08 | 5.26 | 81% | 96% | 100% |
| torch-mps | n/a | fp32 | 5.36 | 5.43 | 84% | 99% | 103% |
| (a) naive | 1 thread per output | fp16 | 0.55 | 0.55 | 8% | 9% | 9% |
| (b) tiled threadgroup memory | 64x64x16, 4x4 per thread | fp16 | 2.40 | 2.35 | 36% | 39% | 39% |
| (c) simdgroup, threadgroup-staged | 64x64x16, 2x2 simdgroups | fp16 | 5.57 | 5.43 | 84% | 90% | 91% |
| (c) simdgroup, direct loads (`best_matmul.metal`) | 64x64, 4x1 simdgroups | fp16 | 5.80 | 5.82 | 90% | 97% | 98% |
| (d) MPP `matmul2d` | 64x64, 4 simdgroups | fp16 | 6.15 | 6.19 | 95% | 103% | 104% |
| MPS | n/a | fp16 | 6.07 | 6.01 | 92% | 100% | 101% |
| MLX | n/a | fp16 | 5.70 | 5.95 | 92% | 99% | 100% |
| torch-mps | n/a | fp16 | 5.90 | 6.00 | 92% | 100% | 101% |
| (c) simdgroup, threadgroup-staged | 64x64x16, 4x1 simdgroups | bf16 | 5.61 | 5.45 | 84% | n/a | 92% |
| (c) simdgroup, direct loads (`best_matmul.metal`) | 64x64, 4x1 simdgroups | bf16 | 5.77 | 5.82 | 90% | n/a | 98% |
| (d) MPP `matmul2d` | 64x64, 4 simdgroups | bf16 | 6.15 | 6.06 | 93% | n/a | 102% |
| MLX | n/a | bf16 | 5.74 | 5.95 | 92% | n/a | 100% |
| torch-mps | n/a | bf16 | 5.92 | 5.99 | 92% | n/a | 101% |

All simdgroup kernels accumulate in FP32. BF16 compiles and runs at runtime, both as
`simdgroup_bfloat8x8` operands with a `simdgroup_float8x8` accumulator and in MPP.

### Tile-shape sweep and the register cliff

The following table shows the 2048^3 sweep. *Staged* stages A and B tiles through
threadgroup memory with 16-byte row padding. *Direct* calls `simdgroup_load` from device
memory with no threadgroup memory and no barriers.

| BM x BN x BK, simdgroups | Per-simdgroup tile (fragments) | fp32 staged | fp32 direct | fp16 staged | fp16 direct |
|---|---|---|---|---|---|
| 32x32x16, 2x2 | 16x16 (2x2) | 4.33 | 4.02 | 4.95 | n/m |
| 32x64x16, 2x2 | 16x32 (2x4) | **4.75** | 5.22 | 5.17 | 5.56 |
| 64x32x16, 2x2 | 32x16 (4x2) | 4.26 | 4.71 | 5.17 | 5.14 |
| 64x64x16, 4x2 | 16x32 (2x4) | 4.49 | 5.21 | 5.16 | n/m |
| 64x64x16, 4x4 | 16x16, 512 threads | 3.30 | n/m | 3.26 | n/m |
| 64x64x16, 2x2 | 32x32 (4x4) | **0.49** | **0.76** | **5.55** | 5.80 |
| 128x64x16, 4x2 | 32x32 (4x4) | 0.52 | n/m | 5.23 | 5.78 |
| 128x128x16, 4x4 | 32x32, 512 threads | 0.71 | n/m | 3.82 | 5.80 |
| 64x64x16, 4x1 | 16x64 (2x8) | n/m | **5.36** | n/m | **5.90** |
| 32x128x16, 2x1 | 16x128 (2x16) | n/m | **0.16** | n/m | 5.79 |
| 64x64x16, 1x2 | 64x32 (8x4) | 0.32 | 0.31 | **0.37** | 0.35 |

"n/m" means not measured. Findings:

- There is a 7-15x cliff that depends on per-simdgroup tile shape and dtype, not on a simple
  register count. FP32 collapses at 4x4 fragments (16 accumulators) but runs at 5.36 TFLOPS
  with 2x8 fragments (also 16 accumulators, and more operand fragments). FP16 with an FP32
  accumulator handles 4x4, 2x8, and 2x16, but collapses at 8x4. The behavior matches
  register spilling. `[[max_total_threads_per_threadgroup(N)]]` and
  `maxTotalThreadsPerThreadgroup` did not change it and don't reveal it.
- Direct device loads beat threadgroup staging on M4 by 7-15% for every good shape. The
  GPU caches supply the reuse, and removing two barriers per K step helps more than the
  explicit staging.

### Other matmul knobs

The following table lists the knobs that were tested and their measured effect:

| Knob | Result |
|---|---|
| Threadgroup swizzle (log2 1-3), 4096^3 | No effect for direct loads (5.18 in all cases); staged FP32 drops 4.68 to 4.11 |
| Threadgroup-memory padding 0 versus 16 bytes | No difference (FP32 4.78 versus 4.76, FP16 5.55 versus 5.57) |
| Register double-buffering of next K fragments (`PREFETCH`) | 1-7% slower |
| Streaming B fragments one at a time (`STREAM`) | No effect on good shapes; doesn't fix the cliff |
| FP16 accumulator (`simdgroup_half8x8` acc) | At most 2% faster for the same tile (5.68 versus 5.56 TFLOPS) and below the best FP32-accumulator config (5.90), with about 60x larger error (1.6e-2 versus 2.6e-4). It spills at 32x32 per simdgroup (0.85 TFLOPS) |
| `[[max_total_threads_per_threadgroup]]` | No measurable effect |
| BK (unroll depth) 8, 16, or 32 | Within 2% for good shapes |

### Metal Performance Primitives `matmul2d` (Metal 4)

MPP works on M4 when compiled at runtime. It's the fastest GEMM measured, and it handles
arbitrary sizes internally.

| Config | dtype | 2048^3 | 4096^3 | 2000^3 | 2047^3 |
|---|---|---|---|---|---|
| 64x64, 4 simdgroups, dynamic K | fp32 | 5.63 | 5.64 | 5.33 | 5.24 |
| 64x32, 4 simdgroups | fp32 | 4.43 | 4.28 | n/m | n/m |
| 128x128, 8 simdgroups | fp32 | 5.40 | n/m | n/m | n/m |
| 64x64, 4 simdgroups, dynamic K | fp16 | 6.15 | 6.19 | 5.81 | 6.13 |
| 64x64, 4 simdgroups, dynamic K | bf16 | 6.15 | 6.06 | 5.72 | 6.13 |
| 64x64, manual K loop (KT = 64), cooperative accumulator | fp16 | 6.07 | n/m | n/m | n/m |
| 64x64, manual K loop (KT = 32), cooperative accumulator | fp16 | 6.01 | 6.05 | n/m | n/m |
| 64x64, manual K loop (KT = 32), cooperative accumulator | fp32 | 5.36 | 5.43 | n/m | n/m |
| 64x64, fp16 inputs, fp32 output | fp16 to fp32 | 6.15 | n/m | n/m | n/m |

`relaxed_precision = true` made no difference (6.15 TFLOPS FP16). The M4 GPU doesn't have
the neural accelerators that MPP targets on later chips, so the fast path here is
presumably a well-tuned `simdgroup_matrix` lowering. Treat it as an opaque library call.

The following list describes what it takes to use MPP from a runtime-compiled shader
(`kernels/matmul_mpp.metal`):

- Set `MTLCompileOptions.languageVersion = MTLLanguageVersion4_0`, then add
  `#include <metal_tensor>` and
  `#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>`. The headers ship in
  `/System/Library/Frameworks/MetalPerformancePrimitives.framework`, and the offline
  `metal` compiler isn't needed.
- Build tensors in the shader from ordinary buffer pointers with
  `tensor<device half, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(K, M));`.
  This avoids `MTLTensor` objects and MTL4 argument tables on the host. The classic
  `MTLComputeCommandEncoder` exposed through PyObjC has no `setTensor` method.
- Extents are innermost first: a row-major M x K matrix has extents `(K, M)`, and
  `slice(col, row)` takes column then row.
- Don't define single-letter preprocessor macros such as `T`. They collide with template
  parameter names inside `metal_tensor` and produce unrelated-looking parse errors.
- The default descriptor mode is `multiply`, which overwrites the destination. A manual K
  loop needs `matmul2d_descriptor::mode::multiply_accumulate`.
- The header's own example code doesn't compile as written: `get_mask(i)` is
  `is_valid_element(i)` in this SDK, `#pragma unroll full` is invalid, and a
  `cooperative_tensor` can only `store` to a tensor of the same element type. For a
  fused epilogue or a dtype cast, walk `get_capacity()` elements, then use
  `get_multidimensional_index(i)`, which returns (column, row) within the tile, and write
  to device memory yourself.

## Experiment 4: bounds checking, fast math, and threadgroup size

### Masked loads in matmul

*MASK 0* has no bounds checks, *MASK 1* checks every load and store, and *MASK 2* checks
only edge tiles (a uniform per-threadgroup branch) plus the K tail. Values are TFLOPS.

| Kernel | dtype | 2048 MASK 0 | 2048 MASK 1 | 2048 MASK 2 | 2000 MASK 1 | 2000 MASK 2 | 2047 MASK 2 |
|---|---|---|---|---|---|---|---|
| Staged, 32x64x16 | fp32 | 4.76 | 4.43 (-7%) | 4.48 (-6%) | 4.24 | 4.25 | n/a (needs K % 4 = 0) |
| Direct, 32x64x16 | fp32 | 5.22 | 2.87 (-45%) | 5.15 (-1%) | 2.75 | 4.75 | 4.36 |
| Staged, 64x64x16 | fp16 | 5.57 | 5.47 (-2%) | 5.54 (-1%) | 5.15 | 5.22 | n/a |
| Direct, 64x64x16 | fp16 | 5.80 | 3.28 (-43%) | 5.78 (0%) | 3.12 | 5.06 | 5.35 |
| `best_matmul.metal` (per-fragment masks in edge tiles) | fp32 | 5.39 | n/a | n/a | n/a | 4.72 | 4.65 |
| `best_matmul.metal` | fp16 | 5.80 | n/a | n/a | n/a | 5.02 | 5.13 |
| MPP `matmul2d` (internal) | fp16 | 6.15 | n/a | n/a | n/a | 5.81 | 6.13 |

For elementwise kernels, `if (i < n)` masking has no measurable cost: add with float4 and
EPT 4 measures 234 GB/s unmasked and 236 GB/s masked.

### Fast math

`fastMathEnabled` has no effect on the bandwidth-bound softmax (within 2%), and it gives
1.8x `exp` throughput when `exp` is the bottleneck, at 3.7e-6 max relative error.

### Threadgroup size for elementwise kernels

TG = 64, 128, 256, 512, and 1024 are all within 2% (232-237 GB/s). For reductions, TG = 1024
with a tree reduction and little data per thread is the one bad case.

## Codegen lessons for Forge

### Elementwise and 1D blocks

- Map a `BLOCK_SIZE` program to one threadgroup of 128-256 threads. Give each thread
  `VEC` contiguous elements (4 x 32-bit, or 8 x 16-bit, which is 16 bytes), interleaved
  across threads, and loop `BLOCK_SIZE / (TG * VEC)` times with stride `TG * VEC`. This
  is Triton's blocked layout, and it measures 230-238 GB/s.
- Avoid per-thread contiguous chunks larger than 16 bytes per access: 32 bytes costs 13%,
  and 128 bytes costs 25%.
- Emit `if (idx < n)` masks unconditionally for elementwise kernels, because they're free.
  No specialization for the unmasked case is needed.
- Don't autotune threadgroup size for elementwise kernels; pick 256.

### Reductions and softmax

- Lower `tl.max` and `tl.sum` over a block to `simd_max`/`simd_sum`, then one
  threadgroup-memory exchange of `TG / 32` partials, then a second `simd_*` pass. It's flat
  across TG = 128-1024, whereas a tree reduction degrades at 1024.
- Keep the row slice in registers when it fits (up to about 16 `float4` per thread). Re-reading
  is nearly as fast because of the caches, so an online or looped fallback for long rows
  is cheap.
- Accumulate in FP32 for FP16 inputs. Use vector loads (`half4` or `float4`).
- Default to fast math on for softmax-style kernels. It matters only when compute-bound,
  and 3.7e-6 relative `exp` error is fine for ML workloads.

### Matmul (`tl.dot`)

- **Prefer MPP `matmul2d` when the device supports Metal 4.** It's 3-6% faster than the
  best hand-written kernel, matches or beats MPS, handles ragged edges internally, supports
  BF16, FP8, and int formats, and offers a cooperative-tensor accumulator for fused
  epilogues. A `tl.dot` inside a K loop maps to `op.run(sliceA, sliceB, coop_acc)` with
  `multiply_accumulate` mode.
- **Emit the `simdgroup_matrix` template as the fallback** and for older OS versions. The
  template has the following properties:
  - Fragments load straight from device memory with `simdgroup_load(frag, ptr, ld)`, with
    no threadgroup memory and no barriers.
  - The FP32 accumulator uses `simdgroup_float8x8`, with FP16 or BF16 operand fragments
    (mixed-precision `simdgroup_multiply_accumulate` compiles and runs at full rate).
  - The default config is a 64x64 threadgroup tile with 4 simdgroups stacked along M, each
    computing a 16x64 strip (2x8 fragments), and K stepped by 8.
  - Masking is split into an interior fast path and an edge path, selected by a uniform
    per-threadgroup branch. In edge tiles, only fragments that straddle M or N use masked,
    per-lane loads through `thread_elements()`.
  - The epilogue writes each lane's two elements as a `vec<T, 2>` store; that's where
    fused bias, activation, and casts go.
- **Autotune a small config set and verify each config.** There are 10x cliffs, so a
  heuristic can silently pick a spilling config. A reasonable search space is per-simdgroup
  tiles 16x32, 16x64, 32x16, and 32x32 (FP16 and BF16 only), with 4-8 simdgroups.
- **Never always-mask the direct-load path.** Per-lane masked loads cost about 45%. Checking
  only edge tiles costs 0-1% on aligned sizes and 5-15% on ragged ones.
- **Don't bother with** threadgroup swizzling, threadgroup-memory padding, register
  double-buffering, or FP16 accumulators on this GPU.
- **Pad leading dimensions for small problems.** Odd leading dimensions and many edge tiles
  hurt small problems: 513^3 is 2.4x slower than 512^3 even when batched. Consider padding
  allocations to multiples of 8 or 16, or pick a smaller tile when there are few
  threadgroups (M = 100, N = 3000 runs at 1.1 TFLOPS).

### Runtime

- **Batch dispatches.** Launch and clock-ramp latency dominates small kernels: one
  isolated 512^3 matmul runs at 1.0 TFLOPS, and 20 in one command buffer run at 4.8 TFLOPS.
  The Forge runtime needs to batch many dispatches per command buffer and avoid
  `waitUntilCompleted` per kernel.
- **Load benchmark operands from memory.** Constant-filled MMA operands get partly folded
  by the compiler and report 1.6x the real peak.

## Best matmul MSL source

The following kernel is `kernel_bench/kernels/best_matmul.metal`, the reference template
for a `tl.dot` lowering. It's verified for FP32, FP16, and BF16 at 2048, 4096, 2000,
2047, 1000, and 513 (square) and at 4096x1024x4096, 1024x4096x1000, 8192x8192x1024, and
100x3000x777. It measures 5.39 and 5.46 TFLOPS for FP32 and 5.80 and 5.82 TFLOPS for FP16
at 2048^3 and 4096^3. Dispatch it with `threadgroups = (ceil(N/64), ceil(M/64), 1)` and
`threadsPerThreadgroup = (128, 1, 1)`.

```metal
// Best hand-written simdgroup_matrix GEMM measured on M4 Pro (16-core GPU).
//   C[M,N] = A[M,K] @ B[K,N], row-major, fp32 accumulation, T in {float, half, bfloat}.
//   Threadgroup tile 64x64, 4 simdgroups stacked along M (each owns a 16x64 strip = 2x8
//   fragments of 8x8), K stepped by 8, fragments loaded straight from device memory.
//   Interior tiles take an unmasked fast path; in edge tiles only the fragments that straddle
//   M/N use masked loads; the K % 8 tail is masked.
// Measured: fp32 5.4 TFLOPS, fp16/bf16 5.9 TFLOPS at 2048^3 and 4096^3
// (~96-98% of MPSMatrixMultiplication, ~95% of MPP matmul2d).
//
// This is the reference shape of what a Forge `tl.dot` lowering should emit. Each constant
// below is a compile-time specialization a compiler would bake in.
#include <metal_stdlib>
using namespace metal;

template <typename T, int BM, int BN, int WM, int WN>
struct GemmCfg {
  static constexpr constant int SM = BM / WM, SN = BN / WN;   // per-simdgroup C tile
  static constexpr constant int TM = SM / 8, TN = SN / 8;     // 8x8 fragments per simdgroup
  static constexpr constant int NT = WM * WN * 32;            // threads per threadgroup
};

// Masked 8x8 fragment load: every lane fetches the 2 elements it owns; OOB -> 0.
template <typename T>
inline void load_frag_masked(thread simdgroup_matrix<T, 8, 8>& f, device const T* p, uint ld,
                             int rows_left, int cols_left, uint fm, uint fn) {
  thread auto& e = f.thread_elements();
  const bool rok = int(fm) < rows_left;
  e[0] = (rok && int(fn) < cols_left) ? p[fm * ld + fn] : T(0);
  e[1] = (rok && int(fn) + 1 < cols_left) ? p[fm * ld + fn + 1] : T(0);
}

template <typename T, int BM, int BN, int WM, int WN>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]]
void gemm(device const T* A [[buffer(0)]], device const T* B [[buffer(1)]],
          device T* C [[buffer(2)]], constant uint3& MNK [[buffer(3)]],
          uint2 tgid [[threadgroup_position_in_grid]],
          uint sg [[simdgroup_index_in_threadgroup]],
          uint lane [[thread_index_in_simdgroup]]) {
  using Cfg = GemmCfg<T, BM, BN, WM, WN>;
  constexpr int TM = Cfg::TM, TN = Cfg::TN;
  const uint M = MNK.x, N = MNK.y, K = MNK.z;

  // --- program-id -> tile origin (Forge: tl.program_id(0/1)) ---
  const uint row0 = tgid.y * BM, col0 = tgid.x * BN;
  const uint sr = (sg / WN) * Cfg::SM, sc = (sg % WN) * Cfg::SN;

  // --- lane -> (row, col) of its two elements in an 8x8 fragment ---
  const uint qid = lane / 4;
  const uint fm = (qid & 4) + ((lane / 2) % 4);
  const uint fn = (qid & 2) * 2 + (lane % 2) * 2;

  // --- accumulator = tl.zeros((BM, BN), tl.float32), distributed over simdgroups ---
  simdgroup_matrix<float, 8, 8> acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  device const T* Ap = A + (row0 + sr) * K;   // this simdgroup's A strip
  device const T* Bp = B + col0 + sc;         // this simdgroup's B strip

  // --- uniform (per-threadgroup) interior/edge split ---
  const bool edge = (row0 + BM > M) || (col0 + BN > N);
  const uint kmain = (K / 8) * 8;

  if (!edge) {
    // --- fast path: for k in range(0, K, 8): acc += tl.dot(a_frag, b_frag) ---
    for (uint k = 0; k < kmain; k += 8) {
      simdgroup_matrix<T, 8, 8> a[TM], b[TN];
      #pragma unroll
      for (int i = 0; i < TM; ++i) simdgroup_load(a[i], Ap + i * 8 * K + k, K);
      #pragma unroll
      for (int j = 0; j < TN; ++j) simdgroup_load(b[j], Bp + k * N + j * 8, N);
      #pragma unroll
      for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
  } else {
    // --- edge tile: only fragments that straddle M/N use masked loads; the per-fragment
    //     test is uniform across the simdgroup, so there is no divergence ---
    for (uint k = 0; k < kmain; k += 8) {
      simdgroup_matrix<T, 8, 8> a[TM], b[TN];
      #pragma unroll
      for (int i = 0; i < TM; ++i) {
        const int rows_left = int(M) - int(row0 + sr + i * 8);
        if (rows_left >= 8) simdgroup_load(a[i], Ap + i * 8 * K + k, K);
        else load_frag_masked(a[i], Ap + i * 8 * K + k, K, rows_left, 8, fm, fn);
      }
      #pragma unroll
      for (int j = 0; j < TN; ++j) {
        const int cols_left = int(N) - int(col0 + sc + j * 8);
        if (cols_left >= 8) simdgroup_load(b[j], Bp + k * N + j * 8, N);
        else load_frag_masked(b[j], Bp + k * N + j * 8, N, 8, cols_left, fm, fn);
      }
      #pragma unroll
      for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
  }
  // --- K tail (K % 8 != 0): masked, zero-filled fragment loads ---
  for (uint k = kmain; k < K; k += 8) {
    simdgroup_matrix<T, 8, 8> a[TM], b[TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
      load_frag_masked(a[i], Ap + i * 8 * K + k, K, int(M) - int(row0 + sr + i * 8), int(K - k), fm, fn);
    #pragma unroll
    for (int j = 0; j < TN; ++j)
      load_frag_masked(b[j], Bp + k * N + j * 8, N, int(K - k), int(N) - int(col0 + sc + j * 8), fm, fn);
    #pragma unroll
    for (int i = 0; i < TM; ++i)
      #pragma unroll
      for (int j = 0; j < TN; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
  }

  // --- epilogue: registers -> C (fused bias/activation/cast would go here) ---
  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      const uint r = row0 + sr + i * 8 + fm;
      const uint c = col0 + sc + j * 8 + fn;
      thread auto& e = acc[i][j].thread_elements();
      if (!edge) {
        *(device vec<T, 2>*)(C + r * N + c) = vec<T, 2>(T(e[0]), T(e[1]));
      } else if (r < M) {
        if (c < N) C[r * N + c] = T(e[0]);
        if (c + 1 < N) C[r * N + c + 1] = T(e[1]);
      }
    }
  }
}

// Explicit specializations (Forge would emit one per (dtype, config) it autotunes).
#define INST(T, NAME)                                                                          \
  template [[host_name(NAME)]] [[kernel]] void gemm<T, 64, 64, 4, 1>(                          \
      device const T*, device const T*, device T*, constant uint3&, uint2, uint, uint);
INST(float, "gemm_f32")
INST(half, "gemm_f16")
INST(bfloat, "gemm_bf16")
```

The MPP equivalent (`kernel_bench/kernels/matmul_mpp.metal`) has the following core:

```metal
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void mpp_mm(device half* A [[buffer(0)]], device half* B [[buffer(1)]],
                   device half* C [[buffer(2)]], constant uint3& MNK [[buffer(3)]],
                   uint2 tgid [[threadgroup_position_in_grid]]) {
  const int M = MNK.x, N = MNK.y, K = MNK.z;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(K, M));
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tB(B, dextents<int32_t, 2>(N, K));
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(N, M));
  constexpr auto desc = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent));
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA = tA.slice(0, tgid.y * 64);
  auto mB = tB.slice(tgid.x * 64, 0);
  auto mC = tC.slice(tgid.x * 64, tgid.y * 64);
  op.run(mA, mB, mC);   // grid (ceil(N/64), ceil(M/64)), 128 threads per threadgroup
}
```

## Files

The following files are in
`/private/tmp/claude-501/-Users-kashifulhaque-Documents-test-forge/76fe23fb-ffeb-469d-b877-8176e503b51f/scratchpad/kernel_bench/`:

- `mtl.py`: PyObjC harness (runtime compile, shared buffers, GPU timing, wait-for-idle).
- `bw.py`, `softmax.py`, `softmax_baselines.py`: experiments 1, 2, and 4.
- `peak.py`, `fma2.py`, `mma2.py`: compute ceilings. The MMA numbers in `peak.py` are the
  folded ones; use `mma2.py`.
- `matmul.py`: naive, tiled, and simdgroup sweeps (phases `simple`, `sweep`, `knobs`, `mask`,
  `bf16`, and `final`). `t7.py`, `t9.py`, `t10.py`, `t11.py`, and `t15.py` are follow-up
  sweeps.
- `mpp.py`, `t12.py`, `t14.py`: MPP `matmul2d`.
- `baselines.py`: MPS, MLX, and torch-mps GEMM.
- `best.py`: verification and benchmark of `best_matmul.metal`.
- `kernels/best_matmul.metal`, `kernels/matmul_simdgroup.metal` (macro-parameterized
  sweep kernel), `kernels/matmul_mpp.metal`, `kernels/matmul_simple.metal`.
- `results_*.json`: raw results, with min and median times.

Source for the 6.5 TFLOPS specification figure:
[Apple M4 Pro 16-core GPU FP32 performance](https://www.cpu-monkey.com/en/igpu-apple_m4_pro_16_core).
