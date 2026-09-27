# Apple GPU architecture and Metal for the Forge kernel compiler

This report covers the Apple GPU hardware, Metal Shading Language (MSL), and toolchain facts that a Triton-like DSL needs in order to generate fast kernels for M-series GPUs. Sources are cited inline. Facts marked **[measured]** come from microbenchmarks run on the dev machine (M4 Pro, 16-core GPU, macOS 27, `applegpu_g16s`) in this session. The benchmark sources are in `scratchpad/bench/` (`bench.swift`, `bw.swift`, `fma.swift`, `gemm.swift`, `tgm.swift`, `comp.swift`, `dbg.swift`, `async.swift`, and `api.swift`). Facts marked **[uncertain]** are inferred or come from reverse engineering.

## 1. Microarchitecture

### GPU families

The following table maps chips to Metal GPU families. Source: [Metal Feature Set Tables (May 2026)](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf).

| Chip | Metal family | Notes |
|---|---|---|
| M1 series, A14 | Apple7 | SIMD-group matrix, SIMD reductions, and float atomics start here. The Metal 4 API is available from Apple7. |
| M2 series, A15, A16 | Apple8 | Only `ulong` atomic min and max (macOS only). SIMD shift-and-fill. |
| M3 and M4 series, A17 Pro, A18 | Apple9 | Dynamic caching, 64-bit atomics (only `ulong` min and max compile; see [Atomics](#atomics)), hardware BF16 (as MFA's code treats it). |
| M5 series, A19 | Apple10 | Per-core *neural accelerators*, which TensorOps (MPP) use. |

The M4 Pro on the dev machine reports `supportsFamily(.apple9) == true`, `.apple10 == false`, `.metal4 == true`, and architecture name `applegpu_g16s` **[measured]**. MLX gates its neural-accelerator (NAX) path on `get_architecture_gen() >= 17` and macOS 26.2 ([mlx `device.cpp`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/device.cpp)), so M4 (g16) never takes that path.

### Hardware numbers

The following table lists the hardware numbers that matter for code generation.

| Quantity | Value | Source |
|---|---|---|
| SIMD width (`threadExecutionWidth`) | 32 | Feature Set Tables; **[measured]** |
| Max threads per threadgroup | 1024 (per pipeline: check `maxTotalThreadsPerThreadgroup`) | Feature Set Tables; **[measured]** |
| Max threadgroup memory per threadgroup | 32 KB (32768 B), 16 B length alignment | Feature Set Tables; **[measured]** `maxThreadgroupMemoryLength` |
| Buffer argument table entries | 31 per kernel (the same number for `MTL4ArgumentTable`) | Feature Set Tables |
| `setBytes` inline data | 4 KB | Feature Set Tables |
| Function constants | 65,536 per function | Feature Set Tables |
| Max buffer length (M4 Pro, 24 GB) | 14.3 GB; `recommendedMaxWorkingSetSize` is 19.1 GB | **[measured]** |
| ALUs per core | 128 (four schedulers, each issuing one 32-wide instruction per cycle) | [metal-benchmarks](https://github.com/philipturner/metal-benchmarks) |
| FP32 and FP16 FMA per core per cycle | 128 lanes (256 FLOP) for both | metal-benchmarks |
| INT32 multiply per core per cycle | 32 (a quarter of the FMA rate) | metal-benchmarks; **[measured]** approximately 34 |
| Integer divide | No hardware instruction. It's a multi-instruction sequence. Philip Turner reports roughly 6 cycles of throughput for `DIV32`. | metal-benchmarks |
| GPRs | Up to 128 32-bit registers per thread, addressable as 16-bit halves; 256 uniform registers per SIMD-group | [Dougall Johnson G13 ISA docs](https://dougallj.github.io/applegpu/docs.html) |
| Register file per core (M1 and M2) | Approximately 208 KB; 384 to 3072 threads per core | metal-benchmarks |
| Threadgroup memory per core (M1 and M2) | Approximately 60 KB physical | metal-benchmarks |
| L1D, L1I (M1 and M2) | 8 KB and 12 KB; 128 B global cache line | metal-benchmarks; [Chips and Cheese M2 Pro](https://chipsandcheese.com/p/a-brief-look-at-apples-m2-pro-igpu) |
| L2 | M2 Pro: 3 MB, more than 1 TB/s | Chips and Cheese |
| SLC and DRAM latency (M2 Pro) | 234 ns and more than 342 ns | Chips and Cheese |
| SIMD shuffle bandwidth | 256 B per cycle per core (double NVIDIA's) | metal-benchmarks |
| `SHUFFLE_ROTATE32` latency | Approximately 5.4 cycles | metal-benchmarks |
| FADD, FMUL, and FFMA latency | Approximately 2.2 cycles (F32) and 2.16 cycles (F16) | metal-benchmarks |

The following table lists bandwidth and throughput by chip.

| Chip | GPU cores | DRAM bandwidth | FP32 peak |
|---|---|---|---|
| M4 | 10 | 120 GB/s | Approximately 3.6 TFLOPS (MFA README lists 3580 GFLOPS) |
| M4 Pro | 16 or 20 | 273 GB/s | 16-core: **[measured]** 5.8 TFLOPS with an FMA loop and 6.1 TFLOPS with MPP GEMM. The theoretical peak is approximately 6.4 TFLOPS at about 1.57 GHz **[uncertain clock]**. |
| M4 Max | 32 or 40 | 410 or 546 GB/s | Approximately 18 TFLOPS on 40 cores **[uncertain]** |
| M5 | 10 | 153 GB/s | Adds neural accelerators |

For M4 Pro and M4 Max bandwidth, see [Apple Newsroom](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/). For M4 and M5 bandwidth, see [Apple ML Research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5).

The following bandwidth results were measured on the M4 Pro **[measured]**:

- A streaming `float4` read reaches 257-263 GB/s, which is 95% of 273 GB/s.
- A `float4` copy reaches 224-238 GB/s of read plus write traffic.
- An uncoalesced gather (stride of 33 floats) reaches 27.6 GB/s effective, about 8x worse than streaming.
- The first cold run of the copy benchmark reported only 109 GB/s, and the first FMA run was lower too. The GPU needs tens of milliseconds of load before it reaches full clocks. Forge's autotuner must warm up before timing and take the minimum of N runs.

### Dynamic caching (M3 and later)

The [Explore GPU advancements in M3 and A17 Pro](https://developer.apple.com/videos/play/tech-talks/111375/) tech talk describes three changes in Apple9:

- **Register file as a cache.** On Apple7 and Apple8, a SIMD-group reserved its peak register count for its entire lifetime, so peak register pressure dictated occupancy. On Apple9, registers are allocated and freed dynamically, and "the maximum register usage no longer dictates how many SIMDgroups can be run."
- **Flexible on-chip memory.** Registers, threadgroup memory, tile memory, stack, and buffer data share fewer, larger on-chip caches. Unused threadgroup memory isn't wasted. The shader core monitors behavior and throttles occupancy to avoid spilling.
- **More parallel FP16, FP32, and integer issue.** Apple claims "up to 2x ALU performance" from mixed instruction types, and recommends FP16 wherever possible because conversions are free and FP16 uses fewer registers.

Consequences for code generation:

- On M3 and M4, direct device-to-register loads work well. MFA switches to `blockDimensions = (32, 32, 8)`, one SIMD-group per threadgroup, and `preferAsyncLoad = false` on Apple9 ([`GEMMDescriptor.swift`](https://github.com/philipturner/metal-flash-attention/blob/main/Sources/FlashAttention/GEMM/GEMMDescriptor/GEMMDescriptor.swift)). On M1 and M2, it uses 48x48 tiles, 2x2 SIMD-groups, threadgroup staging, and async copies.
- A register cliff still exists. In the GEMM benchmark, raising a SIMD-group's accumulator tile from 32x32 (16 `simdgroup_float8x8`) to 32x64 (32 fragments) dropped throughput from 5.4 to 0.36 TFLOPS, a 15x collapse **[measured]**. Forge needs a register-pressure model and must reject configurations that are too large.
- Philip Turner notes that ALU utilization saturates at about 24 SIMD-groups per core on M1-class GPUs. Beyond that point, occupancy gains nothing.

## 2. MSL features a compiler would emit

All references in this section are to the [MSL specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) (2026-06-04 edition, 383 pages). The `-std=metal4.0` flag targets macOS 26, and `metal4.1` targets macOS 27. From Swift, `MTLLanguageVersion.version4_0` compiles on this machine, and `MTLLanguageVersion(rawValue: 0x40001)` (4.1) also compiles **[measured]**. The [Rigel paper](https://arxiv.org/html/2606.12765v1) reports that a 4.1 binary doesn't load on macOS 26.5.

### Types and address spaces

- Scalar types: `half`, `float`, `bfloat` (MSL 3.1 and later, with the literal suffix `bf`), `char`, `short`, `int`, `long`, and `uint64_t` (MSL 2.2 and later). There's no `double`. Vectors go up to 4 components (`half4`, `float4`), plus `packed_*` variants.
- Address spaces: `device` (read-write global), `constant` (read-only, cached, suited to uniforms and `setBytes`), `threadgroup`, and `thread`. Pointer kernel arguments must carry an address space.
- `bfloat` is **excluded** from the `simd_shuffle*` and `simd_sum` type set (spec, Table 6.14). Forge must bitcast to `ushort` to shuffle it.

### Kernel signature and thread attributes

The following code shows the kernel signature pattern Forge would emit:

```metal
[[kernel, max_total_threads_per_threadgroup(128)]]
void k(device const half* A [[buffer(0)]],
       device float*      C [[buffer(1)]],
       constant Params&   p [[buffer(2)]],   // setBytes, <= 4 KB
       uint3  tgid  [[threadgroup_position_in_grid]],
       uint3  tid   [[thread_position_in_threadgroup]],
       uint3  gid   [[thread_position_in_grid]],
       ushort lane  [[thread_index_in_simdgroup]],
       ushort sgid  [[simdgroup_index_in_threadgroup]]);
```

The `max_total_threads_per_threadgroup` attribute, or `MTLCompileOptions.maxTotalThreadsPerThreadgroup`, lets the backend compiler allocate more registers per thread. Setting 256 in compile options is reflected in `pso.maxTotalThreadsPerThreadgroup == 256` **[measured]**.

### SIMD-group matrix: the Metal 3 matmul primitive

`simdgroup_matrix<T,8,8>` supports `T` values of `half`, `float`, and `bfloat` (MSL 3.1 and later). The element-to-lane mapping is "unspecified" (spec 2.4). MLX's `BaseMMAFrag` hard-codes the layout: each lane holds 2 elements (1 row by 2 adjacent columns), with coordinates computed as follows ([`steel/gemm/mma.h`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/mma.h)):

```metal
const short qid = lane / 4;
const short fm  = (qid & 4) + ((lane / 2) % 4);        // row
const short fn  = (qid & 2) * 2 + (lane % 2) * 2;      // col (2 consecutive)
```

MLX accesses fragments through `mat.thread_elements()` (a `vec<T,2>`), which lets it run elementwise epilogues and masked loads in registers. The following code shows the core API:

```metal
simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float,8,8>(0.f);
simdgroup_half8x8 a, b;
simdgroup_load(a, A + row*K + k, /*elements_per_row=*/K);    // device or threadgroup
simdgroup_load(b, B + k*N + col, N /*, ulong2 origin, bool transpose */);
simdgroup_multiply_accumulate(acc, a, b, acc);               // half x half -> float OK
simdgroup_store(acc, C + row*N + col, N);
```

Mixed `half` inputs with a `float` accumulator compile and give exact results **[measured]**. Calls must occur in SIMD-uniform control flow. The spec calls for tensors and MPP instead of `simdgroup_matrix` (spec 6.8).

### SIMD-group functions

The SIMD-group functions are `simd_shuffle`, `simd_shuffle_xor`, `simd_shuffle_up`, `simd_shuffle_down`, `simd_shuffle_rotate_{up,down}`, `simd_shuffle_and_fill_{up,down}` (with `modulo` values of 2 to 32), `simd_broadcast`, `simd_ballot`, `simd_all`, `simd_any`, `simd_sum`, `simd_product`, `simd_min`, `simd_max`, `simd_and`, `simd_or`, `simd_xor`, and `simd_prefix_{inclusive,exclusive}_{sum,product}`. Quad variants (`quad_*`) also exist. Reductions require Apple7.

### Barriers

`threadgroup_barrier(mem_flags)` and `simdgroup_barrier(mem_flags)` accept these flags: `mem_none`, `mem_device`, `mem_threadgroup`, and `mem_texture`. MSL 4.1 adds `memory_order` and `thread_scope` parameters. On Apple silicon, "a thread that has ended no longer participates" in barriers, so early return before a barrier is legal there (spec 6.10.1).

### Atomics

`atomic_int`, `atomic_uint`, `atomic_bool`, `atomic_ulong` (MSL 2.4 and later), and `atomic_float` (MSL 3 and later, **device memory only**). Apple9 is required for full 64-bit atomics. **[corrected]** On this machine (macOS 27, MSL 3.2 to 4.1), the only 64-bit atomics that compile are `atomic_max_explicit` and `atomic_min_explicit` on `device atomic_ulong`, which return `void`. There's no `atomic_long`, and no 64-bit fetch, exchange, load, store, or compare-and-swap, even through the `__metal_atomic_*` builtins. Orderings are relaxed in older versions. MSL 3.2 adds `thread_scope` and `seq_cst` fences, and MSL 4.1 adds real acquire, release, and seq_cst on atomic operations. MFA's README states that `atomic<float>` "is emulated" (a CAS loop) **[uncertain on M4]**, so avoid float atomics in hot paths such as split-K and dQ accumulation. GPUCompiler's Metal target lowers what AIR lacks (8-bit and 16-bit atomics, `nand`, float min and max) to CAS loops, and notes that device memory is coherent only within a threadgroup without acquire semantics ([GPUCompiler PR #942](https://github.com/JuliaGPU/GPUCompiler.jl/pull/942)).

### Function constants

Function constants are specialization knobs resolved at `newFunctionWithName:constantValues:`. They can gate arguments, which MLX uses for alignment and batch variants ([`steel_gemm_fused.h`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h)):

```metal
constant bool align_M [[function_constant(200)]];
constant bool has_batch [[function_constant(10)]];
const constant int* batch_shape [[buffer(6), function_constant(has_batch)]],
```

Specializing and building a pipeline from an already-parsed library took about 25 ms, including archiving **[measured]**. For Forge, generating distinct source text per specialization is simpler and roughly as fast, because the front end is the cheap part (see section 6).

### Math modes

`-fmetal-math-mode=fast|relaxed|safe` and `-fmetal-math-fp32-functions=fast|precise` map to `MTLCompileOptions.mathMode` and `.mathFloatingPointFunctions`, which are available from macOS 15. `fastMathEnabled` is the legacy flag. **The default is fast**, which assumes no NaN or Inf. This setting breaks `-INFINITY` masking in softmax unless Forge sets `relaxed` (which honors Inf and NaN) or uses `#pragma METAL fp math_mode(safe)`. Contraction into FMA is on by default.

### Argument buffers

Tier 2 is available on all Apple7 and later GPUs, with no limit on buffers reachable through an argument buffer. Metal 3 bindless (`gpuAddress`) lets a kernel take a `device T*` stored inside a struct. Forge doesn't need argument buffers until it exceeds 31 bindings.

## 3. Metal 4 and MSL 4 tensors

### Host API

The Metal 4 host API includes the following types, per [Discover Metal 4 (WWDC25 205)](https://developer.apple.com/videos/play/wwdc2025/205/):

- Queues and command buffers: `MTL4CommandQueue`, `MTL4CommandBuffer` (created from the device), and `MTL4CommandAllocator`.
- Bindings: `MTL4ArgumentTable` (`setAddress:` and `setResource:`).
- Residency sets, which are mandatory: resources must be added to a residency set.
- Encoders: `MTL4ComputeCommandEncoder`, which unifies compute, blit, and acceleration-structure work. Synchronization uses stage-to-stage `barrierAfterStages:beforeQueueStages:visibilityOptions:`.
- Compilation: `MTL4Compiler`, which is separate from the device (`device.makeCompiler(descriptor:)` returns `AGXG16XFamilyCompiler` here **[measured]**), inherits caller QoS, and supports async compile tasks. Pipeline dataset serialization and harvesting cover offline caching.
- Machine learning: `MTL4MachineLearningCommandEncoder` runs whole Core ML networks packaged with `metal-package-builder`.

Metal 4 is supported on M1 and later.

### MTLTensor

`MTLTensorDescriptor` exposes `dimensions` (`MTLTensorExtents`), `strides` (the innermost stride must be 1), `dataType`, and `usage` (`.compute`, `.render`, or `.machineLearning`). You create a tensor from the device or from an `MTLBuffer`. macOS 27 adds FP8 (E4M3, E5M2), FP4 E2M1, and int2 types, plus auxiliary *scales* planes (MXFP8 with UE8M0 block scales). For details, see the [WWDC26 330 session](https://developer.apple.com/videos/play/wwdc2026/330/) and the Feature Set Tables "Tensor limits" table. That table specifies alignment rules, for example a row stride that's a multiple of 64 B divided by the element size for ML usage, and a 128 B buffer offset for sub-byte types.

### MSL tensor, cooperative_tensor, and MPP TensorOps

The following types and operations are defined in spec 2.22 and chapter 7:

- `tensor<device half, dextents<int,2>[, tensor_handle | tensor_inline]>` is a tensor type. A `tensor_handle` binds an `MTLTensor` at `[[buffer(n)]]`. A **`tensor_inline`** tensor wraps a raw `device` pointer with extents and optional strides. Extents are ordered innermost first, so a row-major MxK matrix has extents `(K, M)`.
- `.slice(x0, y0)` produces a tile view.
- `cooperative_tensor<T, Extents, Layout>` is a register fragment spread across the threads of an execution scope, with a device-specific, opaque layout. You get one from an op, not by constructing it. It supports `begin()`, `end()`, `map_iterator`, `load()`, `store()`, and `reduce_rows` and `reduce_columns` with `reduction_operation::max` and `sum`.
- `mpp::tensor_ops::matmul2d<desc, Scope>` and `convolution2d` are the TensorOps. Scope is `execution_thread`, `execution_simdgroup`, or `execution_simdgroups<N>`, where N must be 1 or the full `simdgroups_per_threadgroup`.
- The descriptor constructor is `matmul2d_descriptor(M, N, K = dynamic_length_v<int>, transpose_left, transpose_right, relaxed_precision, mode::multiply | mode::multiply_accumulate)`.
- Supported type combinations: Metal 4 covers `char`, `half`, and `float` combinations. OS 26.1 adds `bfloat`, OS 26.4 adds `uchar` and int4, and Metal 4.1 adds int2, FP8, and FP4.
- OS 26.1 adds `get_left_input_cooperative_tensor`, `is_compatible_as_left_input`, and column reductions. These let the output of QK^T feed the P·V multiply without a threadgroup round trip (the FlashAttention pattern in WWDC26 330).

The spec spells `dynamic_length_v<int>` without a namespace, but it resolves only as **`tensor_ops::dynamic_length_v<int>`**. `int(dynamic_extent)` and `0` also compile **[measured]**. The WWDC snippet without the namespace fails to compile.

The following kernel was verified on this machine. It runs on a plain Metal 3 `MTLComputeCommandEncoder` with ordinary `MTLBuffer` values, without `MTLTensor` or argument tables **[measured]**:

```metal
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal; using namespace mpp;
kernel void mpp_gemm(device half* A [[buffer(0)]], device half* B [[buffer(1)]],
                     device float* C [[buffer(2)]], constant uint3& dims [[buffer(3)]],
                     uint2 tgid [[threadgroup_position_in_grid]]) {
  int M = dims.x, N = dims.y, K = dims.z;
  tensor<device half,  dextents<int,2>, tensor_inline> tA(A, dextents<int,2>(K, M));
  tensor<device half,  dextents<int,2>, tensor_inline> tB(B, dextents<int,2>(N, K));
  tensor<device float, dextents<int,2>, tensor_inline> tC(C, dextents<int,2>(N, M));
  constexpr auto desc = tensor_ops::matmul2d_descriptor(64, 64,
                           tensor_ops::dynamic_length_v<int>);
  tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;   // 128 threads
  auto mA = tA.slice(0, tgid.y * 64);
  auto mB = tB.slice(tgid.x * 64, 0);
  auto mC = tC.slice(tgid.x * 64, tgid.y * 64);
  auto ct = op.get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
  op.run(mA, mB, ct);      // epilogue on ct elements goes here
  ct.store(mC);
}
```

### Hardware acceleration by chip

On M5 and A19 (Apple10), TensorOps dispatch to the per-core neural accelerators. MLX reports up to 4x faster time to first token on M5 compared with M4 ([Apple ML Research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)). The same source runs on all Apple silicon.

On M4, `matmul2d` lowers to the ordinary SIMD-group matrix and ALU path. The [Rigel paper (arXiv 2606.12765)](https://arxiv.org/abs/2606.12765) reaches this conclusion for M4 Max from three findings:

- An FP16 ceiling of 14.8 TFLOPS.
- Only a 1.05-1.21x advantage over hand-written `simdgroup_matrix` code.
- FP8 running at 0.87-0.94x of FP16, which indicates emulation.

Rigel also finds an accumulator of at least FP32 and an 8x8 base fragment layout in which each lane holds 2 vertically adjacent elements per 8x8 tile.

The following table shows GEMM throughput on the M4 Pro 16-core GPU for FP16 x FP16 to FP32, taking the best of 22 warm runs **[measured]**.

| Kernel | 2048³ | 4096³ |
|---|---|---|
| `simdgroup_matrix`, direct device loads, 2x2 SIMD-groups, TG tile 64x64 (32x32 per SIMD-group) | 5.51 TFLOPS | 5.35 TFLOPS |
| The same kernel with a 32x64 TG tile | 5.09 TFLOPS | 5.08 TFLOPS |
| The same kernel with a 64x128 TG tile (register spill) | 0.35 TFLOPS | 0.35 TFLOPS |
| MPP `matmul2d` 64x64, `execution_simdgroups<4>`, device C | 6.13 TFLOPS | 6.02 TFLOPS |
| MPP 64x64 with a cooperative-tensor destination | 6.14 TFLOPS | 6.03 TFLOPS |
| MPP 128x128 | 5.76 TFLOPS | 3.81 TFLOPS |

MPP reaches approximately 6.1 TFLOPS, which exceeds the FMA-loop result of 5.8 TFLOPS and is about 95% of the approximately 6.4 TFLOPS theoretical peak **[uncertain clock]**. Per core, 6.1 / 16 = 0.38 TFLOPS, which matches Rigel's 14.8 / 40 = 0.37 TFLOPS on M4 Max. A naive 30-line `simdgroup_matrix` kernel gets within 10-12% of MPP. The per-core match between two measurements suggests this is the real ceiling on Apple9.

What MPP offers a DSL:

- Near-peak GEMM tiles without a hand-tuned inner loop.
- Automatic neural accelerator use on M5 and later.
- Quantized and FP8 operand support with in-op dequantization.
- Register-resident epilogues through cooperative tensors.

What it costs:

- The fragment layout is opaque. Element access goes through iterators, not known lane coordinates.
- Scope N is restricted to 1 or the whole threadgroup.
- It requires MSL 4.0 and macOS 26 or later.
- Compiles are slow: approximately 100 ms front end plus approximately 90 ms pipeline build per MPP kernel, compared with approximately 3 ms plus 3 ms for a simple kernel **[measured]**.

## 4. Fast matmul and attention on Apple GPUs

### MLX steel GEMM

For MLX steel GEMM, see [`matmul.cpp`](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/matmul.cpp) and [`kernels/steel`](https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal/kernels/steel). The architecture suffix of `applegpu_gNNx` selects parameters: `g` and `p` are base or phone chips, `s` and `c` are medium chips, and `d` is Ultra **[uncertain mapping]**. The following table lists the tile configurations.

| Device class | Case | BM×BN×BK, WM×WN |
|---|---|---|
| Small (`g`, `p`) | NT (A normal, B transposed) | 64×32×32, 2×2 |
| Small (`g`, `p`) | half or bf16 | 64×64×16, 1×2 |
| Small (`g`, `p`) | complex64 | 64×32×8, 4×1 |
| Medium (`s`, including M4 Pro) | default | **64×64×16, 2×2** (128 threads) |
| Large (`d`) | half, large, reasonable K | 64×64×16, 1×2 |
| Large (`d`) | NN with large K | 32×64×16, 1×2 |
| Large (`d`) | float NN | 64×32×32, 2×2 |
| NAX (M5 and later) | any | 128×128×512 with 4×4 SIMD-groups (Max: BM=64, BK=64 or 256) |

The techniques are as follows:

- Threadgroup staging with padding of `16 / sizeof(T)` elements per row (`tgp_padding_a`).
- Two barriers per K-step, with a single buffer.
- Each SIMD-group owns TM×TN 8x8 fragments, where TM = BM / (8·WM).
- A tile swizzle (`swizzle_log`) remaps `tgid` for L2 locality.
- Function constants for `align_M`, `align_N`, and `align_K`. Aligned kernels use `load_unsafe`, and ragged edges use `load_safe` with zero fill.
- Split-K when (M/16)·(N/16) ≤ 1024 or 2048 and K ≥ max(M, N), accumulated by a second kernel rather than by atomics.
- A GEMV path for M = 1.

MLX attention (`steel_attention.metal`) instantiates BQ=32, BK=16 or 32, BD=64-256, WM=4, and WN=1, or WN=2 for D=256. `sdpa_vector` and `sdpa_vector_2pass` handle decoding with a query length of 8 or less.

### metal-flash-attention (MFA)

For MFA, see the [repository](https://github.com/philipturner/metal-flash-attention). It generates MSL source strings at runtime in Swift and JIT-compiles them, which is the same approach Forge is taking. Its design choices:

- It blocks along D as a third dimension and uses deliberate, controlled register spilling for large head dimensions.
- It warps block aspect ratios to 16-32 along the parallelized dimension and 80-128 along the traversed dimension.
- A parameter file maps D to the operands that stay in registers.
- It avoids FP32 atomics with a separate dQ kernel and a separate dK and dV kernel (7 GEMMs instead of 5).

Reported ALU utilization for the FP16 forward pass on M3 and M4 is 94%, 91%, and 82% for D = 64, 128, and 256. For forward plus backward, it's 71%, 69%, and 61%. M1 reaches 83-86% forward. GEMM on M1 and M2 uses 48×48×(24 or 32) blocks, 2x2 SIMD-groups, and async copies. On Apple9, it uses 32×32×8 blocks, a single SIMD-group, and direct loads.

The async copy uses the undocumented `simdgroup_async_copy`, which MFA declares with `__asm("air.simdgroup_async_copy_2d.p3i8.p1i8")`. On macOS 27, `newLibraryWithSource` rejects this with **"illegal string literal in 'asm'"**, even inside `#pragma METAL internals : enable` **[measured]**. Forge must not depend on it. The compiler suggested a `__metal_wait_wg_events` builtin, which suggests internal threadgroup-event intrinsics exist **[uncertain, undocumented]**.

### Typical percentage of peak

Well-tuned kernels reach 80-95% of FP16 or FP32 FMA peak on large GEMMs, and 60-94% on attention. Measured on M4 Pro: MPP reaches about 95%, and a simple `simdgroup_matrix` kernel reaches about 85%.

## 5. Performance pitfalls

- **Threadgroup memory access patterns.** Metal-benchmarks leaves the bank count as "TBD." In the microbenchmark (256 threads, each reading `tg[(tid·stride) & 8191]`), time relative to stride 1 was as follows **[measured]**:
  - Stride 2: 0.97x.
  - Stride 4: 1.63x.
  - Stride 8: 1.88x.
  - Stride 16: 3.0x.
  - Stride 32: 5.1x.
  - Stride 64: 5.35x.
  - Odd strides 17 and 33: 1.75x and 1.87x.

  Power-of-two strides of 16 or more are pathological. Pad rows the way MLX (16 bytes) and MFA (for example, 24 → 28 for FP32) do. Contiguous, lane-linear access is best. Philip Turner notes that Apple deliberately provisions low threadgroup memory bandwidth and invests in shuffles and matrix instructions instead, so prefer `simd_shuffle` and register tiles over threadgroup memory round trips.
- **Coalescing.** Lanes reading consecutive 4-16 B elements hit about 95% of DRAM bandwidth. A strided gather lost about 8x **[measured]**. Vectorize loads to `float4` or `half8`-equivalents where alignment allows.
- **Register spilling.** A spill cliff can cost 15x **[measured]**. Compute the live fragment count. Each `simdgroup_matrix<float,8,8>` costs 2 registers per lane, and threads have up to 128 GPRs. Prefer `half` for operands, because 16-bit values halve register cost and reduce dependency latency at low occupancy (metal-benchmarks: 1.56 cycles against 1.84 cycles for dependent FMUL).
- **Barriers.** In a latency-bound ping-pong loop, `threadgroup_barrier(mem_threadgroup)` cost the same as `simdgroup_barrier`, and TG=32 and TG=1024 performed the same **[measured, narrow test]**. Barriers aren't expensive in themselves. The real cost is the pipeline stall they create when every SIMD-group waits for loads. Use a double buffer so each K-step needs one barrier.
- **Integer division and modulo.** Measured throughput is about 106 G/s against 5,800 GFLOP/s for FMA, a 50x gap in operation rate **[measured]**. IMUL runs at a quarter of the FMA rate. Forge must strength-reduce `/` and `%` by constants and powers of two, hoist index math out of loops, and use `ushort` or `short` for lane and tile indices, as MLX does.
- **Half compared with float.** FP16 FMA throughput equals FP32 (6.15 against 5.81 TFLOPS **[measured]**). FP16's benefits are registers, bandwidth, and latency, not peak FLOPS. Accumulate in FP32.
- **Bounds checks.** Use function-constant or template-specialized "aligned" variants with no checks, plus a checked variant for edge tiles, as MLX does. `simdgroup_load` has no clamping, so edge tiles need manual masked element loads through `thread_elements()`.
- **Fast math.** The default fast math mode breaks `-inf` and NaN semantics (see section 2).

## 6. Compilation pipeline

`newLibraryWithSource:options:error:` runs the Clang-based front end, which turns MSL into AIR (LLVM bitcode wrapped in a `.metallib` container). `newComputePipelineStateWithFunction:` runs the backend (`AGXCompilerCore` in `MTLCompilerService`), which turns AIR into a GPU binary. The latency measurements on this machine **[measured]**:

- The first compile in a process takes about 31 ms in the front end and about 16 ms in the pipeline build, mostly one-time initialization.
- A new simple kernel after that takes about 3.6 ms front end and 3.4 ms pipeline.
- An MPP matmul kernel takes about 100 ms front end and about 90 ms pipeline.
- Identical source in the same process returns from a cache in 0 ms. A source that differs only in a comment also hit the system cache (about 2 ms).

tinygrad calls `libMTLCompiler.dylib` directly (request type 13) and cuts compile time from about 27 ms to 7.7 ms ([tinygrad PR 7842](https://github.com/tinygrad/tinygrad/pull/7842), later merged as #7920). This approach relies on private API.

`MTLCompileOptions` fields include `languageVersion`, `mathMode`, `mathFloatingPointFunctions`, `fastMathEnabled` (legacy), `preserveInvariance`, `optimizationLevel`, `maxTotalThreadsPerThreadgroup`, `preprocessorMacros` (useful for emitting tile sizes as `-D` macros), `enableLogging`, `libraryType`, and `installName`. All were verified to compile in Swift **[measured]**.

**Binary archives.** The following calls work together:

- `device.makeBinaryArchive(descriptor:)`
- `addComputePipelineFunctions(descriptor:)`
- `serialize(to:)`
- On the next run, `MTLBinaryArchiveDescriptor.url` plus `pipelineDescriptor.binaryArchives` with `.failOnBinaryArchiveMiss`

Loading a pipeline from the archive took 0.07 ms, and the archive was 13 KB for one kernel **[measured]**. Forge can use this as its persistent autotune cache, keyed by source hash, OS build, and GPU architecture. Metal 4 replaces this with `MTL4Archive` and pipeline dataset serialization. `MTLDynamicLibrary` (`libraryType = .dynamic`) is available for shared device helpers, but Forge doesn't need it.

**Emitting AIR directly.** Apple doesn't publicly document AIR. Existing projects that emit it:

- **Metal.jl and GPUCompiler.jl** go from Julia to LLVM IR, target the `air64-apple-macosx` triple, and emit AIR intrinsics such as `@air.convert.f.f32.s.i64`. A downgrader (`LLVMDowngrader_jll`, derived from @a2flo's libfloor) rewrites the bitcode to a format Apple's reader accepts and restores typed pointers, because AIR predates opaque pointers. Metal.jl writes the reverse-engineered `.metallib` container itself (after MetalLibraryArchive and libfloor) and loads it with `newLibraryWithData:`. Thread-position intrinsics become kernel arguments. `@device_code_air` shows the output. For details, see the [JuliaGPU blog](https://juliagpu.org/post/2022-06-24-metal/) and [zigpp issue #11](https://github.com/mattneel/zigpp/issues/11), which quotes accepted bitcode from the LLVM 5, 7, 14, 15, and 18.1 releases.
- **metal-ir-pipeline** (imperatormk) is a 26-pass LLVM-to-AIR pipeline for Triton IR ([DeepWiki](https://deepwiki.com/imperatormk/metal-ir-pipeline)). Its author later abandoned the AIR route: [triton-ext PR #126](https://github.com/triton-lang/triton-ext/pull/126) lowers TTGIR to **MSL source**, with `tl.dot` lowered to 8x8 `simdgroup_matrix`, citing the lack of public AIR support.

Emitting MSL is the lower-risk path. It gives access to MPP, which can't be expressed in hand-written AIR without replicating Apple's header internals.

## 7. Debugging and profiling

- **Timing.** `MTLCommandBuffer.gpuStartTime` and `gpuEndTime` (seconds, `CFTimeInterval`) work. For finer granularity, `MTLCounterSampleBuffer` with the `timestamp` counter set works. On M4 Pro, `supportsCounterSampling(.atStageBoundary)` is true, and `.atDispatchBoundary` and `.atDrawBoundary` are false. The only counter set is `timestamp:GPUTimestamp` **[measured]**. For per-kernel timing, put each dispatch in its own encoder or command buffer. Metal 4 adds `MTL4CounterHeap`.
- **Frame capture from the command line.** Create an `MTLCaptureDescriptor` with `destination = .gpuTraceDocument` and `outputURL = *.gputrace`, and call `MTLCaptureManager.shared().startCapture(with:)`. This requires **`MTL_CAPTURE_ENABLED=1`**; without it, the call fails with "Capture layer is not inserted." Capture produced a 132 KB `.gputrace` that Xcode opens. Capture fails if the queue uses shader logging ("Capturing Shader logging is not supported") **[measured]**. For a timeline, use `xcrun xctrace record --template 'Metal System Trace' --launch -- CMD`.
- **Shader printf.** Compile with `#include <metal_logging>` and `os_log_default.log("x=%f", v)`, set `MTLCompileOptions.enableLogging = true` and `languageVersion` to 3.2 or later, create a `device.makeLogState(descriptor:)` with an `addLogHandler` callback, and assign it to `MTLCommandQueueDescriptor.logState`. This printed on this machine **[measured]**. Offer it as a Forge debug mode.
- **Validation.** `MTL_DEBUG_LAYER=1` enables API validation. `MTL_SHADER_VALIDATION=1`, or per-pipeline `pipelineDescriptor.shaderValidation = .enabled`, enables GPU validation. In a quick test, an out-of-bounds store (256 B buffer, index +64 floats or +100,000 floats) was **not** reported in `commandBuffer.error`, even with `errorOptions = .encoderExecutionStatus` **[measured]**. Reports might go only to Xcode or the system log. Don't rely on validation as Forge's bounds checker. Also set `MTLCommandBufferDescriptor.errorOptions = .encoderExecutionStatus` so the command buffer reports GPU faults and timeouts.

## Implications for Forge's codegen

1. **Emit MSL text and compile it with `newLibraryWithSource`, not AIR.** This path is the documented one. MFA and triton-ext converged on it. It's fast (about 3-7 ms per simple kernel), and it's the only practical route to MPP. Pass tile sizes as literals or `preprocessorMacros`. Cache pipelines in memory by source hash and on disk through `MTLBinaryArchive`, keyed by OS build and `device.architecture.name`.
2. **Use two matmul backends behind one `tl.dot`:**
   - *MPP path* (MSL 4.0, macOS 26 and later): `tensor_inline` over raw buffers plus `matmul2d<desc, execution_simdgroups<4>>` into a `cooperative_tensor`, with epilogues through iterators. This path measured fastest on M4 (about 95% of peak) and picks up M5 neural accelerators with no code change. The trade-offs are slow compiles (about 200 ms) and opaque layouts.
   - *`simdgroup_matrix` path* (MSL 3.1 and later): explicit 8x8 fragments with MLX's known lane mapping. This path supports arbitrary fused epilogues, shuffles, and FlashAttention softmax in registers, at about 85-90% of peak.
3. **Start autotuning on M3 and M4 with these defaults:** 128 threads (2x2 SIMD-groups), BM=BN=64, BK=16 or 32, a 32x32 accumulator per SIMD-group (16 `float8x8` fragments), FP16 or BF16 operands, FP32 accumulation, and direct device loads. Also try the MLX-style threadgroup-staged variant with 16 B row padding and a single barrier per double-buffered step. Include MFA's 32×32×8 single-SIMD-group configuration. Treat more than about 16-24 live float fragments per SIMD-group as a hard pruning bound, given the 15x spill cliff.
4. **Tune the IR and passes for Apple hardware.** Strength-reduce all integer division and modulo. Use 16-bit indices. Hoist address math. Vectorize global loads. Pad threadgroup arrays to avoid strides that are powers of two of 16 or more. Prefer `simd_shuffle_xor` and `simd_sum`/`simd_max` reductions over threadgroup memory, which is the scarcer resource. Bitcast `bfloat` to `ushort` for shuffles.
5. **Handle masks and edges with specialization.** Generate aligned (unchecked) and edge (masked, zero-fill) variants selected by the launcher, following MLX's `align_M`, `align_N`, and `align_K` pattern. Default to `mathMode = .relaxed`, or scope `safe` pragmas, whenever the kernel uses `-inf` or NaN.
6. **Keep `setBytes` and launch within Metal limits.** Put scalar arguments in one `constant` struct through `setBytes` (4 KB or less), and use at most 31 buffers. Split-K reduces through a second kernel, not `atomic_float`, which is device-memory-only and possibly emulated.
7. **Build benchmarking hygiene into the autotuner.** Warm up the GPU for about 50 ms or more, time with `gpuStartTime` and `gpuEndTime`, take the minimum of N runs, and report percentage of roofline using approximately 6.4 TFLOPS FP32 and 273 GB/s for this M4 Pro.
8. **Build in debug hooks.** Offer an `FORGE_DEBUG_PRINT` mode that uses MSL `os_log` (incompatible with capture), an `FORGE_CAPTURE` mode that sets `MTL_CAPTURE_ENABLED=1` and writes `.gputrace` files, and optional `MTL_DEBUG_LAYER` support. Don't rely on shader validation to catch out-of-bounds access. Generate explicit debug-mode bounds asserts instead.
9. **Avoid these dead ends:** undocumented `simdgroup_async_copy` (rejected by the macOS 27 compiler), hand-rolled AIR (undocumented, and it requires a bitcode downgrader and a metallib writer), and depending on M5-only speedups when tuning on M4.
