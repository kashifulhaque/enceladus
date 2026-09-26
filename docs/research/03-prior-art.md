# Forge prior-art and competitive survey: GPU kernel DSLs for Apple silicon

Snapshot date: 2026-09-26. Star counts and last-push dates come from the GitHub API on
that date. PyPI dates come from the PyPI JSON API. Numbers marked **measured locally**
were taken on an Apple M4 Pro running macOS 27.0 with PyTorch 2.11.

## Executive answer

**Yes. Tile-based, Triton-like DSLs for Apple GPUs exist and are in active development in
2026.** Forge's premise ("MPS or hand-written Metal are your only options") no longer
holds. Three efforts come close to Forge's goal, and one of them is mature:

1. **TileLang** (tile-ai/tilelang, 7.5k stars, pushed 2026-09-25). This is a Python tile
   DSL on TVM. Metal is a supported target that ships in release wheels and has CI.
   `T.gemm` lowers to `simdgroup_matrix` on M1-M5 and to Metal 4 cooperative tensors on
   M5. This project is the strongest direct competitor.
2. **Triton on Metal.** Two tracks exist:
   - `triton-lang/triton-ext` `backend/AppleGPU` is an out-of-tree plugin under the
     official Triton org. Its minimal version, which supports elementwise operations
     and masked loads and stores, merged on 2026-09-14. A fuller PR with 8x8 simdgroup
     MMA encoding is still open.
   - `bledden/triton-msl` (21 stars, `pip install triton-msl` 0.3.0, 2026-09-16) is a
     single-author alpha. It passes 5,560 of 9,342 upstream `test_core` cases with zero
     failures.
3. **Mojo (Modular).** Mojo has shipped Apple GPU support since 25.6 (September 2025). As
   of MAX 26.4 (June 2026), it can serve LLMs on M1-M5. Mojo uses a SIMT and
   layout-tensor model, not a Python eDSL.

Adjacent competitors include Metal.jl 1.10 (Julia, native simdgroup and `tensor_ops`
GEMMs), warp-metal (NVIDIA Warp with tiles on Metal, community project), CubeCL (Rust),
and Apple's own `coreai-torch` `TorchMetalKernel`, which uses a raw MSL body in the same
style as MLX.

Forge's remaining niche is narrower: an **Apple-first**, **lightweight**,
**framework-neutral** tile DSL. Its semantics follow Apple hardware facts, not CUDA
facts. It needs no LLVM, TVM, or Triton source build, and it interoperates natively with
both PyTorch MPS and MLX. For details, see
[Positioning and lessons for Forge](#positioning-and-lessons-for-forge).

## 1. Triton on Apple and Metal backends

### Upstream demand

- [triton-lang/triton#4824](https://github.com/triton-lang/triton/issues/4824), "Adding
  Metal Backend to Triton," opened 2024-09-28 and is still open. It has 16 comments, and
  most of them are "+1." No maintainer has committed to an in-tree backend.
- [Discussion #1796, "MPS backend support"](https://github.com/triton-lang/triton/discussions/1796)
  dates from 2023-06 and has 16 upvotes.
- Forks announced in the issue:
  - [ben594/triton-metal](https://github.com/ben594/triton-metal) lowers
    TTGIR → AIR (LLVM bitcode) → metallib and ran the first five tutorials. It has been
    inactive since 2026-05.
  - [mocusez/triton-metal](https://github.com/mocusez/triton-metal) lowers
    TTGIR → Metal dialect → MSL. It has 3 stars and was pushed 2026-09-09.
  - Both build on [NicolaLancellotti/metal-dialect](https://github.com/NicolaLancellotti/metal-dialect),
    an MLIR Metal dialect.

### triton-ext AppleGPU (official org, out-of-tree)

- The repository is [triton-lang/triton-ext](https://github.com/triton-lang/triton-ext)
  (38 stars, pushed 2026-09-25). The author is `imperatormk`.
- PR history:
  - #45 and #48 (2026-03) lowered to **AIR via LLVM**. They were closed because
    "that path relies on AIR internals Apple does not publicly document/support."
  - [#127](https://github.com/triton-lang/triton-ext/pull/127), minimal, **merged
    2026-09-14**.
  - [#126](https://github.com/triton-lang/triton-ext/pull/126), full, **open**, about
    54k added lines.
  - #133, open 2026-09-19, lowers `expand_dims` and `broadcast`.
- Architecture: TTIR → TTGIR → an `EmitMSL` MLIR pass → MSL text → `xcrun metal` →
  metallib. PR #126 adds `AccelerateAppleMatmul` (`tt.dot` → `AppleMmaEncoding`, 8x8
  simdgroup MMA) and a `StoreShuffleLayout` pass. It also adds `agpu`, an MLIR-free
  emitter library with an MSL AST and 1,187 CMake unit tests.
- Merged scope covers `program_id`, `make_range`, `splat`, `addptr`, masked
  `load`/`store`, constants, and elementwise and comparison operations. "An op with no
  handler declines by name."
- The driver has two parts: `driver.py`, which handles MPS dispatch, buffer binding, and
  scalar packing, and `metal_torch.mm`, an ObjC++ bridge **over torch's MPS stream**. A
  downstream tester also drove it through a torch-free native runtime.
- Friction points:
  - You need Triton built with `TRITON_EXT_ENABLED=1` and an asserts build of LLVM,
    which the README describes as "roughly 4 GB of clone and an hour of build."
  - No macOS Triton wheel exists.
  - Xcode 27 Metal-toolchain quirks came up in testing.
  - The author deliberately **deferred Metal 4 and `MTLTensor`** for stability.
- Traction: Liger-Kernel has a [draft PR #1442](https://github.com/linkedin/Liger-Kernel/pull/1442)
  that routes through PR #126, and its RMSNorm tests pass on an M5. NeuroBrix reports 12
  models running end to end on an M4 Pro.

### triton-msl (independent, pip-installable)

- The repository is [bledden/triton-msl](https://github.com/bledden/triton-msl) (21
  stars, 752 commits by one author, created 2026-02, v0.3.0 on 2026-09-16). Forks
  include [Janssena](https://github.com/Janssena/triton-msl) and
  [benkelaya](https://github.com/benkelaya/triton-msl).
- Pipeline: TTGIR is walked **in Python** (`mlir_walker.py` → `generic_lowerer.py`),
  then emitted as MSL, compiled with `xcrun metal`, and dispatched.
- **Core design weakness:** the generic lowerer uses a **1D, one-thread-per-element
  scalar model**. `expand_dims`, `broadcast`, and `convert_layout` are no-ops. `tt.dot`,
  softmax, layernorm, sort, and transpose instead use **hand-written MSL templates
  selected by pattern detectors**. The project's own docs call these "structural debt."
  A multi-element-per-thread register-array mode exists but is "perf-neutral."
- Interop: MPS tensors run **zero-copy through `torch.mps.compile_shader`** inside the
  active MPS stream. CPU tensors use `newBufferWithBytesNoCopy`.
- Performance, as the project reports:
  - Large memory-bound kernels reach about 315-347 GB/s on an M4 Max, which is 58-64% of
    peak.
  - The fast matmul path reaches about 11-12 TFLOP/s.
  - Attention runs 1.94-2.18x faster than PyTorch SDPA but is **23-44% slower than
    MLX**.
  - Cold specialization can take up to 9.5 s.
- Supports `@triton.autotune` through `GPUStartTime` and `GPUEndTime` timing, plus an
  Inductor integration.

### triton-shared and linalg

[microsoft/triton-shared](https://github.com/microsoft/triton-shared) (347 stars) has
been quiet since 2025-12-05. A path of Triton → linalg → SPIR-V → SPIRV-Cross → MSL is
theoretically possible. No project does it, because every live effort chose direct
TTGIR → MSL instead. Lowering through linalg also discards the tile and layout
information that matters for simdgroup MMA.

### vllm-metal

[vLLM's Metal blog, 2026-09-22](https://vllm.ai/blog/2026-09-22-vllm-metal-v0-28-0)
describes a **hand port** of vLLM's unified Triton attention kernel to a paged varlen
Metal kernel. It isn't Triton-compiled. The rest of the model reuses `mlx_lm` layers.
vllm-metal also added a zero-copy DLPack bridge between MLX and torch
([vllm-metal#758](https://github.com/vllm-project/vllm-metal/pull/758)).

**Lesson:** Porting Triton was hard because TTGIR's layout system is CUDA-shaped. Both
serious efforts struggled most with `tt.dot` and `convert_layout`. Emitting MSL text,
not AIR, is the consensus choice because it is documented, debuggable, and survives
Xcode updates.

## 2. MLX

[ml-explore/mlx](https://github.com/ml-explore/mlx) has 28.6k stars, and v0.32.2 shipped
on 2026-08-25.

### `mx.fast.metal_kernel`

The [custom Metal kernels docs](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
describe the following behavior:

- You write **only the MSL body**. MLX generates the `[[kernel]]` signature from
  `input_names` and `output_names`. Inputs become `const device T*` and outputs become
  `device T*`. MLX adds `NAME_shape`, `NAME_strides`, and `NAME_ndim` only if the body
  references them. It also adds whichever Metal attributes (`thread_position_in_grid`
  and others) the body mentions.
- `template=[("T", mx.float32), ...]` instantiates template parameters. The kernel name
  is mangled per instantiation, and each unique instantiation triggers a JIT compile.
- `grid` is the **total thread count** because MLX calls `dispatchThreads`, not
  `dispatchThreadgroups`. `threadgroup` sets the group size.
- `ensure_row_contiguous=True` (the default) copies non-contiguous inputs. When it's
  `False`, you use `elem_to_loc()` from MLX's `utils.h`, which MLX includes
  automatically.
- `atomic_outputs` declares outputs as `device atomic<T>*`. `init_value` pre-fills
  outputs.
- `output_shapes` and `output_dtypes` are required at call time. `verbose=True` prints
  the generated source. `math_mode` defaults to `"safe"`.
- For autodiff, you pair it with `@mx.custom_function` and a `.vjp`.

**Limitations for kernel authors:**

- It's a string template. You get no tiling abstraction, no autotuner, and no bounds
  or masking help.
- You declare `threadgroup` memory by hand inside the body.
- Error messages come straight from the Metal compiler and point at generated code.
- It's MLX-only.
- A bug filed as [mlx#4534](https://github.com/ml-explore/mlx/issues/4534) (open,
  2026-09-19) reports that MLX clamps `group_dims` to `min(threadgroup, grid)`. The
  result is a **silently truncated dispatch** in the common
  one-threadgroup-per-row pattern.

### Other MLX mechanisms

- **C++ extensions** subclass `Primitive`, implement `eval_gpu` with metal-cpp, and bind
  through nanobind and CMake. This gives full control at a high build cost.
- **`mx.compile`** fuses elementwise graphs into generated Metal kernels. It doesn't
  generate matmul or attention kernels, because those come from MLX's hand-written
  "Steel" templates.
- **DLPack:** MLX #3531 added zero-copy Metal DLPack with PyTorch MPS. PyTorch 2.12 and
  later allocate ordinary MPS tensors in shared storage, so sharing is zero-copy there.
  Private buffers still copy.
- MLX ships Neural Accelerator matmuls on M5 and is the performance baseline everyone
  compares against.

**Ecosystem signal:** [ZMLX](https://github.com/Hmbown/ZMLX) (53 stars) calls itself a
"Triton-style kernel toolkit for MLX." It is a library of fused `metal_kernel` patches
with a 3-13% decode speedup, not a compiler.

**Lesson:** The MLX-style "body-only MSL plus generated signature" is the de facto
low-level API. Forge's generated code can target it directly as one launcher.

## 3. PyTorch MPS

### `torch.mps.compile_shader` (PyTorch 2.5 and later)

The source for this API is
[torch/mps/\_\_init\_\_.py](https://github.com/pytorch/pytorch/blob/main/torch/mps/__init__.py)
and `torch/csrc/mps/Module.cpp`.

- It takes a **full MSL source** and returns a library object. Each kernel is a
  callable: `lib.kernel(*args, threads=..., group_size=..., arg_casts=...)`.
- `threads` is a total thread count with one to three dimensions, using
  `dispatchThreads` semantics. It defaults to `numel` of the first tensor.
- Tensors bind as buffers at their storage offset. A Python `float` becomes a `float`,
  and an `int` becomes an `int64` unless you override it with `arg_casts`. Lists become
  arrays.
- Kernel objects expose `max_threads_per_threadgroup`, `thread_execution_width`, and
  `static_thread_group_memory_length`.
- **The kernel runs inside PyTorch's MPS stream.** This is the key interop fact:
  - It orders correctly with surrounding torch operations.
  - It needs no `MTLBuffer` extraction.
  - It composes with `torch.compile`.
  - Both triton-msl and TileLang use it as their launcher.

### `torch.mps.load_metallib(bytes | path)`

This API merged in April 2026 as PR #177276, after a revert and reland. Its docstring
explicitly names "Triton, MetalASM" as producers. It lets a compiler ship precompiled
metallibs and skip runtime MSL compilation.

`torch.mps._host_alias_storage` (private, 2026-04) exposes a CPU alias of MPS storage.

### Measured locally (M4 Pro, torch 2.11)

| Workload | Host enqueue per launch | Wall time per launch |
|---|---|---|
| `compile_shader` add, 1,024 elements | about 1.3 µs | about 24 µs |
| Built-in `torch.add`, 1,024 elements | - | about 26 µs |
| `compile_shader` add, 16M floats | - | 862 µs |

The 16M-float add moves 192 MB in 862 µs, which is about 223 GB/s, or roughly 82% of the
M4 Pro's 273 GB/s peak. JIT-compiling a trivial shader took 6.4 ms. Custom kernels
dispatched through `compile_shader` cost no more than native ATen operations. The
per-kernel floor comes from the MPS stream, not Python.

### Other torch paths

- **C++ and ObjC++ extensions** use `torch::mps::get_command_buffer()`,
  `get_dispatch_queue()`, and the `id<MTLBuffer>` from `tensor.storage().data()`. This
  is the classic "custom MPS op" route.
- **Inductor MPS codegen** lives in `torch/_inductor/codegen/mps.py`, about 52 KB. Its
  header still reads "not a feature-complete compiler backend… early prototype." It
  emits MSL for elementwise operations and reductions: threadgroup reductions through
  `c10::metal::threadgroup_*` with `max_threadgroup_size = 1024` and
  `simd_group_size = 32`, plus multistage reductions. It has **no matmul or tile
  codegen**, because matmul goes to MPSGraph. Development is active, with commits on
  2026-09-14 and 2026-09-18, and correctness bugs remain open
  ([#152155](https://github.com/pytorch/pytorch/issues/152155) and
  [#196233](https://github.com/pytorch/pytorch/issues/196233)).
- **Apple coreai-torch** ([apple/coreai-torch](https://github.com/apple/coreai-torch),
  157 stars, created 2026-05, v0.4.3 on 2026-09-24) offers `TorchMetalKernel`. You write
  an MSL body with `TYPE` dtype placeholders and a torch reference implementation, and
  you set `threads_per_grid`, `threads_per_thread_group`, and `result_shapes` at the
  call site. The kernel is embedded in a Core AI `.aimodel`. It's aimed at export and
  deployment, not eager research, and Apple labels it experimental
  ([WWDC26 session 325](https://developer.apple.com/videos/play/wwdc2026/325/)).

## 4. tinygrad

[tinygrad/tinygrad](https://github.com/tinygrad/tinygrad) has 33.7k stars and was pushed
daily as of the snapshot.

- **Renderer:** `MetalRenderer(CStyleLanguage)` in `renderer/cstyle.py` emits MSL. It
  uses one argument struct (`constant args_t& args [[buffer(0)]]`) plus
  `gid [[threadgroup_position_in_grid]]` and `lid [[thread_position_in_threadgroup]]`.
  Shared memory is declared as `threadgroup __attribute__((aligned(16)))`, and bf16
  transcendentals are upcast to float. Tensor cores (`tc.metal`, `simdgroup_matrix`)
  turn on for Apple7 and later.
- **Runtime:** `runtime/ops_metal.py` uses **pure ctypes plus a small objc-msgSend
  helper** (`tinygrad.runtime.support.objc`) with autogenerated Metal bindings. It
  doesn't use PyObjC.
  - It compiles through the **private `MTLCodeGenServiceBuildRequest` in
    MTLCompiler**, not `newLibraryWithSource`, and checks for the `MTLB` and `ENDT`
    magic bytes. It selects `metal4.0` on macOS 26.
  - It allocates shared-storage buffers and uses residency sets when available.
  - It batches graphs into **indirect command buffers (ICBs)** and uses
    `MTLSharedEvent` for synchronization and `GPUStartTime` and `GPUEndTime` for
    timing.
- **Model:** tinygrad is array-level. Kernels come from the scheduler and the
  optimizer, including BEAM search. You don't author kernels directly.

**Lessons:**

- ICB and graph replay is the way to beat per-dispatch overhead.
- ctypes plus `objc_msgSend` removes the PyObjC dependency and its call overhead.
- The private compiler service is fast but fragile across OS releases. The tinygrad repo
  has several issues about Metal path changes.

## 5. Other compilers and DSLs

### Mojo and MAX (Modular)

- Mojo lowers to LLVM IR, then to **AIR 2.7**, then to a metallib through metal-cpp,
  with a `MetalDeviceContext` runtime. It requires macOS 15 or later and Xcode 16 or
  later ([forum thread](https://forum.modular.com/t/apple-silicon-gpu-support-in-mojo/2295)).
- Initial support arrived in 25.6 (September 2025). By 2025-12, almost all GPU puzzles
  ran. MAX 26.4 (2026-06-27) serves Llama and Qwen on M1-M5, with best results on M5
  Neural Accelerators ([forum announcement](https://forum.modular.com/t/max-models-can-now-run-on-apple-silicon-gpus/3283)).
- Gaps: **no PyTorch interop**, and atomics and matrix intrinsics arrived late.
  Modular's own issue [#7181](https://github.com/modular/modular/issues/7181) shows
  that emitting AIR directly breaks on OS and Xcode updates ("Failed to create Metal
  function" on macOS 27.2).
- **Model:** SIMT plus `LayoutTensor` tiles, in a new language instead of a Python
  eDSL.

### TileLang

- The model is **tile-based with explicit memory scopes**: `T.Kernel(grid, threads=)`,
  `T.alloc_shared`, `T.alloc_fragment`, `T.copy`, `T.gemm`, and `T.Pipelined`. It is
  closer to CUTLASS/CuTe-in-Python than to Triton, because you place data explicitly
  instead of relying on compiler-chosen layouts.
- Metal timeline:
  - [PR #799](https://github.com/tile-ai/tilelang/pull/799) (2025-10-07) added the
    first backend, which used TVM Metal codegen and launched through
    `torch.mps.compile_shader`. Its early GEMM was 2-3x faster than torch's native Metal
    matmul and 2-3x slower than MPSGraph.
  - [PR #1869](https://github.com/tile-ai/tilelang/pull/1869) (2026-05-22) added
    `T.gemm` through `simdgroup_matrix` 8x8. A pass rewrites `local.fragment` to
    `metal.simdgroup` scope, so **the same kernel source runs on CUDA and Metal**.
  - [PR #2252](https://github.com/tile-ai/tilelang/pull/2252) (2026-07-28) added the
    M5 cooperative-tensor `T.gemm`, with a simdgroup fallback and capability gating.
- The PR #2252 design note says: "explicit threadgroup staging is not automatically a
  faster path than feeding cooperative tensor operands directly." The CUDA-shaped
  staging path is kept only for compatibility.
- The heavy dependency on a TVM fork is its weakness.

### ThunderMittens (Hazy Research)

- The [ThunderMittens blog post](https://hazyresearch.stanford.edu/blog/2024-11-28-tk-mlx)
  describes ThunderKittens' tile primitives ported to MSL as a C++ header library that
  MLX calls. The repository, [HazyResearch/ThunderMittens](https://github.com/HazyResearch/ThunderMittens),
  has 23 stars and is inactive since 2025-08.
- The only abstraction change from CUDA was 16x16 → **8x8 base register tiles**.
- The Hazy Research team reported these Apple findings:
  - Async copies are deprecated on M2.
  - Swizzling isn't worthwhile.
  - Shared memory matters little, because loading directly from device memory to
    registers is enough.
  - Register occupancy dominates performance.
  - The compiler handles bf16 poorly.
- Performance: GEMM runs about 9% faster than MLX, and attention runs within 15% of
  MLX.

### metal-flash-attention (Philip Turner)

- The repository is [philipturner/metal-flash-attention](https://github.com/philipturner/metal-flash-attention)
  (612 stars, inactive since 2024-09).
- It generates MSL at runtime from Swift and specializes it per problem.
- It uses intentional register spilling at head dimension 256 with skewed blocks
  (16-32 × 80-128).
- Its backward pass uses 7 GEMMs instead of 5 to avoid FP32 atomics.
- It reached 62-71% ALU utilization and measures performance in instructions per
  second.

### llama.cpp ggml-metal

- The code lives in [llama.cpp `ggml/src/ggml-metal`](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-metal).
  The kernels split into per-family `kernels/*.metal` files, such as `mul_mv`, `fa_*`,
  `fa_vec_*`, `conv`, and `argsort`. Each quantization type gets its own FlashAttention
  file.
- The host side spans several files: `ggml-metal-ops.cpp` (200 KB),
  `ggml-metal-device.m`, `ggml-metal-fusion.cpp`, and `ggml-metal-tuning.cpp`, which
  holds per-device tuned tables.
- The design is hand-written templates, function constants, simdgroup reductions,
  `simdgroup_matrix` `mul_mm`, and a large dispatch-selection layer.

**Lesson:** Production Apple performance comes from per-shape variant selection and
tuning tables, not from one generic kernel.

### Metal.jl (Julia)

- [JuliaGPU/Metal.jl](https://github.com/JuliaGPU/Metal.jl) has 464 stars and is pushed
  daily. It compiles Julia to LLVM, then to AIR, then to a metallib, and it emits the
  newest AIR and MSL versions the OS supports. It supports KernelAbstractions.
- [Metal.jl 1.10](https://juliagpu.org/post/2026-07-01-metal-1.10/index.html)
  (2026-07-01) adds native GEMMs in three variants: `:scalar`, `:simd`
  (`simdgroup_matrix`), and `:tensor` (`tensor_ops::matmul2d`).
- Runtime improvements in 1.10 include the following:
  - It keeps one command buffer open for batched submission.
  - Idle-queue sync dropped from 15.87 µs to 0.19 µs.
  - A small-kernel loop improved 2.4x, from 359 µs to 149 µs.
  - Time to first kernel dropped from 8.1 s to 0.16 s.
- **Model:** SIMT with simdgroup intrinsics.

### Warp on Metal

- Official NVIDIA [Warp](https://github.com/NVIDIA/warp) (7.1k stars) ships
  **CPU-only macOS wheels**.
- The community [innate-inc/warp-metal](https://github.com/innate-inc/warp-metal)
  (created 2026-09-18) adds a `metal:0` device to stock `warp-lang` 1.17. It supports
  tiles with a threadgroup arena, backward kernels, graph capture, and DLPack and torch
  interop over unified memory, and it passes MuJoCo Warp's test suite.
- Its documented limits include the following:
  - No float64.
  - "Some atomics are not atomic," because Metal has only 32-bit atomics.
  - No forward-progress guarantee.
  - An Apple compiler miscompile with unrolled matrix products of 4x4 and larger.
- **Model:** SIMT plus cooperative tiles.

### CubeCL (Rust, Burn)

- [tracel-ai/cubecl](https://github.com/tracel-ai/cubecl) has 2.4k stars and is pushed
  daily. Rust procedural macros generate code for CUDA, ROCm, and wgpu. On Apple, wgpu
  has an **MSL compiler path** (`wgpu-msl`) with CMMA mapped to `simdgroup_matrix`.
- It found an Apple compiler bug: with `max_total_threads_per_threadgroup(32)`,
  `simdgroup_load` can observe stale stage data. CubeCL therefore "never declares a
  single-simdgroup bound" ([cubek#645](https://github.com/tracel-ai/cubek/pull/645)).
- **Model:** SIMT with a "cube" hierarchy and comptime specialization.

### Taichi

- [taichi-dev/taichi](https://github.com/taichi-dev/taichi) has 28.4k stars, but its
  last release was v1.7.4 on 2025-07-31, so it is effectively in maintenance.
- It has a Metal backend. Genesis maintains the fork [Quadrants](https://github.com/Genesis-Embodied-AI/quadrants),
  with 217 stars and active development.
- **Model:** SIMT kernels over sparse fields, aimed at simulation and not ML tiles.

### Halide

[halide/Halide](https://github.com/halide/Halide) (6.6k stars, active) has a mature Metal
target. Its algorithm and schedule separation is a proven idea for keeping kernel math
separate from its mapping to hardware. Halide targets image processing and has no
matmul-tile focus.

### TVM and MLC-LLM

[apache/tvm](https://github.com/apache/tvm) has 13.8k stars and
[mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm) has 23.2k. TVM's Metal codegen is the
base for TileLang. MLC-LLM deploys LLMs to Metal, but in 2026 comparisons MLX
consistently outperforms it on Macs.

### IREE

[iree-org/iree](https://github.com/iree-org/iree) has a Metal HAL that the project marks
**experimental**. Its codegen goes from SPIR-V through SPIRV-Cross to MSL. Its runtime
limits include one-shot command buffers only and no executable caching
([Metal HAL design doc](https://iree.dev/developers/design-docs/metal-hal-driver/)).
Going through SPIR-V loses access to Metal-specific features such as `simdgroup_matrix`
and `tensor_ops`.

### wgpu, naga, and WebGPU

[gfx-rs/wgpu](https://github.com/gfx-rs/wgpu) (18k stars) translates WGSL to MSL through
naga. Cooperative-matrix support is experimental in naga (PR #8251). Chromium exposes
`chromium-experimental-subgroup-matrix`, which maps to `simdgroup_matrix` and is
float-only on Metal. This path is portable, but WebGPU's lowest-common-denominator
design sacrifices features Apple offers.

### Slang

[shader-slang/slang](https://github.com/shader-slang/slang) (5.7k stars, active) has a
first-class Metal target that emits readable MSL, plus autodiff. SlangPy runs on Metal.
Slang is graphics-oriented and SIMT, with no tile or MMA abstraction for ML.

### Numba and JAX

- Numba has no Metal backend.
- Apple's `jax-metal` plugin stalled at 0.1.1 (2024-10), and Pallas has no Metal
  target.

## 6. Python-to-Metal bridges

| Bridge | Mechanism | Status | Pros | Cons |
|---|---|---|---|---|
| PyObjC (`pyobjc-framework-Metal` 12.2.2, 2026-08) | Full Objective-C bridge | Active | Complete API, and you can prototype everything | Each message costs microseconds, a dispatch needs about 10 calls, and it's a heavy dependency |
| Raw ctypes plus `objc_msgSend` (tinygrad) | Hand or autogenerated selectors | Active in tinygrad | No dependencies and lower overhead | Fragile, and you manage ARC and blocks yourself |
| [metalcompute](https://github.com/baldand/py-metal-compute) (89 stars, 0.2.9, 2025-01) | C extension | Dormant | Minimal API | No torch or MLX interop |
| [metalgpu](https://github.com/Al0den/metalgpu) (36 stars, 1.0.5, 2024-08) | C++ wrapper, NumPy views | Dormant | Zero-copy NumPy | Dormant, and no framework interop |
| nanobind plus metal-cpp | Custom C++ extension | Pattern used by MLX and TileLang internals | Lowest overhead and full control | You must build and ship native wheels |
| `torch.mps.compile_shader` and `load_metallib` | Runs in the torch MPS stream | Official and active | Zero-copy, ordered with torch, and about 1 µs host enqueue (measured) | Torch-only, and you inherit the MPS stream's roughly 24 µs per-dispatch floor |
| `mx.fast.metal_kernel` | Runs in the MLX stream | Official and active | Lazy graph, autodiff hooks | MLX-only, body-only codegen, and a truncation bug |

The best-measured runtime-overhead data points come from Metal.jl 1.10's numbers and the
local `compile_shader` measurement. No authoritative PyObjC per-dispatch benchmark was
found.

## 7. Comparison table

| Project | Programming model | Author writes | Apple lowering | Matmul hardware | Framework interop | Activity (2026-09) |
|---|---|---|---|---|---|---|
| TileLang | Tile, explicit scopes (shared, fragment) | Python eDSL | TVM → MSL | simdgroup 8x8 and M5 cooperative tensor | torch MPS (`compile_shader`) | 7.5k stars, very active |
| triton-ext AppleGPU | Tile (Triton, compiler layouts) | `@triton.jit` | TTGIR → MSL (C++) | simdgroup MMA in open PR #126 | torch MPS stream, native | Minimal merged 2026-09-14 |
| triton-msl | Tile frontend, 1D scalar backend plus templates | `@triton.jit` | TTGIR → MSL (Python) | Template matmul, about 11-12 TFLOP/s on M4 Max | torch MPS, Inductor, partial MLX | 21 stars, alpha, one author |
| Mojo and MAX | SIMT plus layout tensors | Mojo | LLVM → AIR | M5 Neural Accelerators through MAX | No torch interop | 29.9k stars (monorepo), active |
| MLX `metal_kernel` | SIMT, raw MSL body | MSL string | `newLibraryWithSource` | Manual | MLX, DLPack to torch | 28.6k stars, active |
| `torch.mps.compile_shader` | SIMT, raw MSL | MSL string | Runtime MSL or metallib | Manual | torch | Active |
| Inductor MPS | Generated (pointwise and reduction) | PyTorch code | MSL | None (MPSGraph) | torch | Prototype, active |
| coreai-torch | SIMT, raw MSL body | MSL string | Core AI asset | Manual or `tensor_ops` | torch export | 157 stars, experimental |
| tinygrad | Array-level, auto-scheduled | Tensor ops | MSL through private MTLCompiler | simdgroup tensor cores | Own runtime | 33.7k stars, very active |
| ThunderMittens | Tile (8x8 register tiles) | C++ MSL headers | MSL | simdgroup | MLX | Inactive |
| Metal.jl | SIMT (KernelAbstractions) | Julia | LLVM → AIR | simdgroup and `tensor_ops` | Julia | Active |
| warp-metal | SIMT plus cooperative tiles | Python (`@wp.kernel`) | Warp → MSL | Tiles | DLPack and torch (UMA) | New (2026-09), community |
| CubeCL | SIMT (cube), comptime | Rust | wgpu-msl | simdgroup CMMA | Burn | 2.4k stars, active |
| Taichi and Quadrants | SIMT, fields | Python | SPIR-V and MSL | None | NumPy and torch (copy) | Taichi in maintenance, Quadrants active |
| IREE | Compiler from MLIR | Model graphs | SPIR-V → SPIRV-Cross → MSL | None | Own runtime | Metal HAL experimental |
| Slang | SIMT shader language | Slang | MSL | None | SlangPy | Active |
| Halide | Algorithm plus schedule | C++ DSL | MSL | None | Own runtime | Active |

## 8. Hardware and API facts any Forge design must respect

- **SIMD width is 32.** A threadgroup holds at most 1,024 threads and **32 KB of
  threadgroup memory**. The GPU has no FP64 and no FP8 compute, and only 32-bit atomics
  are native. The GPU gives no forward-progress guarantee.
- **The MMA unit is `simdgroup_matrix` 8x8** on M1 and later, and it supports float
  types only.
- **On M5, the Neural Accelerators are reachable only through Metal 4 `tensor_ops`**,
  meaning MPP `matmul2d` and `cooperative_tensor`.
  - The [Rigel paper](https://arxiv.org/html/2606.12765v1) measured `matmul2d` on an
    M4 Max. It runs on the shader cores, peaks at 14.8 TFLOP/s in fp16, and is only
    1.05-1.21x faster than hand `simdgroup_matrix`. FP8 is emulated at 0.94x of fp16.
  - Rigel also found that fusing cheap epilogues gains 6.5-12.9%. Replacing MMA with
    scalar code, as in scalar FlashAttention, is 3.6-5x slower.
- [WWDC26 session 330](https://developer.apple.com/videos/play/wwdc2026/330/) covers
  `tensor_ops`:
  - It recommends slicing tensors by threadgroup ID and calling `matmul2d`.
  - It demonstrates keeping FlashAttention intermediates in cooperative tensors, with
    `reduce_rows`, `map_iterator`, and `get_left_input_cooperative_tensor`.
  - It adds quantized planes (int4 and int8 on macOS 26, and FP8, int2, and MX scaling
    on macOS 27).
- Registers and on-chip memory are hardware-managed and unified. Staging through
  threadgroup memory is often a loss, according to ThunderMittens and TileLang PR #2252.
- Known Apple compiler bugs:
  - A single-simdgroup bound miscompiles (CubeCL).
  - Unrolled products of 4x4 and larger matrices miscompile (warp-metal).
  - bf16 code generation is weak (ThunderMittens).

## Positioning and lessons for Forge

### Where the field stands

The "Triton for Apple" slot now has three occupants:

- **TileLang** is the most complete. It offers cross-vendor source, simdgroup and M5
  cooperative-tensor GEMM, wheels, and CI, but it carries a heavy TVM stack.
- **triton-ext AppleGPU** has official-org legitimacy and a real MLIR pipeline, but it
  is only elementwise today, needs a from-source Triton and LLVM build, and deliberately
  skips Metal 4.
- **triton-msl** is pip-installable but architecturally limited by its scalar lowering
  and template zoo.

All three are **CUDA-first abstractions ported to Apple**. Each one had to fight
CUDA-shaped assumptions: TTGIR layouts, shared-memory staging, and 16x16 MMA. None is
Apple-native in its semantics. Only TileLang exposes M5 Neural Accelerators, and none of
them treats MLX as a first-class host.

### Where Forge can differentiate

1. **Apple-first tile semantics.**
   - Base tiles are 8x8, and the tile shapes map onto `simdgroup_matrix` and
     `cooperative_tensor`.
   - Data loads directly from device memory to registers by default, and threadgroup
     staging is opt-in.
   - A single `dot` lowers to `simdgroup_matrix` on M1-M4 and to `tensor_ops` on M5,
     chosen at runtime by capability.
   - Forge doesn't pretend to be portable to CUDA. Portability is the incumbents'
     selling point, and Apple-native performance and ergonomics are an open lane.
2. **Zero-heavy-dependency install.**
   - `pip install forge` must work in seconds, with no LLVM, TVM, or Triton build.
   - Forge emits **MSL text**, the consensus choice. Direct AIR broke for Triton PR #48
     and Mojo #7181.
   - Compilation uses `xcrun metal` or runtime `newLibraryWithSource`, which took 6.4 ms
     locally for a trivial kernel, with a disk cache for metallibs.
3. **Framework-neutral launching.** Forge supports three launchers behind one API:
   - `torch.mps.compile_shader` and `load_metallib` for PyTorch. This path is
     zero-copy and ordered in the MPS stream.
   - `mx.fast.metal_kernel` for MLX. This path is lazy and hooks into autodiff.
   - A standalone ctypes or nanobind runtime with ICB graph replay for NumPy and UMA
     workloads.

   Forge also offers DLPack handoff. No competitor covers torch and MLX equally.
4. **A readable, first-class MSL escape hatch.** Forge can show the generated MSL (as
   MLX's `verbose=True` does), let you inline raw MSL snippets, and map errors back to
   Python source lines, which TileLang added only in 2026-07.
5. **Honest correctness.**
   - Forge adopts triton-msl's "refuse loudly, never return wrong numbers" integrity
     model.
   - It avoids MLX #4534-style silent grid truncation by validating grid and group
     sizes.
   - It carries workarounds for known Apple compiler bugs.
6. **Built-in autotuning and specialization.** Forge times kernels with
   `GPUStartTime` and `GPUEndTime`, specializes per shape (as metal-flash-attention and
   ggml tuning tables do), and keeps per-chip-variant caches, as NeuroBrix requested.

### Lessons to adopt

- **Emit MSL, not AIR.** It's documented, debuggable, and survives Xcode updates.
- **Don't build a 1D scalar lowerer.** Design register-array and tile lowering from day
  one. triton-msl shows that the scalar model forces a template zoo for anything
  involving `dot`, layout conversion, or transpose.
- **Treat per-dispatch overhead as the stream's cost, not Python's.** The floor is about
  24 µs per tiny kernel on the torch stream. To go lower, you need batching, persistent
  command buffers, or ICBs, as tinygrad and Metal.jl 1.10 do.
- **Make fusion the product.** Rigel and ZMLX show that wins come from fusing epilogues
  and decode paths around matmul units. Scalar rewrites of MMA-heavy work lose.
- **Gate M5 features by capability** with a simdgroup fallback, as TileLang does.
- **Accept that `tensor_ops` is a black box.** Expect only 5-21% over hand
  `simdgroup_matrix` on M4. The real gain is on M5 Neural Accelerators.
- **Benchmark against MLX, not MPS.** MLX is the bar that triton-msl attention (23-44%
  slower) and TileLang early GEMM (2-3x slower than MPSGraph) failed to clear.

### Strategic risk

If triton-ext PR #126 lands and Triton ships macOS wheels, "write Triton on your Mac" is
solved for portability-seeking users. If TileLang keeps its pace, it holds the
performance-seeking tile-DSL slot. Forge should treat the following as an explicit
alternative, not an afterthought: become a frontend or tile library that targets MSL and
reuses these launchers, or contribute Apple-native lowering ideas to triton-ext or
TileLang. Build Forge standalone only if the Apple-first ergonomics and MLX-plus-torch
story from the preceding list is compelling.
