# Triton internals and lessons for Forge

This report covers how Triton works internally, which parts of it are worth copying for
Forge (a Python-first tile DSL that targets Apple M-series GPUs through Metal), and which
parts to drop or simplify. File paths refer to the
[triton-lang/triton](https://github.com/triton-lang/triton) `main` branch as of
September 2026 unless noted.

---

## 1. Frontend: `@triton.jit`, specialization, and caching

### 1.1 `JITFunction`

`@triton.jit` wraps a Python function in `JITFunction`
(`python/triton/runtime/jit.py`). At decoration time, it stores the source text
(`inspect.getsource`), parses the signature into `KernelParam` objects (each tracking
annotations such as `tl.constexpr`, `do_not_specialize`, and
`do_not_specialize_on_alignment`), and builds a fast argument binder with
`create_function_from_signature()`.

The launch syntax `kernel[grid](args...)` is plain Python. `JITFunction.__getitem__`
returns a lambda that calls `self.run(grid=grid, warmup=False, *args, **kwargs)`. Inside
`run()`, the following happens:

1. The binder maps positional and keyword args to parameters.
2. Each argument is *specialized* into a short type string (see 1.2).
3. A cache key is built from the specialization tuple, the constexpr values, and the
   options (`num_warps`, `num_stages`, and others): `compute_cache_key(kernel_key_cache,
   specialization, options)`.
4. On a miss, `_do_compile()` calls `triton.compile()`; on a hit, the cached
   `CompiledKernel` is reused.
5. If `grid` is callable, it's invoked with the bound-argument dict (which includes
   constexprs and autotuner-injected meta-parameters): `grid = grid(bound_args)`. This is
   why the idiom is `grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)`.
6. `kernel.run(grid_0, grid_1, grid_2, stream, ...)` calls into a generated C launcher.

### 1.2 Argument specialization

Specialization lives in C++ for speed (`python/src/specialize.cc`,
`native_specialize_impl`), with hooks on the backend (`BaseBackend.get_int_specialization`,
`get_tensor_specialization`, `parse_attr` in `python/triton/backends/compiler.py`). The
rules are the following:

- **Tensors** become a pointer type string from their dtype, for example `*fp16` or
  `*fp32`. If `data_ptr() % 16 == 0`, the key gets a `D` marker, which becomes the MLIR
  attribute `tt.divisibility = 16` on the argument. The compiler uses it to prove
  alignment for vectorized 128-bit loads.
- **Integers** become `i32`, `i64`, or `u64` depending on range. If `val % 16 == 0`, they
  also get `D` (strides and sizes divisible by 16 enable vectorization). If `val == 1`,
  the argument is treated as a **constexpr** and removed from the runtime argument list.
  This `==1` specialization mainly exists so that `stride == 1` becomes a compile-time
  fact, letting the compiler prove contiguity and coalesce. It's also a known source of
  bugs, because it silently changes the launcher's argument layout and the kernel's types
  (see [triton#2939](https://github.com/openai/triton/issues/2939),
  [triton#9639](https://github.com/triton-lang/triton/pull/9639)).
- **Floats** become `fp32`; **bools** become `u1`.
- **Tensor descriptors** become `tensordesc<dtype[block_shape]>`.
- **`tl.constexpr` params** are folded: their Python values are part of the cache key and
  are substituted during code generation, never passed at runtime.

`do_not_specialize=["n"]` opts a parameter out, which avoids recompiling when `n` flips
between divisible and non-divisible values.

### 1.3 AST to IR: `CodeGenerator`

`python/triton/compiler/code_generator.py` defines `CodeGenerator(ast.NodeVisitor)` and
the entry point `ast_to_ttir()`. It walks the Python AST and emits MLIR through a pybind
builder (`self.builder`, an `ir.builder` from `python/src/ir.cc`; a
`GluonOpBuilder` for Gluon). Type promotion and broadcasting rules live in a separate
`semantic` module (`python/triton/language/semantic.py`). Key behavior:

- `visit_FunctionDef` creates the `tt.func`, maps args to block arguments, and infers
  return types with `handle_returns()`.
- `visit_If` evaluates a constexpr condition in Python and emits only the taken branch.
  A dynamic condition emits `scf.if`, with SSA values that are live across branches
  becoming results.
- `visit_For` unrolls `tl.static_range` at compile time. `range` and `tl.range` become
  `scf.for` with loop-carried values discovered by tracking `local_defs`. `tl.range`
  carries per-loop hints (`num_stages`, `loop_unroll_factor`, `warp_specialize`,
  `flatten`).
- `visit_While` emits `scf.while`.
- `visit_Call` dispatches `@triton.jit` callees to `call_JitFunction`, which mangles the
  name by argument types and constexpr values, generates a separate `tt.func`, and relies
  on the inliner pass later.
- Constexprs are wrapped in `tl.constexpr` objects; `_unwrap_if_constexpr()` gets the
  Python value. This makes Python itself the metaprogramming language: any Python
  expression over constexprs is evaluated at compile time.
- Errors become `CompilationError` with the source location from `jit_fn.src`.

Only a subset of Python is allowed: no closures over tensors, no lists of tensors
(tuples are supported in recent versions), no early `return` inside dynamic control
flow, no `break` in loops, and no recursion.

### 1.4 Cache keys and the on-disk cache

Two levels of caching exist:

- **In memory:** per device, a dict keyed by (specialization string, constexpr values,
  options) maps to `CompiledKernel`.
- **On disk:** `triton.compile()` (`python/triton/compiler/compiler.py`) hashes
  `ASTSource.hash()` (`fn.cache_key` + attrs + signature + constants), the backend hash
  (`BaseBackend.hash()`, which includes, for example, the `ptxas` version), the options,
  relevant env vars, and `triton_key()` (a hash of the installed Triton package source
  and `libtriton`). `JITFunction.cache_key` is a SHA-256 over the kernel's source plus
  everything `DependenciesFinder` (an AST visitor) finds: transitively called JIT
  functions and referenced global constexprs. Changing a helper function invalidates
  its callers.

`get_cache_manager(hash)` returns a directory under `TRITON_CACHE_DIR` (default
`~/.triton/cache/<hash>/`). It stores one file per stage (`.ttir`, `.ttgir`, `.llir`,
`.ptx`, `.cubin`) plus a `.json` metadata file and a `__grp__*.json` group manifest.
Knobs in `python/triton/knobs.py` include `TRITON_CACHE_DIR`, `TRITON_ALWAYS_COMPILE`,
`TRITON_CACHE_MANAGER` (pluggable, for example remote caches), `TRITON_KERNEL_DUMP`,
`TRITON_DUMP_DIR`, and `TRITON_KERNEL_OVERRIDE` (swap in hand-edited IR).

The launcher is a small C file generated per signature
(`third_party/nvidia/backend/driver.py`), compiled once, and cached the same way. Python
launch overhead was a long-standing complaint and moved into C++ over time.

---

## 2. The `triton.language` surface and a v1 ranking

The programming model: one *program* (a CUDA block, a Metal threadgroup) operates on
*tiles*, which are statically shaped, power-of-two-sized values. The user never sees
threads.

The following table ranks `tl` ops by what the canonical kernels need. The kernel
columns reflect the official tutorials
([tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html):
vector add, fused softmax, layer norm, matmul, fused attention).

| Tier | Ops | Needed for |
|---|---|---|
| P0 (v1 must-have) | `program_id`, `num_programs`, `arange`, `load(ptr, mask, other)`, `store(ptr, val, mask)`, `constexpr`, `cdiv`, arithmetic and comparisons, `where`, broadcasting via `[:, None]` and `[None, :]`, `zeros`, `full`, `.to(dtype)`, `exp`, `exp2`, `log`, `sqrt`, `rsqrt`, `abs`, `maximum`, `minimum`, `sum`, `max`, `min`, `range` loops, `static_range` | Elementwise, softmax, layer norm, RMSNorm, row reductions |
| P0 for matmul and attention | `dot(a, b, acc)`, `trans`, `max`/`sum` along an axis, fp32 accumulation, `fma` | Matmul, flash attention |
| P1 | `argmax`, `argmin`, `atomic_add`, `atomic_max`, `atomic_cas`, `cumsum`, `associative_scan`, `reshape`, `permute`, `broadcast_to`, `sigmoid`, `erf`, `tanh`, `static_assert`, `static_print`, `device_print`, `multiple_of`, `max_contiguous` hints | Cross-entropy, top-k helpers, split-K, scans, GELU, debugging |
| P2 | `make_block_ptr`/`advance` (deprecated), `make_tensor_descriptor`, `join`, `split`, `interleave`, `histogram`, `gather`, `sort`, `inline_asm_elementwise`, `extern_elementwise`, `debug_barrier`, `dot_scaled`, `tl.range(num_stages=...)` | Performance tuning, niche kernels |

Notes on specific ops:

- `tl.load(ptrs, mask=m, other=0.0)` takes a *tile of pointers*. Pointer tiles come from
  `base + offs[:, None] * stride0 + offs[None, :] * stride1`. The compiler's
  `AxisInfo` analysis recovers contiguity and divisibility from this arithmetic to decide
  vector width. This is powerful but makes the compiler's job harder than it needs to be.
- `tl.make_block_ptr` described a strided 2D block but is deprecated in favor of
  `tl.make_tensor_descriptor(base, shape, strides, block_shape)` with `desc.load([m, n])`
  and `desc.store([m, n], v)`
  ([docs](https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html)).
  On Hopper and Blackwell it maps to TMA hardware; out-of-bounds accesses are padded
  automatically, so no mask is needed.
- `tl.dot` requires each dimension to be at least 16 and has precision knobs
  (`input_precision="tf32"|"ieee"`), a frequent numerics surprise for fp32 users.
- `tl.reduce` and `tl.associative_scan` take a `@triton.jit` combine function, which
  gives user-defined reductions (for example, Welford) for free.

---

## 3. The IR pipeline and layouts

### 3.1 Stages

`BaseBackend.add_stages()` fills a dict of `ir_name -> function`, and `compile()` runs
them in order. For NVIDIA (`third_party/nvidia/backend/compiler.py`, `CUDABackend`),
the stages are the following:

1. **`make_ttir`** (Triton IR, hardware-agnostic `tt` dialect): inliner, rewrite tensor
   descriptors to pointers (when unsupported), canonicalize, combine, reorder broadcast,
   CSE, symbol DCE, loop unroll.
2. **`make_ttgir`** (TritonGPU IR, `ttg` dialect, every tensor carries a layout
   encoding): `convert_to_ttgpuir` (assigns default blocked layouts from `num_warps`),
   `coalesce`, `remove_layout_conversions`, `accelerate_matmul` (retags `tt.dot`
   operands with MMA and dot-operand layouts), `optimize_dot_operands`, LICM,
   `assign_latencies`, `schedule_loops`, `pipeline` (software pipelining driven by
   `num_stages`), `prefetch`, warp specialization, TMA lowering,
   `reduce_data_duplication`, `reorder_instructions`, fence insertion. Blackwell adds
   tensor-memory passes.
3. **`make_llir`**: SCF to CF, shared-memory allocation (`allocate_shared_memory_nv`),
   `membar` (barrier insertion), `to_llvmir`, then NVVM and debug info.
4. **`make_ptx`**: the LLVM NVPTX backend.
5. **`make_cubin`**: runs `ptxas` as a subprocess.

AMD mirrors this in `third_party/amd/backend/compiler.py` (AMDGCN, `hsaco`).

### 3.2 What layouts are

A *layout* (an MLIR *encoding* attribute on a tensor type) is the function that maps each
element of a logical tile to a hardware location: which register of which thread of which
warp, or which shared-memory address. Triton needs layouts because the language is
tile-level but the hardware is SIMT. Every tile value must be physically distributed
across threads, and the compiler picks that distribution.

The main kinds are the following (see the
[Gluon layouts tutorial](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html)):

- **`BlockedLayout(size_per_thread, threads_per_warp, warps_per_cta, order)`:** the
  tile is tiled by a block of shape `size_per_thread * threads_per_warp *
  warps_per_cta`. `size_per_thread=[1, 4]` with `order=[1, 0]` means each thread holds
  4 contiguous elements of a row, so a load becomes one 128-bit vector load (fp32).
  This is the default layout for loads, stores, and elementwise ops.
- **`SliceLayout(dim, parent)`:** the layout of the result of reducing `parent` along
  `dim`, and the layout of `arange` feeding `x[:, None]`. It makes broadcasting back into
  the parent free.
- **MMA layouts (`NvidiaMmaEncoding`, AMD `MFMA`/`WMMA`):** the fixed register
  arrangement that a tensor-core instruction produces for its accumulator.
- **`DotOperandLayout(op_idx, parent)`:** the register arrangement an MMA instruction
  requires for its A or B operand.
- **Shared layouts (`SwizzledSharedLayout`, `NVMMASharedLayout`):** shared-memory
  placement with XOR swizzling to avoid bank conflicts.

Why layouts matter: they determine coalescing (a warp's 32 threads must touch contiguous
memory), vectorization width, whether an op needs cross-thread communication, and whether
a value can feed the tensor core directly. A `convert_layout` between two register
layouts usually requires a round trip through shared memory plus a barrier, so
`RemoveLayoutConversions` (which propagates layouts forward and backward to cancel
conversions) is one of the most important and most fragile passes.

The [Linear Layouts paper](https://arxiv.org/abs/2505.23819) (Zhou et al., 2025)
replaced the zoo of hand-written layout classes with a single representation: a binary
matrix over F2 that maps the bits of the hardware index (register, lane, warp) to the bits
of the logical tensor coordinate. Any conversion between two layouts becomes linear
algebra, which removes the quadratic number of hand-written conversion paths and fixed
many bugs. The cost is that shapes must be powers of two, which Triton already required.

### 3.3 Coalescing, pipelining, and matmul acceleration

- **`Coalesce`** uses `AxisInfo` (contiguity, divisibility, constancy per dimension) to
  choose a blocked layout where each thread reads the widest aligned vector along the
  contiguous dimension.
- **`AccelerateMatmul`** rewrites `tt.dot` to hardware MMA (`mma.sync`, `wgmma`,
  `tcgen05`), inserting conversions to dot-operand layouts.
- **Software pipelining** (`num_stages`): for a `for k` loop that loads A and B tiles
  and calls `dot`, the compiler allocates `num_stages` shared-memory buffers, issues
  asynchronous copies (`cp.async` or TMA) for iterations `k+1 ... k+S-1` while computing
  iteration `k`, and inserts waits. This is the main reason Triton matmuls are fast
  without the user writing double buffering.
- **`Prefetch`** hoists shared-to-register loads of the next dot operand slice into the
  previous iteration.

### 3.4 The minimal version for a backend that emits MSL

Apple's Metal compiler (MSL to AIR to GPU ISA) handles register allocation, instruction
scheduling, and most vectorization within a thread. Forge doesn't need LLVM, PTX-level
concerns, or asynchronous copy engines. Apple GPUs have no `cp.async`/TMA equivalent,
so multistage pipelining matters much less. Forge still needs a layout concept, because
the tile-to-thread distribution is the one decision Metal's compiler can't make.

A sufficient minimal design has four distributions:

1. **`Blocked(elems_per_thread, threads, simdgroups, order)`:** the default for loads,
   stores, and elementwise ops. With a SIMD-group width of 32 and `num_warps` mapped to
   SIMD groups per threadgroup, this is the same as Triton's `BlockedLayout`. Choose
   `elems_per_thread` along the contiguous axis so loads become `float4` or `half4`
   (requires 16-byte alignment, which is exactly what Triton's divisibility-by-16
   specialization proves).
2. **`Slice(parent, dim)`:** the result of a reduction, implemented as a per-thread
   partial reduction, then `simd_sum` or `simd_max` across the SIMD group, then a
   threadgroup-memory exchange across SIMD groups.
3. **`SimdMatrix`:** an opaque layout for `simdgroup_float8x8` or `simdgroup_half8x8`
   fragments (`simdgroup_load`, `simdgroup_multiply_accumulate`, `simdgroup_store`). On
   Metal 4, an optional `CooperativeTensor` layout for `mpp::tensor_ops::matmul2d`,
   which uses the M5 Neural Accelerators
   ([WWDC25 session 262](https://developer.apple.com/videos/play/wwdc2025/262/),
   [Metal 4 matmul example](https://github.com/liuliu/example_matmul_metal4)).
4. **`Threadgroup(padding or swizzle)`:** tiles staged in `threadgroup` memory
   (32 KB per threadgroup).

Conversion rule: any mismatch between two register distributions lowers to "write to
threadgroup memory, `threadgroup_barrier(mem_flags::mem_threadgroup)`, read back." Start
with a single pass that assigns layouts greedily from anchors (loads, `dot`, reductions)
and inserts conversions where they disagree. Skip the Triton-style propagation-and-removal
fixpoint until profiles show a need.

---

## 4. Autotuning

`@triton.autotune` (`python/triton/runtime/autotuner.py`) wraps a `JITFunction` in an
`Autotuner`. The following code sample shows the typical usage:

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=2, num_stages=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr,
                  BLOCK_N: tl.constexpr):
    # Kernel body elided.
    pass
```

- **`triton.Config`** holds `kwargs` (meta-parameters injected as constexprs),
  `num_warps`, `num_stages`, `num_ctas`, `maxnreg`, `pre_hook`, and `ir_override`.
- **`key`** lists argument names; the tuple of their values (and dtypes) indexes the
  results cache. A new `M` value triggers a full re-tune, which is a known cost with
  dynamic shapes. The usual workaround is to key on a bucketed value computed in the
  caller.
- **`prune_configs_by`** accepts `early_config_prune(configs, named_args)`, a
  `perf_model`, and `top_k`, to avoid benchmarking everything.
- **`reset_to_zero` and `restore_value`** protect in-place outputs (for example, atomic
  accumulators) across benchmark runs.
- **Benchmarking** uses `triton.testing.do_bench` (or the driver's
  `get_benchmarker()`): it runs warmup iterations, clears L2 by writing a large buffer
  before each run, times with device events, and returns a mean or requested quantiles.
  Each config also costs a full compile, so tuning 20 configs can take tens of seconds.
- **Caching:** results live in memory per process. `cache_results=True` or
  `TRITON_CACHE_AUTOTUNING=1` persists them to the Triton cache dir.
  `TRITON_PRINT_AUTOTUNING=1` prints the winner.
- **`@triton.heuristics({"BLOCK": lambda args: ...})`** computes meta-parameters
  without benchmarking, a cheap alternative that's often good enough.

---

## 5. Interpreter mode (`TRITON_INTERPRET=1`)

`python/triton/runtime/interpreter.py` replaces compilation with direct execution of the
Python function on the CPU:

- `GridExecutor` copies device tensors to host (`_init_args_hst`), then loops
  `for x in range(grid[0]): for y ...: for z ...`, calling `set_grid_idx` and running
  the function once per program. Afterward, `_restore_args_dev` copies results back.
- `_patch_lang()` swaps `tl.*` builtins for implementations backed by
  `InterpreterBuilder`, which implements the same builder API as the MLIR builder
  (`create_fadd`, `create_load`, `create_dot`) on top of numpy. Values are
  `TensorHandle(data: np.ndarray, dtype)`.
- `ASTTransformer` and `FunctionRewriter` rewrite assignments so Python literals become
  tensors.

Because the language is tile-level, each `tl` op is a single numpy op over the whole
tile, with no thread simulation. That's why the interpreter is small and why it works
with `print()` and `pdb`
([debugging guide](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html)).
Limitations include no `bfloat16`, no indirect memory access patterns, no inline
assembly or extern calls, sequential and slow execution, and no modeling of races or
atomic ordering. Tools such as [triton-viz](https://github.com/Deep-Learning-Profiling-Tools/triton-viz)
build on the interpreter to visualize memory access. JAX Pallas offers the same idea as
`pallas_call(..., interpret=True)`.

For Forge, this is the highest-value feature per line of code: numpy or MLX on unified
memory needs no host-device copies, and the interpreter doubles as the reference
implementation for testing codegen.

---

## 6. Backend plug-in architecture and prior Metal attempts

Backends are discovered by `_discover_backends()` in
`python/triton/backends/__init__.py`, through the `triton.backends` entry-point group
(out of tree), or by scanning `triton/backends/<name>/{compiler,driver}.py` (in tree,
copied from `third_party/<name>/backend/` at build time). Each backend is a
`Backend(compiler: Type[BaseBackend], driver: Type[DriverBase])`.

- **`BaseBackend`** (`python/triton/backends/compiler.py`): `supports_target(GPUTarget)`,
  `hash()`, `parse_options(dict)`, `add_stages(stages, options, language)`,
  `load_dialects(ctx)`, `get_module_map()` (maps `tl.math` or libdevice to device
  implementations), and the specialization hooks. `GPUTarget(backend, arch, warp_size)`
  identifies the device.
- **`DriverBase`** (`python/triton/backends/driver.py`): `is_active()`,
  `get_current_target()`, `get_active_torch_device()`, `get_benchmarker()`,
  `map_python_to_cpp_type()`, plus a launcher and utils module.

Metal-related prior art:

- [triton#4824](https://github.com/triton-lang/triton/issues/4824) and
  [discussion #1796](https://github.com/triton-lang/triton/discussions/1796) requested a
  Metal backend; upstream never built one.
- [triton-ext PR #126](https://github.com/triton-lang/triton-ext/pull/126) and
  [PR #127](https://github.com/triton-lang/triton-ext/pull/127) (`triton-ext` is the
  out-of-tree backends repo) add an Apple GPU backend: TTIR to TTGIR, then C++ MLIR passes
  `AccelerateAppleMatmul` (8x8 simdgroup MMA encodings), `StoreShuffleLayout`, and
  `EmitMSL`, which prints MSL source that `xcrun metal` and `metallib` compile. Dispatch
  goes through MPS and an Objective-C++ bridge. It deliberately emits MSL instead of
  targeting AIR, because AIR is undocumented. Earlier AIR-based attempts broke on macOS
  updates, and in-process metallib serialization failed on some macOS versions.
- [triton-msl](https://github.com/bledden/triton-msl) runs unmodified `@triton.jit`
  kernels on M1 and later (Triton 3.7, PyTorch MPS). Its
  [support matrix](https://github.com/bledden/triton-msl/blob/main/docs/SUPPORTED_OPS.md)
  shows the Metal realities: `tt.dot` on `simdgroup_matrix` (fp16, bf16, and fp32 in,
  fp32 accumulate, about 11 TFLOP/s versus about 2.4 TFLOP/s scalar fallback), reductions
  including argmax, restricted atomics (no multi-element-per-thread scatter, some 16-bit
  RMW ops refused), no fp64 or fp8, fp32 subnormals flushed to zero, and unstructured
  control flow refused. Its policy of refusing a kernel loudly instead of miscompiling it
  is worth copying.
- [triton-shared](https://github.com/microsoft/triton-shared) (unmaintained) lowered TTIR
  to Linalg with `PtrAnalysis` and `MaskAnalysis`, recovering structured strided accesses
  from pointer arithmetic. It's evidence that pointer-tile semantics are expensive to
  analyze.

Takeaway: a Triton-compatible Metal backend needs the full MLIR and C++ toolchain, a
pinned Triton version, and TTGIR concepts designed for NVIDIA. That's the main argument
for Forge being its own small Python compiler instead of a Triton backend. Forge can still
stay source-compatible with a Triton subset.

---

## 7. Developments in 2025 and 2026 and their lessons

- **Gluon** (`triton.experimental.gluon`, `@gluon.jit`,
  [overview](https://triton-lang.org/main/gluon/)): the same Python frontend and
  `CodeGenerator` emit TTGIR directly. Users write `gl.BlockedLayout`, `gl.SliceLayout`,
  `gl.allocate_shared_memory`, explicit `gl.convert_layout`, mbarriers, TMA, and warp
  specialization. It exists because compiler-managed pipelining and layouts couldn't keep
  up with asynchronous Hopper and Blackwell hardware
  ([Lei Zhang's analysis](https://www.lei.chat/posts/gluon-explicit-performance/)).
  *Lesson:* design an escape hatch one level down that shares the frontend, but don't make
  it the default.
- **Tensor descriptors and TMA:** block-shaped access through a descriptor
  (`shape`, `strides`, `block_shape`) with automatic out-of-bounds padding replaced
  `make_block_ptr`. *Lesson:* descriptor-style tile loads with implicit bounds handling
  are both easier for users and easier for compilers than pointer tiles plus masks.
- **Linear layouts** ([arXiv:2505.23819](https://arxiv.org/abs/2505.23819)):
  *Lesson:* if Forge's layout system grows past four or five kinds, move to one
  algebraic representation instead of pairwise conversion code.
- **TileLang** ([repo](https://github.com/tile-ai/tilelang),
  [arXiv:2504.17577](https://arxiv.org/abs/2504.17577)): TVM-based. Users write
  `T.Kernel(grid, threads=128)`, allocate with `T.alloc_shared` or `T.alloc_fragment`,
  and use `T.copy`, `T.gemm`, `T.reduce`, `T.Parallel`, and `T.Pipelined(num_stages=3)`.
  Layout and pipeline inference fill in the rest. *Lesson:* explicit memory scopes
  (shared versus fragment) are a good middle ground and map one-to-one onto Metal's
  `threadgroup` versus thread-private storage.
- **cuTile Python and CUDA Tile IR** (NVIDIA, CUDA 13.1, December 2025;
  [cuda-tile](https://github.com/NVIDIA/cuda-tile)): `@ct.kernel`, `ct.bid(axis)`,
  `ct.load(array, index=(pid,), shape=(TILE,))`, and `ct.store(array, index, tile)`, with
  `ct.Constant[int]` tile sizes. Arrays carry shape and strides, and indexing is in tile
  coordinates, not pointers. Tile IR is an open MLIR-based portable bytecode.
  *Lesson:* array-plus-tile-index loads remove most masks and pointer arithmetic, and a
  stable tile IR decouples frontends from codegen.
- **Helion** ([blog](https://pytorch.org/blog/helion/),
  [repo](https://github.com/pytorch/helion)): "PyTorch with tiles." One function mixes
  host code (allocation, shapes) with device loops (`for tile_m, tile_n in
  hl.tile([m, n])`), indexing is `x[tile_m, tile_k]` with implicit masking, and it
  compiles to Triton. The autotuner searches an *implicit* space (block sizes, loop
  orders, L2 grouping, pointer versus block pointer versus descriptor indexing,
  persistent or flat PID mapping, reduction looping) across roughly 1,500 configs in
  about 10 minutes, then prints a `helion.Config(...)` to paste into the decorator. The
  config is applied only at the final codegen step, so parsing and passes run once.
  *Lesson:* let the compiler own tile sizes as tunables, and make "freeze the tuned
  config in source" a first-class workflow.
- **Pallas** ([design](https://docs.jax.dev/en/latest/pallas/design/design.html),
  [grids and BlockSpecs](https://docs.jax.dev/en/latest/pallas/grid_blockspec.html)):
  kernels operate on `Ref`s; `pallas_call(kernel, grid=..., in_specs=[BlockSpec(block_shape,
  index_map)], out_specs=...)` declares which block each program sees, and the framework
  does the copy-in, copy-out, and pipelining. `interpret=True` gives CPU debugging. Pallas
  moved its GPU lowering from Triton to Mosaic GPU. *Lesson:* declarative block mapping
  separates "which tile" from "what compute," which enables automatic staging.

---

## 8. What makes Triton pleasant and what makes it painful

Pleasant:

- NumPy-like tile code with masks. Users think in blocks, not threads, and still get 80
  to 95 percent of expert CUDA on common kernels.
- The compiler handles coalescing, shared memory allocation, barriers, and pipelining.
- `constexpr` turns Python into a macro language; `static_range` and constexpr `if`
  generate specialized code without templates.
- Seamless PyTorch integration: pass tensors, launch with `kernel[grid](...)`.
- One-decorator autotuning and a small set of excellent tutorials.
- An interpreter that supports `pdb`.

Painful:

- **Compile time and cold start.** Each specialization and config compiles through MLIR,
  LLVM, and `ptxas`, often 0.5 to several seconds. Autotuning multiplies this. Model
  startup (vLLM, `torch.compile`) spends minutes compiling; failures block loading
  entirely.
- **Opaque errors.** Frontend type errors are decent, but failures deep in MLIR passes
  surface as `PassManager::run failed` or crashes in layout conversion, with no user-level
  location.
- **Layouts leak.** Users never write layouts but still hit them: performance cliffs from
  hidden `convert_layout` round trips, register spills from large tiles, shared memory
  overflows (`OutOfResources`), and reading TTGIR dumps (`MLIR_ENABLE_DUMP=1`) to see
  why. Gluon is, in part, an admission of this.
- **Silent recompiles and specialization surprises.** The `==1` specialization,
  divisibility flipping between calls, and new autotune keys each trigger compiles and
  occasionally type bugs.
- **Language restrictions.** Power-of-two `arange`, no `break`, limited lists and
  tuples, no early return in dynamic control flow, mandatory `constexpr` on shapes.
- **Numerics.** TF32 by default for fp32 `dot`, and FTZ differences.
- **Debugging on device.** `device_print` output is interleaved per thread, and
  `device_assert` requires `TRITON_DEBUG=1`. The interpreter diverges on bf16 and
  indirect loads.
- **Performance portability.** Configs tuned for H100 are wrong for other devices, and
  tuning spaces are hand-written.

---

## Recommendations for Forge

### Architecture

1. **Write the compiler in pure Python; skip MLIR and LLVM.** Python AST, then a small
   typed SSA tile IR (dataclasses), then five or six passes, then MSL text. Let Apple's
   Metal compiler do register allocation and scheduling. This keeps install at
   `pip install` and keeps compile time in milliseconds on the Forge side.
2. **Compile MSL at runtime and cache the result.** Use `MTLDevice.newLibraryWithSource`
   for fast iteration, or `xcrun metal` and `metallib` for ahead-of-time builds (the
   route triton-ext adopted after in-process serialization problems). Cache the generated
   `.metal` source and `.metallib` under `~/.forge/cache/<sha256>/`. Include in the hash
   the kernel source plus transitive `@forge.jit` dependencies (copy
   `DependenciesFinder`), the specialization key, options, the Forge version, the macOS
   and Metal language versions, and the GPU family.
3. **Keep the launch path thin.** Bind buffers and dispatch through a small native
   extension or PyObjC, accepting PyTorch MPS tensors, MLX arrays, and numpy arrays.
   Unified memory makes zero-copy host arrays practical. Build the argument binder once
   per signature, as Triton's `create_function_from_signature` does.

### Borrow

- **The `@jit` + `kernel[grid](...)` + `grid=lambda meta: ...` launch model**, with
  `forge.cdiv`.
- **`constexpr` as compile-time Python.** Support constexpr `if`, `static_range`, and
  constexpr function arguments; include values in the cache key.
- **Specialization on dtype, constexprs, and 16-byte alignment** (the `D` marker). On
  Metal, alignment gates `float4` and `half4` loads.
- **The P0 op set from section 2**, including `load` and `store` with `mask` and
  `other` for Triton familiarity, plus `dot`, axis reductions, `where`, and math.
- **`reduce` and `associative_scan` with a user combine function** (Welford, online
  softmax).
- **`@autotune(configs, key)` and `@heuristics`**, with `do_bench`-style timing
  (warmup, median over repeats, Metal command buffer GPU timestamps), persistent result
  caching by default, and a printed winning config the user can paste back.
- **A first-class interpreter**, `FORGE_INTERPRET=1` or `interpret=True`, with numpy
  tiles, sequential grid, `pdb`, and `print`. Use it as the differential-testing oracle
  for every op in CI.
- **IR dumps** (`FORGE_DUMP=1` writes tile IR and MSL next to the cache entry) and an
  override mechanism like `TRITON_KERNEL_OVERRIDE` for hand-edited MSL.
- **Loud refusal** for anything unsupported, with the Python source line, as triton-msl
  does. Never miscompile silently.

### Simplify

- **Layouts:** four internal distributions (`Blocked`, `Slice`, `SimdMatrix`,
  `Threadgroup`), one greedy assignment pass, and conversions that always go through
  threadgroup memory. Map `num_warps` to SIMD groups per threadgroup (32 lanes each, up
  to 1,024 threads). Don't expose layouts in v1.
- **Memory access:** add cuTile and descriptor-style `forge.load(tensor, index=(i, j),
  shape=(BM, BN))` on array objects that carry shape and strides, with automatic bounds
  padding. Contiguity then comes from the stride metadata (specialize on `stride == 1`
  for tensor args) instead of Triton's integer `==1` hack and `AxisInfo` inference.
  Keep pointer tiles for gathers and scatters.
- **`dot`:** lower to `simdgroup_matrix` 8x8 MMA with operands staged in threadgroup
  memory. Add a Metal 4 `matmul2d` path later for M5 Neural Accelerators, gated on GPU
  family.
- **Pipelining:** replace `num_stages` software pipelining with plain double-buffered
  threadgroup staging behind a flag, because Apple GPUs have no asynchronous copy engine.
- **Autotuning:** tune `BLOCK_*`, `num_warps`, and elements per thread only. Add
  Helion-style implicit spaces later.

### Drop (for v1)

- Warp specialization, TMA, mbarriers, clusters (`num_ctas`), fp8, `dot_scaled`, fp64,
  inline assembly, and `extern_elementwise`.
- The integer `==1` constexpr specialization.
- `make_block_ptr` (already deprecated upstream).
- An MLIR-based plug-in architecture. A single backend needs one `Backend` protocol
  (`compile(ir) -> source`, `launch(...)`) at most, kept internal until a second target
  exists.
- A Gluon-like explicit layer. Instead, offer a raw-MSL escape hatch (in the style of
  `mx.fast.metal_kernel`) for experts.

### Avoid Triton's pain points

- Report errors at the Python source line from every pass, not only the frontend.
- Warn when a kernel recompiles for the same function more than N times, and print which
  key component changed.
- Surface performance-relevant decisions (threadgroup memory bytes, conversions inserted,
  estimated registers per thread) in a `kernel.explain()` report.
- Relax the power-of-two rule where Metal allows it, or pad automatically, and document
  the rule precisely.
- Default `dot` to fp32 accumulation with no silent precision reduction, and document
  Apple's FTZ behavior.
