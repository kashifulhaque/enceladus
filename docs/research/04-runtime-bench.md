# Forge host runtime: Metal bridge benchmarks

These are measurements from one machine, taken to help choose how Forge's Python
runtime talks to Metal. Every number here came from code in `runtime_bench/`. Where a
result is noisy or surprising, the report says so.

## Summary

- **Recommendation:** build Forge's runtime as a small Objective-C++ core with a plain
  C ABI, and expose it to Python through a nanobind extension. Put a stream layer on
  top of it: launches go into one open command buffer, which Forge commits every 64 or
  so dispatches or when the host syncs.
- **Why nanobind:** the bridge only matters for the cost of encoding each dispatch.
  nanobind encodes one dispatch in 0.17 µs, close to the 0.11 µs of pure C. The other
  bridges take 0.53 µs (ctypes with a packed record), 1.15 µs (ctypes), 1.8 µs (raw
  `objc_msgSend`), and 2.9 µs (PyObjC). The bridge makes no difference to the
  synchronous round trip. That round trip costs about 95 µs whichever bridge you use,
  and the GPU driver sets that floor.
- **The design choice that matters most is batching, not the bridge.** With one
  command buffer per launch, no bridge gets below about 10 to 11 µs per launch, pure C
  included. The prototype stream runs at 1.1 µs per launch sustained, and the Python
  launcher call inside it takes 0.5 µs. Measured on the same machine, the
  `torch.mps.compile_shader` path sustains 3.4 µs per launch and `mx.fast.metal_kernel`
  sustains 3.7 µs.
- **Interop is already solved by DLPack.** Both torch MPS and MLX export
  `__dlpack__` with `kDLMetal` (8). The `DLTensor.data` field holds the
  `id<MTLBuffer>` pointer, and `byte_offset` holds the offset of a view. Forge can bind
  these tensors without copying them.

## Environment

The following table lists the machine and software versions:

| Item | Value |
|---|---|
| Machine | MacBook Pro, Apple M4 Pro, 16-core GPU, 24 GB (`applegpu_g16s`) |
| OS and tools | macOS 27.0 (26A428), Xcode 27.0 (27A266a), Apple clang 21.0.0, uv 0.12.17 |
| Python | CPython 3.11.14 |
| Packages | pyobjc-core and pyobjc-framework-metal 12.2.2, nanobind 3.1.0, numpy 2.4.6, torch 2.14.0, mlx 0.32.2 (mlx-metal 0.32.2) |
| Compilation | Runtime only, through `newLibraryWithSource:options:error:`. The offline `metal` tool isn't installed. |

## Method

- The test kernel is a Triton-style `vecadd`: 1,024 floats, a grid of 4 threadgroups
  of 256 threads, and `n` passed with `setBytes`. It keeps GPU time small (about 2 µs)
  so that the host side dominates.
- Each dispatch benchmark ran 3,000 iterations after warm-up. Each bridge ran in its
  own process for 3 rounds, with the bridges interleaved. Tables show the median over
  rounds of each round's median and p90.
- Every bridge passed correctness checks: `vecadd`, a 257×257×257 f32 `matmul`
  (checked against numpy), and no-copy buffers in both directions.
  `test_correctness.py` runs these checks.
- **Noise warning:** the GPU also drives the display, and WindowServer was using 45%
  CPU during the runs. In one run, synchronous round trips moved into a slow mode for a
  while, with a median of about 1 ms and a p90 of about 3 ms. After that, the normal
  distribution came back (about 90 µs median, about 110 µs p90). The first 100 ms or so
  of GPU work in a process also runs slower until the GPU clocks up. The published
  numbers come from interleaved and warmed-up runs.

## Experiment 1: bridge options

The benchmark tested five bridges, all behind the same interface in `bridges.py`:

| Name | What it is |
|---|---|
| `pyobjc` | `pyobjc-framework-Metal`, with an `objc.autorelease_pool()` around each launch. |
| `ctypes+dylib` | ctypes calls into `libforge_rt.dylib`, an ObjC++ library with a C ABI. The launch passes a ctypes array of buffer pointers and `uint32[3]` arrays for the grid. |
| `ctypes+dylib packed` | The same dylib. Python builds each launch as one `struct.pack` record and passes it as a single pointer. |
| `nanobind` | A nanobind extension, `forge_nb`, that wraps the same C ABI and adds Python classes. |
| `ctypes+objc_msgSend` | Raw calls into the Objective-C runtime in the style of tinygrad, with one typed `CFUNCTYPE` for each selector signature. No compiled code. |

`build.sh` builds both native artifacts with plain clang in about 2 seconds. It
doesn't need scikit-build-core.

### Per-launch host overhead

The following table shows per-launch costs in µs, as median / p90:

| Metric | ctypes+dylib | ctypes packed | **nanobind** | pyobjc | raw objc_msgSend |
|---|---|---|---|---|---|
| FFI floor: one trivial call (property read) | 0.118 / 0.122 | 0.117 / 0.124 | **0.027 / 0.028** | 0.160 / 0.166 | 0.188 / 0.196 |
| Sync launch round trip (encode, commit, wait) | 95.5 / 112.6 | 95.0 / 109.7 | 94.2 / 107.1 | 107.4 / 123.7 | 99.4 / 113.6 |
| GPU time of that dispatch (`GPUEndTime - GPUStartTime`) | 2.1 / 2.4 | 2.1 / 2.4 | 2.1 / 2.3 | 2.1 / 2.3 | 2.1 / 2.3 |
| Async launch call (own command buffer, no wait) | 7.2 / 23.0 | 6.5 / 24.5 | 5.2 / 25.8 | 12.8 / 23.1 | 8.4 / 20.4 |
| Async sustained, µs per launch (3,000 launches) | 12.0 | 11.3 | 11.3 | 15.7 | 12.6 |

The following table shows only the host cost to encode N dispatches into one encoder
and commit them, without waiting. It's the median µs per dispatch, and it's the
clearest measure of Python-side overhead:

| N | ctypes+dylib | ctypes packed | **nanobind** | pyobjc | raw objc_msgSend | Pure C (native loop) |
|---|---|---|---|---|---|---|
| 1 | 7.42 | 6.42 | **4.79** | 13.29 | 8.29 | 4.0 |
| 10 | 2.03 | 1.26 | **0.72** | 3.87 | 2.59 | 0.54 |
| 100 | 1.15 | 0.58 | **0.22** | 2.90 | 1.83 | 0.17 |
| 1000 | 1.15 | 0.53 | **0.17** | 2.88 | 1.80 | 0.11 |

Subtracting the C loop gives the Python cost for each dispatch: about 0.06 µs for
nanobind, 0.4 µs for ctypes packed, 1.0 µs for ctypes, 1.7 µs for raw `objc_msgSend`,
and 2.8 µs for PyObjC.

The following table shows the full round trip for a batch: encode N dispatches,
commit, and wait. Values are wall-clock µs, as median / p90:

| N | ctypes+dylib | ctypes packed | nanobind | pyobjc | raw objc_msgSend | Pure C | GPU time |
|---|---|---|---|---|---|---|---|
| 1 | 98.7 / 112.3 | 93.9 / 111.1 | 94.8 / 108.0 | 105.6 / 133.8 | 101.0 / 115.3 | 91.9 / 106.9 | 2.1 / 2.3 |
| 10 | 121.5 / 148.9 | 114.6 / 128.9 | 105.5 / 123.4 | 140.4 / 173.3 | 124.2 / 150.5 | 102.3 / 124.8 | 11.2 / 13.0 |
| 100 | 328.8 / 410.1 | 274.5 / 293.4 | 234.5 / 244.0 | 497.0 / 566.4 | 374.5 / 419.9 | 213.7 / 240.8 | 103.8 / 119.8 |
| 1000 | 2372 / 2529 | 1764 / 1865 | 1384 / 1479 | 4340 / 4715 | 3390 / 3653 | 1308 / 1437 | 1098 / 1226 |

At N=1000, nanobind lands within 6% of pure C, and GPU time accounts for most of the
wall time. PyObjC and raw `objc_msgSend` become CPU-bound at 3 to 3.3 times that time.

### Native floors: wait strategy, per-command-buffer cost, and encoder type

These numbers come from loops written entirely in C, in `bench_native.py`. The first
table compares ways to wait for one dispatch to finish. Values are µs, as median /
p90, with samples pooled from 6 interleaved rounds:

| Wait strategy | Round trip |
|---|---|
| `waitUntilCompleted` | 95.8 / 124.8 |
| Spin on `cb.status` | 88.9 / 101.8 |
| Completion handler and `dispatch_semaphore` | 97.7 / 117.8 |
| `encodeSignalEvent` and a spin on `MTLSharedEvent.signaledValue` | **79.3 / 95.1** |
| `commandBufferWithUnretainedReferences` and `waitUntilCompleted` | 94.3 / 118.4 |

With one dispatch per command buffer and no waiting, the queue sustains 9.7 to 11.3 µs
per launch in pure C, for 100, 1,000, and 10,000 launches. That's a cost for each
command buffer, and no bridge avoids it.

The following table compares encoder types for N dispatches in one encoder. All
dispatches write the same buffers:

| Encoder | Host encode per dispatch at N=1000 | GPU per dispatch at N=10 | at N=100 | at N=1000 |
|---|---|---|---|---|
| Serial (default, automatic hazard tracking) | 0.114 µs | 1.15 µs | 1.04 µs | 1.11 µs |
| `MTLDispatchTypeConcurrent` | 0.112 µs | 0.30 µs | 0.16 µs | 0.16 µs |
| Concurrent with `memoryBarrierWithScope:Buffers` between dispatches | 0.36 µs | 1.4 to 1.8 µs | 1.3 µs | 1.2 to 1.4 µs |

A dependent dispatch costs about 1.1 µs of GPU time for the drain and refill. Manual
barriers aren't cheaper than serial hazard tracking, and they cost 3 times as much on
the CPU. Use a concurrent encoder only for runs of independent dispatches.

### Metal 4 command model (optional experiment)

`native/forge_mtl4.mm` tests the Metal 4 command model: `MTL4CommandQueue`, a reused
`MTL4CommandBuffer` and `MTL4CommandAllocator`, an `MTL4ArgumentTable` using
`setAddress:`, a residency set, and a wait on an `MTLSharedEvent`. It uses the classic
`MTLComputePipelineState` and produces correct results. The following table compares
it with the classic queue in the same process. Values are wall-clock µs, as median /
p90, plus host encode cost per dispatch:

| N | Classic serial | Classic concurrent | MTL4, no barrier (spin) | MTL4 with dispatch barrier (spin) |
|---|---|---|---|---|
| 1 | 97.0 / 110.6 (4.08) | 99.7 / 110.5 (4.42) | **76.4 / 93.1** (7.3) | 76.7 / 93.5 (7.0) |
| 10 | 109.9 / 124.7 (0.54) | 100.9 / 107.0 (0.58) | 88.1 / 101.4 (0.96) | 118.6 / 143.5 (0.90) |
| 100 | 217.1 / 241.5 (0.17) | 134.9 / 145.5 (0.17) | 123.7 / 149.5 (0.13) | 457.6 / 554.9 (0.24) |
| 1000 | 1370 / 1467 (0.11) | 371.5 / 416.7 (0.11) | 402.2 / 505.6 (0.09) | 1173 / 1413 (0.11) |

Metal 4 cuts about 20 µs off the synchronous floor, mostly by waiting on an event. Its
encode cost per dispatch at large N is slightly lower, at 0.09 µs compared with
0.11 µs. Barriers between dependent dispatches, which Metal 4 requires because it
doesn't track hazards, cost as much as serial tracking or more, and the measurements
were noisy. Metal 4 isn't worth the extra complexity for the first version. The C ABI
leaves room to add it later.

## Experiment 1(i) and 1(ii): MSL compile and pipeline creation

`bench_compile.py` measures these costs. It puts a nonce into the kernel names so
that the first compile is guaranteed to miss every cache. It runs each case in a
fresh process, repeats the process to test cross-process caching, and repeats the
compile inside one process. The small kernel is `vecadd` (14 lines). The large kernel
is a 173-line tiled `matmul` file with templates and `simdgroup_matrix`, which yields
6 kernels (f32, f16, bf16, simdgroup, and scalar variants).

The following table shows the results in ms:

| Measurement | vecadd | matmul (6 functions) |
|---|---|---|
| First compiler use in a process (connection startup, before any real work) | about 33 ms, once per process | Same |
| `newLibraryWithSource`, cold (unique source) | 2.6 to 3.1 | 39.4 to 41.8 |
| Pipeline state, cold, for each function | 3.6 to 3.9 | 12.3 to 26.7 (about 120 total) |
| `newLibraryWithSource`, identical source, new process | 0.46 to 0.52 | 0.05 to 0.06 |
| Pipeline state, new process (Metal disk cache) | 0.41 to 0.43 | 0.02 to 0.04 each |
| `newLibraryWithSource`, identical source, same process | 0.003 to 0.03 | 0.005 to 0.03 |
| Same source with only a changed **comment**, new process | 2.7 to 3.2 (miss) | 40 to 42 (miss) |
| Pipeline state after a comment-only change | 0.07 to 0.18 (hit) | 0.02 to 0.1 (hit) |
| `MTLCreateSystemDefaultDevice` and queue, including import | 24 to 28 (ctypes and nanobind), 71 (PyObjC) | Not applicable |

The Metal cache has two levels, and both persist across processes. The cache lives
in `$(getconf DARWIN_USER_CACHE_DIR)/com.apple.metal/`:

- The front end, which turns source into AIR, is keyed on the exact source text, so
  even a changed comment misses.
- The back end, which turns AIR into GPU code, is keyed on the IR, so a comment-only
  change still hits.

To test parallel compilation, the benchmark compiled 8 distinct `matmul` variants
(8 × 6 pipelines) as an autotuning workload:

- Serially, the 8 variants took 1,245 to 1,283 ms.
- On 8 Python threads, they took 184 to 191 ms, a speedup of 6.7 to 6.9 times.

The results were the same for nanobind (which explicitly releases the GIL) and PyObjC
(which releases it during Objective-C calls). The device reports
`maximumConcurrentCompilationTaskCount = 12`.

## Experiment 2: zero-copy interop

### numpy and `newBufferWithBytesNoCopy`

The following list describes the results:

- A page-aligned numpy allocation wraps with no copy. GPU writes appear in numpy, and
  numpy writes appear on the GPU. `bench_util.page_aligned_empty` shows the pattern.
- **Surprise:** on macOS 27, `newBufferWithBytesNoCopy` also accepted pointers that
  weren't page-aligned (offsets of 8, 64, and 4096 bytes) and lengths that weren't a
  multiple of the 16 KB page, such as 1,000 and 5,000 bytes. `contents` matched the
  pointer, and GPU writes appeared in numpy. The SDK header doesn't document this
  requirement one way or the other. Forge must not rely on this behavior. Allocate
  page-aligned memory, and copy when the call returns `nil`.
- numpy's own allocator returned page-aligned memory for every live allocation of
  100 KB or more. Below that size, 2% to 34% of allocations were page-aligned.
- The following table compares the cost to wrap, allocate, or copy a buffer. Values
  are µs, as median / p90:

| Size | No-copy wrap | `newBufferWithLength` | Allocate and `memcpy` |
|---|---|---|---|
| 64 KB | 2.8 / 3.3 | 3.6 / 4.8 | 8.9 / 10.3 |
| 1 MB | 3.0 / 11.7 | 4.1 / 4.7 | 63 / 71 |
| 64 MB | 9.5 / 11.9 | 24.6 / 29.6 | 3,774 / 3,860 |

### torch MPS (2.14.0)

The following list describes the results:

- `t.untyped_storage().data_ptr()` returns the `id<MTLBuffer>`, which is an
  `AGXG16XFamilyBuffer` that conforms to `MTLBuffer`. It uses shared storage mode
  (`storageMode` 0), so `contents` is valid, and it belongs to the same `MTLDevice`
  object as `MTLCreateSystemDefaultDevice()`.
- `t.data_ptr()` on a view equals the storage pointer plus the byte offset: `x[100:]`
  added 400 bytes. That value **isn't** an object pointer. Bind the pair
  (`storage.data_ptr()`, `storage_offset * itemsize`) with `setBuffer:offset:`.
- When Forge wrapped torch's buffer (with `CFRetain`) and dispatched on its own queue,
  the result was correct. `torch.mps.synchronize()` costs 0.08 µs when nothing is
  pending, and the full foreign launch round trip costs 105 / 120 µs.
- **Hazard:** without `torch.mps.synchronize()`, a Forge kernel reading a tensor right
  after a torch `fill_` got a stale result in 1 of 50 trials. Torch and Forge queues
  don't order against each other.
- `torch.mps.compile_shader` exists. It compiled `vecadd` in 9.6 ms. A synchronous
  call with `synchronize` takes 98 / 111 µs. An async call takes 1.17 / 1.42 µs, but
  3,000 async calls took 6.7 ms more to drain, which works out to about 3.4 µs per
  launch sustained. For comparison, an async `torch.add` call takes 0.92 µs.
- DLPack: `__dlpack_device__()` returns `(kDLMetal=8, 0)`, and `to_dlpack` succeeds.
  The capsule's `data` is the `id<MTLBuffer>` and `byte_offset` is the view offset,
  40 bytes for `x[10:]`. `np.from_dlpack` rejects the capsule with "Unsupported device".

### MLX (0.32.2)

The following list describes the results:

- `mx.array` implements the buffer protocol. `np.array(a, copy=False)` is a writeable,
  page-aligned view whose pointer equals the underlying `MTLBuffer.contents`.
- `__dlpack_device__()` returns `(8, 0)`. The capsule's `data` is the `id<MTLBuffer>`,
  not the contents pointer. `torch.from_dlpack(mx_array)` returns an `mps:0` tensor that
  shares the same `MTLBuffer`, and a torch write is visible in MLX.
- Forge successfully wrapped MLX memory by calling `newBufferWithBytesNoCopy` on the
  buffer-protocol pointer, and dispatched on it with a correct result. Using the
  DLPack `MTLBuffer` directly is cleaner.
- `mx.fast.metal_kernel` works. A synchronous round trip (`mx.eval` each call) takes
  108 to 110 µs median, with a p90 of 126 µs. Building the graph for one call takes
  0.79 µs. A chain of N dependent kernels, evaluated once, costs 107 µs per launch at
  N=1, 16 µs at N=10, 5.1 µs at N=100, and 3.66 µs at N=1000.

### Prototype Forge stream

`bench_stream.py` measures a prototype stream: `kernel[grid](x, y, out, n)` with
Python argument binding and a precompiled `struct.Struct` for scalars. Launches go
into one open command buffer, which the stream commits every *k* dispatches. The
following table shows the results, measured after warm-up:

| Flush every *k* dispatches | Sustained µs per launch | Python call, median / p90 |
|---|---|---|
| 1 | 11.2 to 11.5 | 6.0 / 24.5 |
| 4 | 2.65 | 1.1 / 6.0 |
| 16 | 1.28 | 0.63 / 3.3 |
| 64 | **1.12** | 0.54 / 0.67 |
| 256 | 1.14 | 0.46 / 0.63 |
| 1024 | 1.19 | 0.46 / 0.54 |

At *k* of 64 or more, the GPU sets the limit (1.1 µs for each serial dispatch), not
the host.

## Experiment 3: GPU timing

`bench_gputime.py` compares `MTLCommandBuffer` `GPUStartTime` and `GPUEndTime` with
host wall clock for synchronous launches through nanobind. The following table shows
the results in µs, as median / p90:

| Kernel | Wall clock | GPU | Wall minus GPU | Throughput |
|---|---|---|---|---|
| `vecadd` n=1K | 125 / 187 | 7.2 / 8.3 | 118 | Not applicable |
| `vecadd` n=1M | 146 / 161 | 22.6 / 23.0 | 124 | 557 GB/s (12 MB fits in the system-level cache) |
| `vecadd` n=16M | 982 / 1,085 | 824 / 833 | 158 | 244 GB/s (close to DRAM) |
| `matmul` f32 256³ (simdgroup 64×64) | 253 / 259 | 125 / 127 | 128 | 0.27 TFLOPS |
| `matmul` f32 1024³ | 3,537 / 3,906 | 3,365 / 3,687 | 173 | 0.64 TFLOPS |
| `matmul` f32 2048³ | 29,246 / 30,727 | 29,045 / 30,536 | 201 | 0.59 TFLOPS |

The test `matmul` kernel is a correctness and compile-time exercise, not a tuned
kernel. The following list describes the timing findings:

- Wall clock includes a fixed synchronization cost of about 100 to 200 µs, which grows
  slightly with kernel length. Use GPU timestamps for kernel timing and autotuning.
- This short run had fewer warm-up iterations, so the GPU was clocked lower: `vecadd`
  n=1K took 7 µs of GPU time here and 2 µs in the dispatch runs. Autotuning must warm
  up first.
- Counter sampling: `supportsCounterSampling` is true only for `AtStageBoundary`, and
  false for dispatch, draw, and blit boundaries. The only counter set is `timestamp`.
  To time individual kernels inside one command buffer, you therefore need a separate
  encoder or command buffer for each kernel, or Metal 4 counter heaps (not tested).

## Experiment 4: device properties and language versions

The following table lists the device properties:

| Property | Value |
|---|---|
| `name` and `architecture` | Apple M4 Pro, `applegpu_g16s` |
| `maxThreadsPerThreadgroup` | 1024 × 1024 × 1024 (1,024 threads total) |
| `maxThreadgroupMemoryLength` | 32,768 bytes |
| `recommendedMaxWorkingSetSize` | 17.76 GiB |
| `maxBufferLength` | 13.32 GiB |
| `hasUnifiedMemory` | true |
| `argumentBuffersSupport` | Tier 2 (1) |
| `supportsFamily` | Apple7, 8, and 9: yes. **Apple10: no.** Mac2, Common3, Metal3, and **Metal4**: yes. |
| `maximumConcurrentCompilationTaskCount` | 12 (`shouldMaximizeConcurrentCompilation` = false) |
| Pipeline `threadExecutionWidth` | 32 for every kernel |
| Pipeline `maxTotalThreadsPerThreadgroup` | 1,024 for every kernel |
| Pipeline `staticThreadgroupMemoryLength` | 4 KB to 8 KB for the `matmul` kernels |

The following table shows which features the runtime compiler accepts at each
language version:

| Feature | Default | 2.4 | 3.0 | 3.1 | 3.2 | 4.0 | 4.1 | 4.2 |
|---|---|---|---|---|---|---|---|---|
| Trivial kernel | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No: `invalid value 'metal4.2'` |
| `bfloat` | Yes | No | No | Yes | Yes | Yes | Yes | No |
| `simdgroup_matrix<half>` | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No |
| `simdgroup_matrix<bfloat>` | Yes | No | No | Yes | Yes | Yes | Yes | No |
| `MetalPerformancePrimitives` `mpp::tensor_ops::matmul2d` | **No** | No | No | No | No | **Yes** | **Yes** | No |

The following list describes the language-version findings:

- The `MTLCompileOptions` default `languageVersion` is **3.2**, even though 4.0 and 4.1
  are available. The MPP header compiles to nothing below 4.0, and the compiler then
  reports `use of undeclared identifier 'mpp'`. You must set 4.0 or later explicitly.
- `matmul2d` works at run time. The probe builds `tensor<device half,
  dextents<int32_t,2>, tensor_inline>` values from raw `device half*` buffer pointers,
  so it needs no host-side `MTLTensor` objects. It uses a 64×32 tile with
  `execution_simdgroups<4>` and 128 threads. It produced a correct 64×64 result with
  f16 inputs and f32 outputs, in 7 µs of GPU time. A cold compile takes 92 to 121 ms.
- The first cold compile at a non-default language version took 180 to 240 ms, which
  is probably a one-time load of that version's standard library. Later processes
  found it in the cache.

## Recommendation

Use an **Objective-C++ core with a C ABI, called from a nanobind extension**. Use
`native/forge_rt.{h,mm}` and `native/forge_nb.mm` as the starting point. The following
list gives the reasons:

1. It has the lowest per-dispatch host cost: 0.17 µs, compared with 0.53 µs to 2.9 µs
   for the alternatives, and an FFI floor of 27 ns. That cost is what limits Forge
   once it batches launches. The synchronous round trip of about 95 µs is the same for
   every bridge.
2. It releases the GIL around compilation and waiting, so compiling autotuning
   variants on threads gives a 6.8-times speedup.
3. The C ABI keeps options open. The same dylib works through ctypes, which is useful
   as a fallback or for debugging. A Metal 4 backend can sit behind the same functions.
4. The build is easy. Plain clang builds everything in about 2 seconds, and
   scikit-build-core can produce wheels. The trade-off is that you ship a compiled
   extension for each Python ABI, or use nanobind's stable-ABI mode, which wasn't
   tested here.

Consider an alternative only if a hard constraint rules out compiled code. In that
case, use raw `objc_msgSend` through ctypes (1.8 µs per dispatch). PyObjC is the
slowest option measured: 2.9 µs per dispatch and 71 ms of startup.

The runtime design the data supports has the following parts:

- **A stream:** keep one open `MTLCommandBuffer` with a serial compute encoder. Append
  each dispatch to it, and commit every 64 dispatches or so, or on sync, host read, or
  a foreign-framework boundary. Use a concurrent encoder only for independent
  dispatches.
- **Synchronization:** have the command buffer signal an `MTLSharedEvent` and wait on
  the event. That was about 15 µs faster than `waitUntilCompleted`.
- **Tensor ingestion:** accept `__dlpack__` with `kDLMetal` and bind
  (`id<MTLBuffer>`, `byte_offset`). For torch, call `torch.mps.synchronize()` before a
  Forge launch that reads torch-written data. For CPU arrays, wrap page-aligned memory
  with no copy.
- **Caching:** keep an in-process cache of source hash to `MTLLibrary` and pipeline
  states, and rely on Metal's disk cache across processes. Keep generated source
  deterministic: no timestamps or other varying comments, because they defeat the
  front-end cache.
- **Compilation:** always pass an explicit `languageVersion`. Use 4.0 when a kernel
  uses MPP, and at least 3.1 for `bfloat`.

## Gotchas

- `MTLCompileOptions.languageVersion` defaults to 3.2. `mathMode` needs macOS 15, and
  `fastMathEnabled` is deprecated.
- A torch view's `data_ptr()` isn't an `MTLBuffer` pointer. Use the storage pointer
  plus an offset. The benchmark API binds offset 0, so the production API needs an
  offset for each buffer.
- Forge's queue and torch's MPS stream don't order against each other. The benchmark
  read stale data in 2% of trials without a sync.
- Measure only after warm-up. The GPU clocks down when idle. In fresh processes, the
  first configuration ran up to 4 times slower, and the sync latency went through a
  slow mode (about 1 ms) that coincided with display activity.
- One command buffer per launch caps throughput at about 90,000 launches per second
  (about 11 µs each), even in C. When more than 64 command buffers are in flight,
  `commandBuffer` blocks, which explains the long p90 of the async call.
- Every Objective-C call that returns an autoreleased object, such as `commandBuffer`
  or `computeCommandEncoder`, needs an autorelease pool. That means
  `objc.autorelease_pool()` in PyObjC, `objc_autoreleasePoolPush` and
  `objc_autoreleasePoolPop` with raw ctypes, and `@autoreleasepool` in C. Without a
  pool, these objects leak.
- On arm64, you must call `objc_msgSend` through an exact, non-variadic prototype,
  meaning one `CFUNCTYPE` for each signature. Passing `MTLSize` by value works with
  ctypes.
- In PyObjC, `buf.contents()` returns an `objc.varlist`. Call
  `.as_buffer(buf.length())` to get a memoryview. `import Metal` alone takes 58 ms.
- When you link `nb_combined.o`, compile it with the same `-mmacosx-version-min`, or
  the linker warns about a version mismatch.
- The `MTL4CommitFeedback` handler, which provides GPU timestamps, runs asynchronously
  after the event signals. Don't block waiting for it inside a loop that's being
  timed. An earlier version of the benchmark did, and that added idle gaps that
  distorted the results.
- Accepting unaligned no-copy buffers is undocumented behavior on macOS 27. Keep a
  fallback path.

## Snippets from the recommended approach

The following code shows the C ABI core that encodes one dispatch (from
`native/forge_rt.mm`, compiled with `-fobjc-arc`):

```objc
static inline void encode_one(id<MTLComputeCommandEncoder> enc, void *pso,
                              void *const *bufs, int nbufs, const void *bytes,
                              size_t nbytes, int bytes_index,
                              const uint32_t grid[3], const uint32_t tg[3]) {
    [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)pso];
    for (int i = 0; i < nbufs; ++i)
        [enc setBuffer:(__bridge id<MTLBuffer>)bufs[i] offset:0 atIndex:i];
    if (nbytes) [enc setBytes:bytes length:nbytes atIndex:bytes_index];
    [enc dispatchThreadgroups:MTLSizeMake(grid[0], grid[1], grid[2])
        threadsPerThreadgroup:MTLSizeMake(tg[0], tg[1], tg[2])];
}
```

The following code shows the nanobind layer, which collects buffers without
allocating and releases the GIL only for slow calls (from `native/forge_nb.mm`):

```cpp
static int collect_bufs(nb::handle seq, void **out, int cap) {
    PyObject *fast = PySequence_Fast(seq.ptr(), "bufs must be a sequence");
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    PyObject **items = PySequence_Fast_ITEMS(fast);
    for (Py_ssize_t i = 0; i < n; ++i) out[i] = nb::inst_ptr<Buffer>(items[i])->p;
    Py_DECREF(fast);
    return (int)n;
}

nb::class_<Batch>(m, "Batch")
    .def("dispatch", [](Batch &b, Pipeline &pso, nb::handle bufs, nb::bytes args,
                        int args_index, const Dim3 &grid, const Dim3 &tg) {
        void *bp[31]; int n = collect_bufs(bufs, bp, 31);
        uint32_t g[3], t[3]; to_arr(grid, g); to_arr(tg, t);
        fr_batch_dispatch(b.b, pso.p, bp, n, args.c_str(), args.size(),
                          args_index, g, t);
    });
m.def("compile", [](Device &d, const std::string &src, uint32_t ver, bool fast) {
    char err[8192]; void *lib;
    { nb::gil_scoped_release nogil;  // lets threads compile in parallel
      lib = fr_library_new(d.p, src.c_str(), ver, fast, err, sizeof err); }
    if (!lib) throw std::runtime_error(std::string("MSL compile failed: ") + err);
    return new Library(lib);
});
```

The following code shows the Python stream and launcher (from `bench_stream.py`),
which sustain 1.1 µs per launch:

```python
class Stream:
    def __init__(self, queue, flush_every=64):
        self.q, self.flush_every, self.batch, self.pending = queue, flush_every, None, 0

    def dispatch(self, pso, bufs, args, args_index, grid, tg):
        if self.batch is None:
            self.batch = forge_nb.batch_begin(self.q)
        self.batch.dispatch(pso, bufs, args, args_index, grid, tg)
        self.pending += 1
        if self.pending >= self.flush_every:
            self.flush()

    def flush(self, wait=False):
        if self.batch is not None:
            r, self.batch, self.pending = self.batch.end(wait), None, 0
            return r
```

The following code reads the `id<MTLBuffer>` and offset from a torch MPS or MLX
DLPack capsule (from `dlpack_util.py`):

```python
cap = tensor.__dlpack__()
info = inspect_capsule(cap)  # reads DLTensor through ctypes
assert info["device_type"] == 8  # kDLMetal
mtl_buffer_ptr, offset = info["data"], info["byte_offset"]
# CFRetain(mtl_buffer_ptr), then setBuffer:offset:.
# Keep `tensor` alive for as long as Forge holds the buffer.
```

## Files

The benchmark code is in `runtime_bench/`. The following table lists the files:

| File | Purpose |
|---|---|
| `kernels.py` | MSL sources: `vecadd`, the 173-line `matmul`, and feature probes, including MPP |
| `bridges.py` | The five bridge implementations |
| `native/forge_rt.{h,mm}`, `native/forge_nb.mm`, `native/forge_mtl4.mm`, `build.sh` | Native code and the build script |
| `bench_dispatch.py`, `summarize_dispatch.py` | Dispatch overhead for each bridge (`results/dispatch_tables.md`) |
| `bench_native.py`, `bench_mtl4.py` | Native floors and the Metal 4 comparison |
| `bench_compile.py` | Compile and cache timing |
| `bench_gputime.py`, `bench_nocopy.py`, `bench_stream.py`, `bench_startup.py` | GPU timing, no-copy buffers, the stream prototype, and startup cost |
| `interop_torch.py`, `interop_mlx.py`, `dlpack_util.py` | torch and MLX interop |
| `probe_device.py` | Device properties and language-version probes |
| `test_correctness.py`, `test_nocopy_alias.py` | Correctness checks |
| `results/*.json` | Raw data |
