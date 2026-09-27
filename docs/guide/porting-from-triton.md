# Porting from Triton

Enceladus follows Triton's programming model and API closely, so most Triton kernels
port by changing the imports and the host code. This page lists what carries over
unchanged, what differs, and what Enceladus doesn't support, and then ports a Triton
matmul step by step.

## What maps directly

The following parts of Triton work the same way in Enceladus:

- The decorator and launch syntax: `@enceladus.jit`, `kernel[grid](...)`, a grid tuple
  or `grid = lambda meta: (...)`, and `tl.constexpr` parameters.
- The language namespace. `import enceladus.language as tl` gives you `tl.program_id`,
  `tl.arange`, `tl.load` and `tl.store` with `mask` and `other`, `tl.where`,
  `tl.dot`, reductions (`sum`, `max`, `min`, `argmax`, `argmin`, and `reduce`), scans
  (`cumsum` and `associative_scan`), atomics, the common math functions,
  `tl.static_range`, `tl.static_assert`, `tl.static_print`, `tl.device_print`, and
  `tl.device_assert`. For the full list, see
  [Language reference](language-reference.md).
- Tensor descriptors: `tl.make_tensor_descriptor(base, shape, strides, block_shape)`
  with `desc.load` and `desc.store`. Enceladus needs no host-side allocator for them.
- Type promotion and broadcasting rules, and tile methods such as `x.to(tl.float16)`
  and `x_ptr.dtype.element_ty`.
- Helper functions decorated with `@enceladus.jit`, which the compiler inlines.
- Autotuning: `enceladus.autotune`, `enceladus.Config`, and `enceladus.heuristics`,
  with `key`, `prune_configs_by` (`early_config_prune`, `perf_model`, and `top_k`),
  `reset_to_zero`, `restore_value`, and `pre_hook`.
- `enceladus.cdiv` and `enceladus.next_power_of_2`, like `triton.cdiv` and
  `triton.next_power_of_2`.
- `do_not_specialize`, and specialization on integers divisible by 16 and on 16-byte
  aligned pointers.
- The interpreter: `ENCELADUS_INTERPRET=1` replaces `TRITON_INTERPRET=1`. Enceladus's
  interpreter also supports `bfloat16`.

Triton hints that have no meaning on Apple GPUs are accepted and ignored, so you don't
have to remove them: `num_stages`, the pipelining hints of `tl.range`, and the
`cache_modifier`, `eviction_policy`, `volatile`, `input_precision`, `allow_tf32`, and
`max_num_imprecise_acc` arguments.

## What differs

The following behavior differs from Triton.

### Launch options

- `num_warps` counts SIMD groups. An Apple GPU SIMD group has 32 threads, like an NVIDIA
  warp, so `num_warps=4` still means 128 threads per program. `num_warps` must be a
  power of two from 1 to 32.
- `dot_warps=(WM, WN)` is a launch option and a `Config` field with no Triton
  equivalent. It arranges the program's SIMD groups as a WM x WN grid over each
  `tl.dot` result. By default, Enceladus stacks SIMD groups along M. Flash attention
  runs fastest with `dot_warps=(num_warps, 1)`, so that each SIMD group owns whole rows.
- `dot_backend` selects how `tl.dot` compiles: `"simdgroup"` (`simdgroup_matrix`
  instructions), `"mpp"` (Metal 4 `matmul2d`), or `"auto"`, the default. For details,
  see [Choose a `tl.dot` backend](performance.md#choose-a-tldot-backend).
- A program is a Metal threadgroup. Tile sizes, register limits, and 32 KB of
  threadgroup memory per program follow from Apple GPU hardware.

### Types

- Enceladus has no `float64`, no FP8 types, and no TF32. `float32` `tl.dot` computes in
  full `float32`. Passing a `float64` array raises an error that suggests `float32`.
- `tl.dot` takes `float16`, `bfloat16`, or `float32` operands of the same dtype, and
  accumulates in `float32` or `float16`. Integer `tl.dot` isn't supported. The K block
  must be a multiple of 8.
- Tile dimensions are limited to 65,536, and a tile must fit in 256 registers per
  thread.

### Atomics

- Memory ordering is relaxed only. `sem` accepts only `None` and `"relaxed"`, and
  `scope` accepts `None`, `"gpu"`, and `"cta"`.
- `float16` and `bfloat16` `atomic_add` isn't supported. Accumulate in `float32`.
- Of the 64-bit atomics, only `uint64` `atomic_max` and `atomic_min` are supported,
  and only when you don't use the returned value.
- 8-bit and 16-bit atomics, `float32` `max` and `min`, and `atomic_cas` run as
  compare-and-swap loops on the surrounding 32-bit word, which is slower than a native
  atomic.
- `tl.atomic_and`, `tl.atomic_or`, and `tl.atomic_xor` are supported on integers.

### Host side

- Kernels take `enceladus.Tensor` objects, NumPy arrays, PyTorch tensors on the `mps`
  device, and MLX arrays. PyTorch launches run on PyTorch's MPS stream; for the ordering
  rules of the other kinds, see
  [Streams and synchronization](programming-model.md#streams-and-synchronization).
- Autotuning keys include the argument dtypes automatically, and Enceladus saves
  results on disk, per GPU, in `~/.cache/enceladus/autotune/`.
  `ENCELADUS_PRINT_AUTOTUNING=1` replaces `TRITON_PRINT_AUTOTUNING=1`.
- `enceladus.testing.do_bench` times kernels with GPU timestamps. The function that you
  pass must launch on `enceladus.Tensor` arguments.
- Instead of inline assembly, Enceladus offers `enceladus.metal_kernel(source, name)`,
  which launches hand-written MSL.

## What isn't supported

The compiler refuses the following Triton features with an `enceladus.CompilationError`:

- Block pointers: `tl.make_block_ptr`, `tl.advance`, and the `boundary_check` and
  `padding_option` arguments of `tl.load` and `tl.store`. Use tensor descriptors.
- `while` loops, `break`, and `continue`.
- `tl.multiple_of` and `tl.max_contiguous` hints, `tl.inline_asm_elementwise`,
  `tl.dot_scaled`, and the `libdevice` functions.
- `tl.join` and `tl.split` on tiles of pointers. Join or split the integer offsets
  instead.
- Other `tl` functions that aren't in the [Language reference](language-reference.md),
  such as `tl.sort`, `tl.flip`, `tl.gather`, `tl.histogram`, and `tl.rand`.
- The `**` operator on runtime values. Use `tl.exp2` and `tl.log2`, or multiply.
- Warp specialization, TMA, and clusters, which have no Apple GPU equivalent.

## Port a matmul

The following Triton kernel is a matmul in the style of Triton's tutorial, with pointer
tiles:

<!-- snippet: skip -->
```python
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=rk[None, :] < K - k * BK, other=0.0)
        b = tl.load(b_ptrs, mask=rk[:, None] < K - k * BK, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk
    c = acc.to(tl.float16)
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(rm[:, None] < M) & (rn[None, :] < N))


a = torch.randn((512, 256), device="cuda", dtype=torch.float16)
b = torch.randn((256, 384), device="cuda", dtype=torch.float16)
c = torch.empty((512, 384), device="cuda", dtype=torch.float16)
grid = (triton.cdiv(512, 64), triton.cdiv(384, 64))
matmul_kernel[grid](a, b, c, 512, 384, 256, a.stride(0), a.stride(1),
                    b.stride(0), b.stride(1), c.stride(0), c.stride(1),
                    BM=64, BN=64, BK=32, num_warps=4, num_stages=3)
```

To port it, make the following changes:

1. Replace `import triton` and `triton.language` with `enceladus` and
   `enceladus.language`, and `@triton.jit` with `@enceladus.jit`.
1. Move the tensors to the `mps` device instead of `cuda`.
1. Mask the M and N edges of the loads too. When M or N isn't a multiple of the block
   size, this Triton kernel reads past the last row or column and relies on the store
   mask. Enceladus doesn't check bounds, so an unmasked load outside an array reads
   whatever memory is there.
1. Replace `triton.cdiv` with `enceladus.cdiv`. You can keep `num_stages`; Enceladus
   ignores it.
1. Optional: switch the loads to tensor descriptors. Descriptors check their own
   bounds, and they let `tl.dot` load fragments straight from device memory, which is
   faster than pointer tiles on Apple GPUs.

The following program is the ported kernel with tensor descriptors:

```python
import torch

import enceladus
import enceladus.language as tl


@enceladus.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_bk, stride_cm,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    a = tl.make_tensor_descriptor(a_ptr, [M, K], [stride_am, 1], [BM, BK])
    b = tl.make_tensor_descriptor(b_ptr, [K, N], [stride_bk, 1], [BK, BN])
    c = tl.make_tensor_descriptor(c_ptr, [M, N], [stride_cm, 1], [BM, BN])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc = tl.dot(a.load([pid_m * BM, k]), b.load([k, pid_n * BN]), acc)
    c.store([pid_m * BM, pid_n * BN], acc.to(tl.float16))


a = torch.randn((512, 256), device="mps", dtype=torch.float16)
b = torch.randn((256, 384), device="mps", dtype=torch.float16)
c = torch.empty((512, 384), device="mps", dtype=torch.float16)
grid = (enceladus.cdiv(512, 64), enceladus.cdiv(384, 64))
matmul_kernel[grid](a, b, c, 512, 384, 256, a.stride(0), b.stride(0), c.stride(0),
                    BM=64, BN=64, BK=32, num_warps=4)
torch.testing.assert_close(c, (a.float() @ b.float()).half(), rtol=1e-2, atol=1e-2)
print("ported matmul matches PyTorch")
```

The descriptors require a unit stride in the innermost dimension, so the port passes
only the row strides. To tune the block sizes for your shapes, see
[Autotune kernels](performance.md#autotune-kernels).
