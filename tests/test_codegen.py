"""Tests for the compiler passes and MSL codegen that differential tests can't see."""

import numpy as np
import pytest
from conftest import check_kernel

import tegula
import tegula.language as tl
from tegula.compiler import ir
from tegula.compiler.passes.axis_info import AxisAnalysis, contiguous_order


@tegula.jit
def _add_transposed(x_ptr, y_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
    y = tl.load(y_ptr + rn[:, None] * M + rm[None, :])  # y is N x M
    tl.store(out_ptr + rm[:, None] * N + rn[None, :], x + tl.trans(y))


def _run_add_transposed(x, y):
    out = np.empty_like(x)
    _add_transposed[(1,)](x, y, out, M=x.shape[0], N=x.shape[1])
    return out


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_layout_conversion_through_threadgroup_memory(rng_np, dtype):
    x = rng_np.standard_normal((32, 64)).astype(dtype)
    y = rng_np.standard_normal((64, 32)).astype(dtype)
    check_kernel(_run_add_transposed, (x, y), lambda a, b: (a + b.T).astype(dtype))
    ck = _add_transposed.warmup(x, y, x, M=32, N=64)
    assert "threadgroup" in ck.msl and ck.threadgroup_memory_bytes > 0


def test_threadgroup_memory_overflow_names_the_line(rng_np):
    x = np.zeros((128, 128), np.float32)
    with pytest.raises(tegula.CompilationError) as e:
        _add_transposed.warmup(x, x, x, M=128, N=128)
    msg = str(e.value)
    assert "threadgroup memory" in msg and "smaller blocks" in msg
    assert "x + tl.trans(y)" in msg


@tegula.jit
def _row_center(x_ptr, out_ptr, N: tl.constexpr, M: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    offs = rm[:, None] * N + rn[None, :]
    x = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, x - tl.max(x, axis=1)[:, None] + tl.sum(x, axis=0)[None, :])


def test_reduce_then_broadcast_back_needs_no_conversion(rng_np):
    x = rng_np.standard_normal((16, 256)).astype(np.float32)

    def run(a):
        out = np.empty_like(a)
        _row_center[(1,)](a, out, N=256, M=16)
        return out

    check_kernel(run, (x,), lambda a: a - a.max(1, keepdims=True) + a.sum(0, keepdims=True),
                 atol=1e-4)  # fmt: skip
    ck = _row_center.warmup(x, x, N=256, M=16)
    assert "buf[" not in ck.msl  # no layout-conversion exchange, only reduction scratch


def test_descriptor_matmul_reads_fragments_from_device_memory():
    from conftest import load_example

    ex = load_example("04_matmul")
    a = np.zeros((128, 128), np.float16)
    ck = ex.matmul_desc_kernel.warmup(a, a, a, 128, 128, 128, 128, 128, 128, BM=64, BN=64,
                                      BK=32)  # fmt: skip
    assert "simdgroup_multiply_accumulate" in ck.msl
    # No staging arrays and no barriers.
    assert ck.threadgroup_memory_bytes == 0 and "threadgroup_barrier" not in ck.msl
    # The pointer-tile variant stages its operands through threadgroup memory instead.
    ck = ex.matmul_kernel.warmup(a, a, a, 128, 128, 128, 128, 1, 128, 1, 128, 1, BM=32, BN=32,
                                 BK=32)  # fmt: skip
    assert "threadgroup_barrier" in ck.msl and ck.threadgroup_memory_bytes > 0


def test_generated_msl_is_deterministic():
    x = np.zeros((16, 256), np.float32)
    first = _row_center.ir(x, x, N=256, M=16)
    second = _row_center.ir(x, x, N=256, M=16)
    from tegula.compiler.pipeline import compile_module

    assert compile_module(first).source == compile_module(second).source


def test_axis_info_tracks_contiguity_and_order():
    @tegula.jit
    def k(p, stride_m, stride_n, BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.arange(0, BM)
        rn = tl.arange(0, BN)
        tl.store(p + rm[:, None] * stride_m + rn[None, :] * stride_n, 0.0)

    # Column-major access (stride_m == 1) makes dimension 0 the fastest.
    m = k.ir(ir.PointerType(ir.f32), 1, 64, BM=32, BN=16)
    store = next(op for op in m.walk() if op.name == "store")
    info = AxisAnalysis(m).get(store.operands[0])
    assert info.contiguity == (32, 1)
    assert contiguous_order(info) == (0, 1)
    # Row-major access keeps the default order, and a divisible pointer stays aligned.
    m = k.ir(np.zeros(4096, np.float32), 64, 1, BM=32, BN=16)  # NumPy data is 16-byte aligned
    store = next(op for op in m.walk() if op.name == "store")
    info = AxisAnalysis(m).get(store.operands[0])
    assert info.contiguity == (1, 16) and contiguous_order(info) == (1, 0)
    assert info.divisibility[1] >= 16


@pytest.fixture
def rng_np():
    return np.random.default_rng(0)
