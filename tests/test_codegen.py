"""Tests for the compiler passes and MSL codegen that differential tests can't see."""

import numpy as np
import pytest
from conftest import check_kernel

import enceladus
import enceladus.language as tl
from enceladus.compiler import ir
from enceladus.compiler.passes.axis_info import AxisAnalysis, contiguous_order


@enceladus.jit
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


@enceladus.jit
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
    from enceladus.compiler.pipeline import compile_module

    assert compile_module(first).source == compile_module(second).source


def test_axis_info_tracks_contiguity_and_order():
    @enceladus.jit
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


@enceladus.jit
def _dot_shared_acc(a_ptr, b_ptr, out_ptr, iters, N: tl.constexpr, CASE: tl.constexpr):
    r = tl.arange(0, N)
    a = tl.load(a_ptr + r[:, None] * N + r[None, :])
    b = tl.load(b_ptr + r[:, None] * N + r[None, :])
    acc = tl.dot(a, b)
    if CASE == "loop":
        res = tl.zeros((N, N), tl.float32)
        for _ in range(iters):
            res = tl.dot(a, b, acc)
        acc = res
    elif CASE == "nested":
        # `acc` is carried by the outer loop but reused unchanged by every inner iteration.
        for _ in range(iters):
            res = tl.zeros((N, N), tl.float32)
            for _ in range(iters):
                res = tl.dot(a, b, acc)
            acc = res
    else:
        # The accumulator is a view that shares storage with `acc`, which is read later.
        acc = tl.dot(a, b, tl.trans(tl.trans(acc))) + acc
    tl.store(out_ptr + r[:, None] * N + r[None, :], acc)


@pytest.mark.parametrize(("case", "iters", "scale"), [
    ("loop", 1, 2), ("loop", 2, 2), ("loop", 3, 2), ("nested", 2, 3), ("nested", 3, 4),
    ("view", 1, 3),
])  # fmt: skip
def test_dot_updates_its_accumulator_in_place_only_when_nothing_rereads_it(rng_np, case, iters,
                                                                           scale):  # fmt: skip
    # Each case gives the accumulator a single use, the dot, while its storage is read again:
    # by the next loop iteration, or through another value.
    a = rng_np.integers(-2, 3, (16, 16)).astype(np.float32)
    b = rng_np.integers(-2, 3, (16, 16)).astype(np.float32)

    def run(a, b):
        out = np.empty_like(a)
        _dot_shared_acc[(1,)](a, b, out, iters, N=16, CASE=case)
        return out

    check_kernel(run, (a, b), lambda a, b: scale * (a @ b))


@enceladus.jit
def _desc_load_at(x_ptr, out_ptr, M, N, o0, o1, B: tl.constexpr):
    d = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [B, B])
    r = tl.arange(0, B)
    tl.store(out_ptr + r[:, None] * B + r[None, :], d.load([o0, o1]))


@enceladus.jit
def _desc_store_at(x_ptr, out_ptr, M, N, o0, o1, B: tl.constexpr):
    d = tl.make_tensor_descriptor(out_ptr, [M, N], [N, 1], [B, B])
    r = tl.arange(0, B)
    d.store([o0, o1], tl.load(x_ptr + r[:, None] * B + r[None, :]))


@enceladus.jit
def _desc_dot_at(x_ptr, out_ptr, M, N, o0, o1, B: tl.constexpr):
    # Both operands read fragments straight from device memory; the second one transposed.
    d = tl.make_tensor_descriptor(x_ptr, [M, N], [N, 1], [B, B])
    acc = tl.dot(d.load([o0, o1]), tl.trans(d.load([o1, o0])))
    r = tl.arange(0, B)
    tl.store(out_ptr + r[:, None] * B + r[None, :], acc)


def _block(x, o0, o1, b):
    """Returns the b x b block of `x` at (o0, o1), zero outside `x`."""
    out = np.zeros((b, b), x.dtype)
    rows = np.arange(o0, o0 + b)
    cols = np.arange(o1, o1 + b)
    rm, cm = (rows >= 0) & (rows < x.shape[0]), (cols >= 0) & (cols < x.shape[1])
    out[np.ix_(rm, cm)] = x[np.ix_(rows[rm], cols[cm])]
    return out


@pytest.mark.parametrize("offsets", [(-3, -5), (-16, 0), (10, 12), (2, 4)])
@pytest.mark.parametrize("kind", ["load", "store", "dot"])
def test_descriptor_accesses_outside_the_tensor_are_masked(rng_np, kind, offsets):
    # The 20 x 24 tensor sits between guard zones of the same buffer, so an access past
    # either edge reads the guard value or overwrites it.
    m, n, b, guard = 20, 24, 16, -7.0
    o0, o1 = offsets
    x = rng_np.integers(-2, 3, (m, n)).astype(np.float32)
    src = rng_np.integers(-2, 3, (b, b)).astype(np.float32)

    def guarded(t):
        buf = np.full(3 * m * n, guard, np.float32)
        buf[m * n : 2 * m * n] = t.ravel()
        return buf

    def run(x, src):
        if kind == "store":
            buf = guarded(np.full((m, n), guard, np.float32))
            _desc_store_at[(1,)](src, buf[m * n : 2 * m * n], m, n, o0, o1, B=b)
            return buf
        out = np.empty((b, b), np.float32)
        k = _desc_load_at if kind == "load" else _desc_dot_at
        k[(1,)](guarded(x)[m * n : 2 * m * n], out, m, n, o0, o1, B=b)
        return out

    def reference(x, src):
        if kind == "load":
            return _block(x, o0, o1, b)
        if kind == "dot":
            return _block(x, o0, o1, b) @ _block(x, o1, o0, b).T
        t = np.full((m + 2 * b, n + 2 * b), guard, np.float32)
        t[b + o0 : 2 * b + o0, b + o1 : 2 * b + o1] = src
        return guarded(t[b : b + m, b : b + n])

    check_kernel(run, (x, src), reference)


@pytest.fixture
def rng_np():
    return np.random.default_rng(0)


SCALE = 3.0
OFFSET = 1.0


@enceladus.jit
def _add_offset(x):
    return x + OFFSET


@enceladus.jit
def _scale_by_globals(x_ptr, out_ptr):
    offs = tl.arange(0, 16)
    tl.store(out_ptr + offs, _add_offset(tl.load(x_ptr + offs) * SCALE))


@pytest.mark.parametrize("name", ["SCALE", "OFFSET"])  # OFFSET is read by an inlined helper
def test_reassigned_global_constant_recompiles(monkeypatch, name):
    x = np.arange(16, dtype=np.float32)
    out = np.empty_like(x)
    _scale_by_globals[(1,)](x, out)
    np.testing.assert_array_equal(out, x * SCALE + OFFSET)
    monkeypatch.setitem(globals(), name, 5.0)
    _scale_by_globals[(1,)](x, out)
    np.testing.assert_array_equal(out, x * SCALE + OFFSET)


@enceladus.jit
def _signed_zeros(x_ptr, out_ptr):
    offs = tl.arange(0, 16)
    x = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, 1.0 / (x * 0.0))
    tl.store(out_ptr + 16 + offs, 1.0 / (x * -0.0))


def test_cse_keeps_signed_zero_constants_apart():
    def run(x):
        out = np.empty(32, np.float32)
        _signed_zeros[(1,)](x, out)
        return out

    def reference(x):
        with np.errstate(divide="ignore"):
            return np.concatenate([1 / (x * 0.0), 1 / (x * -0.0)]).astype(np.float32)

    check_kernel(run, [np.ones(16, np.float32)], reference)
