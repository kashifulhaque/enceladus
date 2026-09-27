"""Tests for the compiler passes and MSL codegen that differential tests can't see."""

import json
import types
from pathlib import Path

import numpy as np
import pytest
from conftest import check_kernel

import enceladus
import enceladus.language as tl
from enceladus.compiler import ir
from enceladus.compiler.passes.axis_info import AxisAnalysis, contiguous_order
from enceladus.runtime import cache


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


@enceladus.jit
def _versioned_dot(a_ptr, b_ptr, out_ptr, M, N, K, lo, hi, koff, STEP: tl.constexpr,
                   TRANS_B: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                   BK: tl.constexpr):  # fmt: skip
    pid_n, pid_m = tl.program_id(0), tl.program_id(1)
    da = tl.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [BM, BK])
    do = tl.make_tensor_descriptor(out_ptr, [M, N], [N, 1], [BM, BN])
    acc = tl.zeros((BM, BN), tl.float32)
    if TRANS_B:
        db = tl.make_tensor_descriptor(b_ptr, [N, K], [K, 1], [BN, BK])
        for k in range(lo, hi, STEP):
            acc = tl.dot(da.load([pid_m * BM, k + koff]), tl.trans(db.load([pid_n * BN, k])), acc)
    else:
        db = tl.make_tensor_descriptor(b_ptr, [K, N], [N, 1], [BK, BN])
        for k in range(lo, hi, STEP):
            acc = tl.dot(da.load([pid_m * BM, koff + k]), db.load([k, pid_n * BN]), acc)
    do.store([pid_m * BM, pid_n * BN], acc)


def _tile(x, o0, o1, r, c):
    """Returns the r x c block of `x` at (o0, o1), zero outside `x`."""
    out = np.zeros((r, c), x.dtype)
    rows, cols = np.arange(o0, o0 + r), np.arange(o1, o1 + c)
    rm, cm = (rows >= 0) & (rows < x.shape[0]), (cols >= 0) & (cols < x.shape[1])
    out[np.ix_(rm, cm)] = x[np.ix_(rows[rm], cols[cm])]
    return out


# (M, N, K, lo, hi, koff, STEP, TRANS_B): aligned; ragged M, N, and K; a shifted K offset
# that makes the last block ragged; negative offsets in early iterations; a step smaller
# than the block; and K smaller than one block.
VERSIONED_CASES = [
    (64, 64, 64, 0, 64, 0, 16, False), (40, 48, 47, 0, 47, 0, 16, True),
    (64, 32, 64, 0, 64, 1, 16, False), (32, 64, 40, -16, 40, 3, 16, True),
    (32, 32, 48, 0, 48, 0, 8, True), (40, 32, 5, 0, 5, 0, 16, False),
]  # fmt: skip


@pytest.mark.parametrize("case", VERSIONED_CASES, ids=str)
def test_edge_versioned_dot_loops(rng_np, case):
    m, n, k, lo, hi, koff, step, trans_b = case
    bm, bn, bk = 32, 32, 16
    a = rng_np.integers(-3, 4, (m, k)).astype(np.float32)
    b = rng_np.integers(-3, 4, (n, k) if trans_b else (k, n)).astype(np.float32)
    grid = (-(-n // bn), -(-m // bm))

    def run(a, b):
        out = np.zeros((m, n), np.float32)
        _versioned_dot[grid](a, b, out, m, n, k, lo, hi, koff, STEP=step, TRANS_B=trans_b,
                             BM=bm, BN=bn, BK=bk)  # fmt: skip
        return out

    def reference(a, b):
        out = np.zeros((grid[1] * bm, grid[0] * bn), np.float32)
        bt = b.T if trans_b else b
        for pm in range(grid[1]):
            for pn in range(grid[0]):
                for kk in range(lo, hi, step):
                    out[pm * bm:(pm + 1) * bm, pn * bn:(pn + 1) * bn] += \
                        _tile(a, pm * bm, kk + koff, bm, bk) @ _tile(bt, kk, pn * bn, bk, bn)
        return out[:m, :n]

    check_kernel(run, (a, b), reference, atol=0, rtol=0)
    # The loop runs as an unmasked main loop, a checked K tail, and a checked edge version.
    ck = _versioned_dot.warmup(a, b, np.zeros((m, n), np.float32), m, n, k, lo, hi, koff,
                               STEP=step, TRANS_B=trans_b, BM=bm, BN=bn, BK=bk)  # fmt: skip
    assert ck.msl.count("simdgroup_multiply_accumulate") == 3


@enceladus.jit
def _dot_after_write(a_ptr, b_ptr, out_ptr, iters, N: tl.constexpr, CASE: tl.constexpr):
    da = tl.make_tensor_descriptor(a_ptr, [N, N], [N, 1], [N, N])
    db = tl.make_tensor_descriptor(b_ptr, [N, N], [N, 1], [N, N])
    do = tl.make_tensor_descriptor(out_ptr, [N, N], [N, 1], [N, N])
    r = tl.arange(0, N)
    a_ptrs = a_ptr + r[:, None] * N + r[None, :]
    b_ptrs = b_ptr + r[:, None] * N + r[None, :]
    b = db.load([0, 0])
    acc = tl.zeros((N, N), tl.float32)
    for _ in range(iters):
        # Each case adds 1 to A or B in memory after loading it, which the dot must not
        # see. The dot is the only use of each load, as a direct operand requires.
        a = da.load([0, 0])
        if CASE == "desc_store":
            da.store([0, 0], tl.load(a_ptrs) + 1.0)
        elif CASE == "store":
            tl.store(a_ptrs, tl.load(a_ptrs) + 1.0)
        elif CASE == "atomic":
            tl.atomic_add(a_ptrs, 1.0)
        elif CASE == "if":
            if iters > 0:
                tl.store(a_ptrs, tl.load(a_ptrs) + 1.0)
        if CASE == "loop":
            acc = tl.dot(a, b, acc)
            tl.store(b_ptrs, tl.load(b_ptrs) + 1.0)  # before the next iteration's dot
        else:
            acc = tl.dot(a, db.load([0, 0]), acc)
    do.store([0, 0], acc)


@pytest.mark.parametrize("backend", ["simdgroup", "mpp"])
@pytest.mark.parametrize("case", ["desc_store", "store", "atomic", "if", "loop"])
def test_dot_reads_its_operands_before_later_writes(rng_np, case, backend):
    # A dot can read a descriptor operand from device memory at the dot instead of at the
    # load, which is wrong when memory is written in between. One iteration except for
    # "loop": threads that reload what other threads stored would race.
    a = rng_np.integers(-2, 3, (32, 32)).astype(np.float32)
    b = rng_np.integers(-2, 3, (32, 32)).astype(np.float32)
    iters = 3 if case == "loop" else 1

    def run(a, b):
        out = np.empty_like(a)
        _dot_after_write[(1,)](a.copy(), b.copy(), out, iters, N=32, CASE=case,
                               dot_backend=backend)  # fmt: skip
        return out

    check_kernel(run, (a, b), lambda a, b: iters * (a @ b))


@enceladus.jit
def _nan_compare(x_ptr, y_ptr, bits_ptr, sel_ptr, N: tl.constexpr):
    r = tl.arange(0, N)
    x = tl.load(x_ptr + r)
    y = tl.load(y_ptr + r)
    bits = ((x == y).to(tl.int32) | (x != y).to(tl.int32) << 1 | (x < y).to(tl.int32) << 2
            | (x <= y).to(tl.int32) << 3 | (x > y).to(tl.int32) << 4
            | (x >= y).to(tl.int32) << 5 | (x != x).to(tl.int32) << 6)  # fmt: skip
    s = tl.load(x_ptr + 3)  # a scalar NaN
    tl.store(bits_ptr + r, bits)
    tl.store(bits_ptr + N, (s != s).to(tl.int32) + (s == s).to(tl.int32) * 2)
    tl.store(sel_ptr + r, tl.where(x == x, x, -1.0))


@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_float_comparisons_follow_ieee_nan_rules(rng_np, dtype):
    # The default relaxed math mode lets Metal assume no NaNs, which folds `x != x` to false.
    n = 64
    x = rng_np.integers(-3, 4, n).astype(dtype)
    y = rng_np.integers(-3, 4, n).astype(dtype)
    x[[3, 20, 41]] = np.nan
    y[[20, 30]] = np.nan
    x[[7, 8]], y[[7, 9]] = np.inf, -np.inf

    def run(x, y):
        bits, sel = np.empty(n + 1, np.int32), np.empty(n, dtype)
        _nan_compare[(1,)](x, y, bits, sel, N=n)
        return bits, sel

    def reference(x, y):
        preds = [x == y, x != y, x < y, x <= y, x > y, x >= y, x != x]
        bits = sum(p.astype(np.int32) << i for i, p in enumerate(preds))
        return np.append(bits, 1).astype(np.int32), np.where(x == x, x, -1).astype(dtype)

    with np.errstate(invalid="ignore"):
        check_kernel(run, (x, y), reference)


@enceladus.jit
def _nan_rows(x_ptr, out_ptr, max_ptr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    x = tl.load(x_ptr + rm[:, None] * N + tl.arange(0, N)[None, :])
    tl.store(out_ptr + rm, tl.argmax(x, 1))
    tl.store(out_ptr + M + rm, tl.argmin(x, 1))
    tl.store(max_ptr + rm, tl.max(x, 1))
    tl.store(max_ptr + M + rm, tl.min(x, 1))


# Rows of 32 to 1024 elements over 1 to 8 SIMD groups, so the combines run within a thread,
# across lanes, and across SIMD groups.
@pytest.mark.parametrize("m, n, num_warps", [(8, 32, 1), (4, 256, 4), (2, 1024, 8), (16, 64, 4)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_compiled_argmax_skips_nan_like_max(rng_np, m, n, num_warps, dtype):
    x = rng_np.integers(0, 4, (m, n)).astype(dtype)  # small integers make ties likely
    x[rng_np.random((m, n)) < 0.2] = np.nan
    x[0] = np.nan  # an all-NaN row gives index 0, and NaN from tl.max
    x[1, 0] = np.nan

    def run(x):
        idx, ext = np.empty(2 * m, np.int32), np.empty(2 * m, dtype)
        _nan_rows[(1,)](x, idx, ext, M=m, N=n, num_warps=num_warps)
        return idx, ext

    def reference(x):
        clean = np.nan_to_num(x.astype(np.float64), nan=np.inf), np.nan_to_num(
            x.astype(np.float64), nan=-np.inf)  # fmt: skip
        # The first index of the NaN-ignoring extreme.
        idx = np.concatenate([np.argmax(clean[1], 1), np.argmin(clean[0], 1)])
        idx[[0, m]] = 0
        return idx.astype(np.int32), np.concatenate([np.fmax.reduce(x, 1), np.fmin.reduce(x, 1)])

    # The interpreter gets the same semantics in another change; this checks compiled code.
    check_kernel(run, (x,), reference, modes=("compiled",))


def test_debug_builds_map_msl_lines_to_source_lines(monkeypatch):
    from pathlib import Path

    from enceladus.compiler.pipeline import compile_module

    x = np.zeros((16, 256), np.float32)
    lines = Path(__file__).read_text().splitlines()
    line = next(i + 1 for i, s in enumerate(lines) if "x - tl.max(x, axis=1)" in s)
    comment = f"// test_codegen.py:{line}\n"
    monkeypatch.delenv("ENCELADUS_DEBUG", raising=False)
    assert comment not in compile_module(_row_center.ir(x, x, N=256, M=16)).source
    monkeypatch.setenv("ENCELADUS_DEBUG", "1")
    assert comment in compile_module(_row_center.ir(x, x, N=256, M=16)).source


@pytest.fixture
def rng_np():
    return np.random.default_rng(0)


SCALE = 3.0
OFFSET = 1.0
TUP = (2.0, 4.0)
FACTOR = 0.5
ZERO = 0.0
_consts = types.ModuleType("_consts")  # stands in for an imported module of constants
_consts.BIAS = 1.0


def _py_factor():  # a plain Python helper that the kernel calls at compile time
    return FACTOR * 2.0


@enceladus.jit
def _add_offset(x):
    return x + OFFSET


@enceladus.jit
def _sub_offset(x):
    return x - OFFSET


_consts.add_offset = _add_offset


@enceladus.jit
def _scale_by_globals(x_ptr, out_ptr):
    offs = tl.arange(0, 16)
    x = tl.load(x_ptr + offs)
    y = _consts.add_offset(x * SCALE) + TUP[1] * _py_factor() + _consts.BIAS
    tl.store(out_ptr + offs, y)
    tl.store(out_ptr + 16 + offs, 1.0 / ((x + 1.0) * ZERO))


def _scale_by_globals_reference(x):
    offset = OFFSET if _consts.add_offset is _add_offset else -OFFSET
    y = x * SCALE + offset + TUP[1] * FACTOR * 2.0 + _consts.BIAS
    with np.errstate(divide="ignore"):
        return np.concatenate([y, np.float32(1.0) / ((x + 1) * np.float32(ZERO))])


def _start_new_process(kernel):
    """Drops a kernel's in-memory state, as a new process starts without it."""
    kernel._cache_key = None
    kernel._deps = ()
    kernel._compiled.clear()


@pytest.mark.parametrize("new_process", [False, True], ids=["same_process", "new_process"])
@pytest.mark.parametrize(
    ("owner", "name", "value"),
    [
        (None, "SCALE", 5.0),
        (None, "OFFSET", 5.0),  # read by a helper reached through a module attribute
        (None, "TUP", (2.0, 5.0)),
        (None, "FACTOR", 5.0),  # read by a plain Python helper
        (None, "ZERO", -0.0),  # equal to 0.0, but 1 / -0.0 is -inf
        (_consts, "BIAS", 5.0),
        (_consts, "add_offset", _sub_offset),
    ],
    ids=["global", "helper_global", "tuple", "python_helper", "signed_zero", "module_attr",
         "module_jit_helper"],
)  # fmt: skip
def test_changed_dependency_recompiles(monkeypatch, tmp_path, owner, name, value, new_process):
    monkeypatch.setenv("ENCELADUS_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("ENCELADUS_INTERPRET", "0")
    x = np.arange(16, dtype=np.float32)
    out = np.empty(32, np.float32)
    _scale_by_globals[(1,)](x, out)  # builds the in-memory state
    _start_new_process(_scale_by_globals)
    _scale_by_globals[(1,)](x, out)  # writes the disk cache entry
    np.testing.assert_array_equal(out, _scale_by_globals_reference(x))
    if owner is None:
        monkeypatch.setitem(globals(), name, value)
    else:
        monkeypatch.setattr(owner, name, value)
    if new_process:  # a new process that finds the old entry in the disk cache
        _start_new_process(_scale_by_globals)
    _scale_by_globals[(1,)](x, out)
    np.testing.assert_array_equal(out, _scale_by_globals_reference(x))


@enceladus.jit
def _reciprocal(out_ptr, C: tl.constexpr):
    offs = tl.arange(0, 16)
    tl.store(out_ptr + offs, 1.0 / (tl.full((16,), 1.0, tl.float32) * C))


def test_float_constexprs_are_keyed_by_bit_pattern(monkeypatch):
    monkeypatch.setenv("ENCELADUS_INTERPRET", "0")
    out = np.empty(16, np.float32)
    got, compiled = [], []
    for c in (0.0, -0.0, float("nan"), float("nan")):
        _reciprocal[(1,)](out, C=c)
        got.append(out[0])
        compiled.append(len(_reciprocal._compiled))
    np.testing.assert_array_equal(got, [np.inf, -np.inf, np.nan, np.nan])
    assert compiled[3] == compiled[2]  # the second NaN launch reuses the first one's kernel


def test_corrupt_disk_cache_entry_is_recompiled(monkeypatch, tmp_path):
    monkeypatch.setenv("ENCELADUS_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("ENCELADUS_INTERPRET", "0")
    out = np.empty(16, np.float32)
    _reciprocal[(1,)](out, C=4.0)
    _start_new_process(_reciprocal)
    _reciprocal[(1,)](out, C=4.0)  # writes the disk cache entry
    (meta,) = tmp_path.glob("*/meta.json")
    meta.write_text(meta.read_text()[:20])  # as if a writer died partway through
    _start_new_process(_reciprocal)
    _reciprocal[(1,)](out, C=4.0)
    np.testing.assert_array_equal(out, 0.25)
    assert json.loads(meta.read_text())["name"]  # rewritten


def test_compiler_hash_covers_every_compiler_and_language_file(monkeypatch):
    root = Path(enceladus.__file__).parent
    files = [*root.glob("compiler/**/*.py"), *root.glob("compiler/**/*.metal"),
             *root.glob("language/*.py")]  # fmt: skip
    base = cache.compiler_hash.__wrapped__()
    read_bytes = Path.read_bytes
    for f in files:
        monkeypatch.setattr(Path, "read_bytes", lambda p, f=f: read_bytes(p) + b"#" * (p == f))
        assert cache.compiler_hash.__wrapped__() != base, f.relative_to(root)


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
