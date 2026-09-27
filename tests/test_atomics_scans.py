"""Differential tests for atomics (`tl.atomic_*`) and scans (`tl.cumsum`, `tl.associative_scan`)."""

import ml_dtypes
import numpy as np
import pytest
from conftest import check_kernel, execution_mode, load_example

import enceladus
import enceladus.language as tl

# Parametrized kernels recompile per case by design.
pytestmark = pytest.mark.filterwarnings("ignore:.*has been compiled")

# ---------------------------------------------------------------------------
# Atomics
# ---------------------------------------------------------------------------


@enceladus.jit
def _atomic_kernel(mem_ptr, idx_ptr, cmp_ptr, val_ptr, old_ptr, n, KIND: tl.constexpr,
                   BLOCK: tl.constexpr):  # fmt: skip
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    # atomic_cas has no mask, so masked lanes point at an unused slot at the end.
    p = mem_ptr + tl.load(idx_ptr + offs, mask=m, other=n)
    v = tl.load(val_ptr + offs, mask=m)
    if KIND == "add":
        old = tl.atomic_add(p, v, mask=m)
    elif KIND == "max":
        old = tl.atomic_max(p, v, mask=m)
    elif KIND == "min":
        old = tl.atomic_min(p, v, mask=m)
    elif KIND == "xchg":
        old = tl.atomic_xchg(p, v, mask=m)
    elif KIND == "and":
        old = tl.atomic_and(p, v, mask=m)
    elif KIND == "or":
        old = tl.atomic_or(p, v, mask=m)
    elif KIND == "xor":
        old = tl.atomic_xor(p, v, mask=m)
    else:
        old = tl.atomic_cas(p, tl.load(cmp_ptr + offs, mask=m), v)
    tl.store(old_ptr + offs, old, mask=m)


_BF16 = np.dtype(ml_dtypes.bfloat16)
# Native paths, compare-and-swap loops on 32-bit words, and loops on 8-bit and 16-bit
# elements packed next to elements that other threads update at the same time.
ATOMIC_CASES = [
    ("add", np.int32), ("add", np.float32), ("add", np.int16), ("add", np.uint8),
    ("max", np.int32), ("max", np.uint32), ("max", np.float32), ("max", _BF16),
    ("min", np.float16), ("min", np.int8), ("xchg", np.float32), ("xchg", np.int16),
    ("and", np.uint32), ("or", np.int16), ("xor", np.int32),
    ("cas", np.int32), ("cas", np.float32), ("cas", np.uint16),
]  # fmt: skip


def _values(rng, n, dt):
    dt = np.dtype(dt)
    if dt.kind == "f" or dt == _BF16:
        return (rng.standard_normal(n) * 8).astype(np.float32).astype(dt)
    info = np.iinfo(dt)
    return rng.integers(info.min, info.max, n, endpoint=True).astype(dt)


@pytest.mark.parametrize("kind,dtype", ATOMIC_CASES)
@pytest.mark.parametrize("n", [1024, 1000])
def test_atomics_return_old_values_and_update_memory(kind, dtype, n):
    rng = np.random.default_rng(len(kind) * n)
    idx = rng.permutation(n).astype(np.int32)  # one atomic per address
    mem = _values(rng, n + 1, dtype)
    val = _values(rng, n, dtype)
    cmp = np.where(rng.random(n) < 0.5, mem[idx], _values(rng, n, dtype)).astype(dtype)

    def run(mem, idx, cmp, val):
        mem, old = mem.copy(), np.zeros(n, dtype)
        _atomic_kernel[(4,)](mem, idx, cmp, val, old, n, KIND=kind, BLOCK=256)
        return mem, old

    def reference(mem, idx, cmp, val):
        mem = mem.copy()
        old = mem[idx].copy()
        new = {
            "add": np.add, "max": np.maximum, "min": np.minimum, "xchg": lambda o, v: v,
            "and": np.bitwise_and, "or": np.bitwise_or, "xor": np.bitwise_xor,
            "cas": lambda o, v: np.where(o == cmp, v, o),
        }[kind](old, val)  # fmt: skip
        mem[idx] = np.asarray(new).astype(dtype)
        return mem, old

    check_kernel(run, (mem, idx, cmp, val), reference)


@enceladus.jit
def _count_kernel(cnt_ptr, old_ptr, cs_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    old = tl.atomic_add(cnt_ptr + offs, 1)
    row = tl.program_id(0) * BLOCK + offs
    tl.store(old_ptr + row, old)
    # A scan reads the copies in every thread that holds an element, not only the owner's.
    tl.store(cs_ptr + row, tl.cumsum(old, 0))


@pytest.mark.parametrize("block,num_warps", [(16, 1), (16, 4), (64, 4)])
def test_atomics_on_broadcast_layouts_run_once_per_element(mode, block, num_warps):
    # Fewer elements than threads: some lane or SIMD-group bits are broadcast. Running the
    # atomic in every thread that holds an element would count 2 to 8 per program, not 1.
    progs = 3
    cnt = np.zeros(block, np.int32)
    old = np.full((progs, block), -1, np.int32)
    cs = np.zeros_like(old)
    with execution_mode(mode):
        _count_kernel[(progs,)](cnt, old, cs, BLOCK=block, num_warps=num_warps)
    np.testing.assert_array_equal(cnt, progs)
    # The programs see each count once, in some order.
    np.testing.assert_array_equal(np.sort(old, axis=0), np.arange(progs)[:, None] + 0 * old)
    np.testing.assert_array_equal(cs, np.cumsum(old, axis=1))


@enceladus.jit
def _scalar_counter_kernel(cnt_ptr, fmax_ptr, out_ptr, vals_ptr):
    pid = tl.program_id(0)
    tl.store(out_ptr + pid, tl.atomic_add(cnt_ptr, 1))
    tl.atomic_max(fmax_ptr, tl.load(vals_ptr + pid))  # result unused


def test_scalar_atomics_hand_out_unique_tickets(mode):
    progs = 40
    vals = np.random.default_rng(3).standard_normal(progs).astype(np.float32)
    cnt, fmax = np.zeros(1, np.int32), np.full(1, -np.inf, np.float32)
    out = np.full(progs, -1, np.int32)
    with execution_mode(mode):
        _scalar_counter_kernel[(progs,)](cnt, fmax, out, vals)
    assert cnt[0] == progs and fmax[0] == vals.max()
    np.testing.assert_array_equal(np.sort(out), np.arange(progs))


@enceladus.jit
def _u64_minmax_kernel(mem_ptr, idx_ptr, val_ptr, n, KIND: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    p = mem_ptr + tl.load(idx_ptr + offs, mask=m)
    v = tl.load(val_ptr + offs, mask=m)
    if KIND == "max":
        tl.atomic_max(p, v, mask=m)
    else:
        tl.atomic_min(p, v, mask=m)


# uint64 max and min are Metal's only 64-bit atomics.
@pytest.mark.parametrize("kind", ["max", "min"])
def test_uint64_max_and_min_with_colliding_addresses(kind):
    rng = np.random.default_rng(4)
    n, slots = 3000, 37
    idx = rng.integers(0, slots, n).astype(np.int32)
    val = rng.integers(0, 2**64 - 1, n, dtype=np.uint64, endpoint=True)
    init = np.uint64(0) if kind == "max" else np.uint64(2**64 - 1)

    def run(idx, val):
        mem = np.full(slots, init, np.uint64)
        _u64_minmax_kernel[(enceladus.cdiv(n, 512),)](mem, idx, val, n, KIND=kind, BLOCK=512)
        return mem

    def reference(idx, val):
        mem = np.full(slots, init, np.uint64)
        (np.maximum if kind == "max" else np.minimum).at(mem, idx, val)
        return mem

    check_kernel(run, (idx, val), reference)


@enceladus.jit
def _half_add_kernel(p, v_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.atomic_add(p + offs, tl.load(v_ptr + offs))


@enceladus.jit
def _u64_old_kernel(p, out_ptr):
    tl.store(out_ptr, tl.atomic_max(p, 1))


@pytest.mark.parametrize(("dtype", "phrase"), [
    (np.float16, "Accumulate in a float32 buffer"),
    # 64-bit atomics other than uint64 max and min: the error names the alternative.
    (np.int64, "two uint32 buffers"),
    (np.uint64, "two uint32 buffers"),
])  # fmt: skip
def test_unsupported_atomics_raise_source_located_errors(mode, dtype, phrase):
    x = np.zeros(16, dtype)
    with execution_mode(mode), pytest.raises(enceladus.CompilationError) as e:
        _half_add_kernel[(1,)](x, x, BLOCK=16)
    assert phrase in str(e.value) and "tl.atomic_add(p + offs" in str(e.value)
    if dtype == np.float16 and mode == "compiled":
        u = np.zeros(1, np.uint64)
        with pytest.raises(enceladus.CompilationError, match="old value"):
            _u64_old_kernel.warmup(u, u)


def test_histogram_example(mode):
    ex = load_example("09_histogram")
    x = np.random.default_rng(5).standard_normal(100_003).astype(np.float32)
    with execution_mode(mode):
        got = ex.histogram(x, 48, -3.0, 3.0)
    np.testing.assert_array_equal(got, ex.reference(x, 48, -3.0, 3.0))


# ---------------------------------------------------------------------------
# Scans
# ---------------------------------------------------------------------------


# int64 has no native SIMD prefix sum or shuffle, so it takes the generic lane scan.
@pytest.mark.parametrize("dtype", [np.float32, np.float16, _BF16, np.int8, np.int64])
@pytest.mark.parametrize("shape", [(5, 1000), (64, 32), (3, 7)])
@pytest.mark.parametrize("reverse", [False, True])
def test_cumsum_example(dtype, shape, reverse):
    ex = load_example("10_cumsum")
    rng = np.random.default_rng(shape[1])
    if np.dtype(dtype).kind in "iu":
        x = rng.integers(-100, 100, shape).astype(dtype)
    else:
        x = (rng.standard_normal(shape) * 4).astype(np.float32).astype(dtype)
    # float32 sums associate differently in each mode; half types round once at the end.
    tol = 1e-4 if dtype == np.float32 else None
    check_kernel(ex.cumsum, (x,), ex.reference, kwargs={"reverse": reverse}, atol=tol,
                 rtol=tol)  # fmt: skip


@enceladus.jit
def _max_op(a, b):
    return tl.maximum(a, b)


@enceladus.jit
def _affine_op(a0, a1, b0, b1):
    # Composes h -> a0 * h + a1, then h -> b0 * h + b1. Associative, not commutative.
    return a0 * b0, b0 * a1 + b1


@enceladus.jit
def _scan2d_kernel(a_ptr, b_ptr, x_ptr, y_ptr, M: tl.constexpr, N: tl.constexpr,
                   AXIS: tl.constexpr, REVERSE: tl.constexpr):  # fmt: skip
    offs = tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :]
    a, b = tl.load(a_ptr + offs), tl.load(b_ptr + offs)
    h0, h1 = tl.associative_scan((a, b), AXIS, _affine_op, reverse=REVERSE)
    tl.store(a_ptr + offs, h0)
    tl.store(b_ptr + offs, h1)
    x = tl.load(x_ptr + offs)
    tl.store(y_ptr + offs, tl.associative_scan(x, AXIS, _max_op, reverse=REVERSE))


def _affine_ref(a, b, axis, reverse):
    a, b = np.moveaxis(a, axis, 0).copy(), np.moveaxis(b, axis, 0).copy()
    order = range(a.shape[0] - 2, -1, -1) if reverse else range(1, a.shape[0])
    prev = a.shape[0] - 1 if reverse else 0
    for i in order:
        a[i], b[i] = a[prev] * a[i], a[i] * b[prev] + b[i]
        prev = i
    return np.moveaxis(a, 0, axis), np.moveaxis(b, 0, axis)


@pytest.mark.parametrize("m,n", [(16, 64), (64, 16), (8, 2)])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("reverse", [False, True])
def test_associative_scan_along_either_axis_of_a_2d_tile(m, n, axis, reverse):
    rng = np.random.default_rng(m * n + axis)
    a = rng.integers(-3, 4, (m, n)).astype(np.int32)  # int32 wraps exactly in any order
    b = rng.integers(-100, 100, (m, n)).astype(np.int32)
    x = rng.standard_normal((m, n)).astype(np.float32)

    def run(a, b, x):
        a, b, y = a.copy(), b.copy(), np.empty_like(x)
        _scan2d_kernel[(1,)](a, b, x, y, M=m, N=n, AXIS=axis, REVERSE=reverse)
        return a, b, y

    def reference(a, b, x):
        f = np.flip if reverse else (lambda z, axis: z)
        return (*_affine_ref(a, b, axis, reverse),
                f(np.maximum.accumulate(f(x, axis=axis), axis=axis), axis=axis))  # fmt: skip

    with np.errstate(over="ignore"):
        check_kernel(run, (a, b, x), reference)


@enceladus.jit
def _dot_cumsum_kernel(a_ptr, b_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr,
                       K: tl.constexpr, AXIS: tl.constexpr):  # fmt: skip
    rm, rn, rk = tl.arange(0, M), tl.arange(0, N), tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    c = tl.cumsum(tl.dot(a, b), axis=AXIS)
    tl.store(out_ptr + rm[:, None] * N + rn[None, :], c)


@pytest.mark.parametrize("axis", [0, 1])
def test_cumsum_of_a_dot_accumulator(axis):
    # The simdgroup_matrix layout interleaves register and lane bits along both axes,
    # with lane bits out of order, and SIMD-group bits along the rows.
    rng = np.random.default_rng(axis)
    a = rng.integers(-2, 3, (64, 16)).astype(np.float32)
    b = rng.integers(-2, 3, (16, 32)).astype(np.float32)

    def run(a, b):
        out = np.empty((64, 32), np.float32)
        _dot_cumsum_kernel[(1,)](a, b, out, M=64, N=32, K=16, AXIS=axis)
        return out

    check_kernel(run, (a, b), lambda a, b: np.cumsum(a @ b, axis=axis))


@enceladus.jit
def _view_scan_kernel(a_ptr, b_ptr, out_a, out_b, REVERSE: tl.constexpr):
    offs = tl.arange(0, 2)[:, None] * 256 + tl.arange(0, 256)[None, :]
    # The flattened transpose puts a SIMD-group bit below the register and lane bits of
    # the scan axis, so every level kind needs threadgroup memory.
    a = tl.reshape(tl.trans(tl.load(a_ptr + offs)), (512,))
    b = tl.reshape(tl.trans(tl.load(b_ptr + offs)), (512,))
    h0, h1 = tl.associative_scan((a, b), 0, _affine_op, reverse=REVERSE)
    tl.store(out_a + tl.arange(0, 512), h0)
    tl.store(out_b + tl.arange(0, 512), h1)


@pytest.mark.parametrize("reverse", [False, True])
def test_scan_of_an_interleaved_view(reverse):
    rng = np.random.default_rng(7)
    a = rng.integers(-3, 4, (2, 256)).astype(np.int32)
    b = rng.integers(-100, 100, (2, 256)).astype(np.int32)

    def run(a, b):
        oa, ob = np.empty(512, np.int32), np.empty(512, np.int32)
        _view_scan_kernel[(1,)](a, b, oa, ob, REVERSE=reverse)
        return oa, ob

    def reference(a, b):
        return _affine_ref(a.T.reshape(-1), b.T.reshape(-1), 0, reverse)

    with np.errstate(over="ignore"):
        check_kernel(run, (a, b), reference)


def test_associative_scan_needs_a_jit_combine_fn():
    @enceladus.jit
    def k(x_ptr):
        x = tl.load(x_ptr + tl.arange(0, 16))
        tl.store(x_ptr + tl.arange(0, 16), tl.associative_scan(x, 0, np.maximum))

    x = np.zeros(16, np.float32)
    with pytest.raises(enceladus.CompilationError, match="@enceladus.jit function"):
        k.warmup(x)
