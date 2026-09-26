"""Differential tests: kernels against NumPy references, in every available mode.

With `TEGULA_VERIFY=1` (set in conftest), each interpreted launch also builds and verifies
the kernel's IR, so these tests cover the frontend for every kernel they run.
"""

from __future__ import annotations

import math

import ml_dtypes
import numpy as np
import pytest
from conftest import check_kernel, execution_mode, load_example

import tegula
import tegula.language as tl

F32, F16, BF16 = np.float32, np.float16, ml_dtypes.bfloat16


def randn(rng: np.random.Generator, shape, dtype) -> np.ndarray:
    return rng.standard_normal(shape).astype(np.float32).astype(dtype)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [F32, F16, BF16])
@pytest.mark.parametrize("n", [1, 1000, (1 << 16) + 3])
def test_vector_add(mode, rng, dtype, n):
    ex = load_example("01_vector_add")
    check_kernel(ex.add, (randn(rng, n, dtype), randn(rng, n, dtype)), ex.reference, modes=(mode,))


@pytest.mark.parametrize("dtype", [F32, F16])
@pytest.mark.parametrize("shape", [(4, 128), (37, 1000)])
def test_softmax(mode, rng, dtype, shape):
    ex = load_example("02_softmax")
    check_kernel(ex.softmax, (randn(rng, shape, dtype) * 4,), ex.reference, modes=(mode,))


@pytest.mark.parametrize("dtype", [F32, F16])
@pytest.mark.parametrize("shape, block", [((8, 256), 256), ((5, 1000), 256)])
def test_layernorm(mode, rng, dtype, shape, block):
    ex = load_example("03_layernorm")
    x = randn(rng, shape, dtype) * 2 + 1
    w, b = randn(rng, shape[1], dtype), randn(rng, shape[1], dtype)
    check_kernel(ex.layernorm, (x, w, b), ex.reference, kwargs={"block": block}, modes=(mode,),
                 atol=1e-4 if dtype is F32 else None)  # fmt: skip


@pytest.mark.parametrize("dtype", [F32, F16, BF16])
@pytest.mark.parametrize("mkn", [(64, 64, 64), (100, 70, 90)])
def test_matmul(mode, rng, dtype, mkn):
    ex = load_example("04_matmul")
    m, k, n = mkn
    a, b = randn(rng, (m, k), dtype), randn(rng, (k, n), dtype)
    check_kernel(ex.matmul, (a, b), ex.reference, modes=(mode,),
                 atol=1e-4 if dtype is F32 else None)  # fmt: skip


@pytest.mark.parametrize("dtype", [F32, F16, BF16])
@pytest.mark.parametrize("shape", [(4, 256), (7, 300)])
def test_fused_gelu(mode, rng, dtype, shape):
    ex = load_example("05_fused_gelu")
    x, bias = randn(rng, shape, dtype), randn(rng, shape[1], dtype)
    check_kernel(ex.fused_gelu, (x, bias, 0.75), ex.reference, modes=(mode,))


@pytest.mark.parametrize("dtype", [F32, F16])
@pytest.mark.parametrize("shape", [(4, 128), (9, 1000)])
def test_rmsnorm(mode, rng, dtype, shape):
    ex = load_example("06_rmsnorm")
    x, w = randn(rng, shape, dtype), randn(rng, shape[1], dtype)
    check_kernel(ex.rmsnorm, (x, w), ex.reference, modes=(mode,))


# ---------------------------------------------------------------------------
# Language semantics
# ---------------------------------------------------------------------------


_erf = np.vectorize(math.erf, otypes=[np.float64])

UNARY = {
    "exp": np.exp, "exp2": np.exp2, "log": np.log, "log2": np.log2, "sqrt": np.sqrt,
    "rsqrt": lambda x: 1 / np.sqrt(x), "sin": np.sin, "cos": np.cos, "tanh": np.tanh,
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)), "erf": _erf, "floor": np.floor,
    "ceil": np.ceil, "abs": np.abs, "neg": np.negative,
}  # fmt: skip


@tegula.jit
def _unary_kernel(x_ptr, out_ptr, n, OP: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    if OP == "neg":
        y = -x
    elif OP == "abs":
        y = tl.abs(x)
    else:
        y = getattr(tl, OP)(x)
    tl.store(out_ptr + offs, y, mask=mask)


@pytest.mark.parametrize("dtype", [F32, F16])
@pytest.mark.parametrize("op", list(UNARY))
def test_unary_ops(mode, rng, op, dtype):
    lo = 0.1 if op in ("log", "log2", "sqrt", "rsqrt") else -3.0
    x = rng.uniform(lo, 3.0, 300).astype(dtype)

    def run(x):
        out = np.empty_like(x)
        _unary_kernel[(1,)](x, out, x.size, OP=op, BLOCK=512)
        return out

    check_kernel(run, (x,), lambda x: UNARY[op](x.astype(np.float64)).astype(dtype), modes=(mode,))


BINARY = {
    "add": lambda x, y: x + y, "sub": lambda x, y: x - y, "mul": lambda x, y: x * y,
    "div": lambda x, y: x / y, "mod": np.fmod, "maximum": np.maximum, "minimum": np.minimum,
    "fma": lambda x, y: x * y + x, "clamp": lambda x, y: np.clip(x, -0.5, 0.5),
    "where": lambda x, y: np.where(x > y, x, 2 * y), "mixed": lambda x, y: x + y,
    "scalar": lambda x, y: 2 * x - 1,
}  # fmt: skip


@tegula.jit
def _binary_kernel(x_ptr, y_ptr, out_ptr, n, OP: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    y = tl.load(y_ptr + offs, mask=mask, other=1.0)
    if OP == "add":
        z = x + y
    elif OP == "sub":
        z = x - y
    elif OP == "mul":
        z = x * y
    elif OP == "div":
        z = x / y
    elif OP == "mod":
        z = x % y
    elif OP == "maximum":
        z = tl.maximum(x, y)
    elif OP == "minimum":
        z = tl.minimum(x, y)
    elif OP == "fma":
        z = tl.fma(x, y, x)
    elif OP == "clamp":
        z = tl.clamp(x, -0.5, 0.5)
    elif OP == "where":
        z = tl.where(x > y, x, y * 2)
    elif OP == "mixed":
        z = x + y.to(tl.float32)  # Promotes to float32; the store converts back.
    else:
        z = 2 * x - 1  # Literals adopt the tile's dtype.
    tl.store(out_ptr + offs, z, mask=mask)


@pytest.mark.parametrize("dtype", [F32, F16, BF16])
@pytest.mark.parametrize("op", list(BINARY))
def test_binary_ops(mode, rng, op, dtype):
    x = rng.uniform(-3, 3, 300).astype(dtype)
    y = (rng.uniform(0.5, 3, 300) * rng.choice([-1, 1], 300)).astype(dtype)

    def run(x, y):
        out = np.empty_like(x)
        _binary_kernel[(1,)](x, y, out, x.size, OP=op, BLOCK=512)
        return out

    def ref(x, y):
        return BINARY[op](x.astype(np.float64), y.astype(np.float64)).astype(dtype)

    check_kernel(run, (x, y), ref, modes=(mode,))


INT_OPS = {
    "and": lambda x, y: x & y, "or": lambda x, y: x | y, "xor": lambda x, y: x ^ y,
    "shl": lambda x, y: x << (y & 7), "shr": lambda x, y: x >> (y & 7),
    "invert": lambda x, y: ~x, "mul_wraps": lambda x, y: x * y * 65536,
    "trunc": lambda x, y: np.trunc(x.astype(np.float32) / np.float32(3)).astype(np.int32),
    "cmp": lambda x, y: (x < y).astype(np.int32) + (x == y),
    "bitcast": lambda x, y: x,
    "unsigned_shr": lambda x, y: (x.astype(np.uint32) >> 1).astype(np.int32),
}  # fmt: skip


@tegula.jit
def _int_kernel(x_ptr, y_ptr, out_ptr, n, OP: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    if OP == "and":
        z = x & y
    elif OP == "or":
        z = x | y
    elif OP == "xor":
        z = x ^ y
    elif OP == "shl":
        z = x << (y & 7)
    elif OP == "shr":
        z = x >> (y & 7)
    elif OP == "invert":
        z = ~x
    elif OP == "mul_wraps":
        z = x * y * 65536
    elif OP == "trunc":
        z = (x.to(tl.float32) / 3.0).to(tl.int32)
    elif OP == "cmp":
        z = (x < y).to(tl.int32) + (x == y)
    elif OP == "bitcast":
        z = x.to(tl.float32, bitcast=True).to(tl.int32, bitcast=True)
    else:
        z = (x.to(tl.uint32) >> 1).to(tl.int32)
    tl.store(out_ptr + offs, z, mask=mask)


@pytest.mark.parametrize("op", list(INT_OPS))
def test_int_ops(mode, rng, op):
    x = rng.integers(-1000, 1000, 100).astype(np.int32)
    y = rng.integers(-1000, 1000, 100).astype(np.int32)
    y[:10] = x[:10]

    def run(x, y):
        out = np.empty_like(x)
        _int_kernel[(1,)](x, y, out, x.size, OP=op, BLOCK=128)
        return out

    check_kernel(run, (x, y), INT_OPS[op], modes=(mode,))


SHAPE_OPS = {
    "trans": lambda x: x.T, "T": lambda x: x.T, "reshape": lambda x: x.reshape(16, 8),
    "row_bcast": lambda x: x - x.max(1, keepdims=True),
    "col_bcast": lambda x: x / x.sum(0, keepdims=True),
    "expand": lambda x: np.broadcast_to(x.min(1)[:, None], x.shape),
    "argmin": lambda x: x.argmin(0).astype(x.dtype),
    "sum_all": lambda x: np.full(2, x.astype(np.float64).sum()),
}  # fmt: skip


@tegula.jit
def _shape_kernel(x_ptr, out_ptr, OP: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    rm, rn = tl.arange(0, M), tl.arange(0, N)
    x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
    if OP == "trans":
        y = tl.trans(x)
    elif OP == "T":
        y = x.T
    elif OP == "reshape":
        y = tl.reshape(x, (N, M))
    elif OP == "row_bcast":
        y = x - tl.max(x, axis=1)[:, None]
    elif OP == "col_bcast":
        y = x / tl.sum(x, axis=0, keep_dims=True)
    elif OP == "expand":
        y = tl.broadcast_to(tl.expand_dims(tl.min(x, 1), 1), (M, N))
    elif OP == "argmin":
        y = tl.argmin(x, 0).to(x.dtype)
    else:
        y = tl.full((2,), tl.sum(x), x.dtype)
    if len(y.shape) == 1:
        tl.store(out_ptr + tl.arange(0, y.shape[0]), y)
    else:
        P, Q = y.shape
        tl.store(out_ptr + tl.arange(0, P)[:, None] * Q + tl.arange(0, Q)[None, :], y)


@pytest.mark.parametrize("dtype", [F32, F16])
@pytest.mark.parametrize("op", list(SHAPE_OPS))
def test_shape_and_reduce_ops(mode, rng, op, dtype):
    x = rng.uniform(0.5, 2, (8, 16)).astype(dtype)

    def run(x):
        expected = SHAPE_OPS[op](x)
        out = np.zeros(expected.size, dtype)
        _shape_kernel[(1,)](x, out, OP=op, M=8, N=16)
        return out.reshape(expected.shape)

    ref = lambda x: np.ascontiguousarray(SHAPE_OPS[op](x.astype(np.float64))).astype(dtype)  # noqa: E731
    check_kernel(run, (x,), ref, modes=(mode,), atol=0.05 if op == "sum_all" else None)


@tegula.jit
def _int_div_kernel(a_ptr, b_ptr, q_ptr, r_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask, other=1)
    b = tl.load(b_ptr + offs, mask=mask, other=1)
    tl.store(q_ptr + offs, a // b, mask=mask)
    tl.store(r_ptr + offs, a % b, mask=mask)


def test_integer_division_truncates(mode):
    a = np.array([7, -7, 7, -7, 0, 5, -1, 123456], np.int32)
    b = np.array([2, 2, -2, -2, 3, 5, 7, -1000], np.int32)

    def run(a, b):
        q, r = np.empty_like(a), np.empty_like(a)
        _int_div_kernel[(1,)](a, b, q, r, a.size, BLOCK=8)
        return q, r

    def ref(a, b):
        q = np.trunc(a / b).astype(np.int32)
        return q, (a - q * b).astype(np.int32)

    check_kernel(run, (a, b), ref, modes=(mode,))


@tegula.jit
def _add_combine(a, b):
    return a + b


@tegula.jit
def _row_stats_kernel(x_ptr, amax_ptr, sum_ptr, max_ptr, N: tl.constexpr):
    row = tl.program_id(0)
    x = tl.load(x_ptr + row * N + tl.arange(0, N)[None, :])  # shape (1, N)
    tl.store(amax_ptr + row + tl.arange(0, 1), tl.argmax(x, axis=1))
    s = tl.reduce(x, 1, _add_combine, keep_dims=True)  # shape (1, 1)
    tl.store(sum_ptr + row + tl.arange(0, 1)[:, None], s)
    tl.store(max_ptr + row, tl.max(x))


def test_reductions(mode):
    # Small integers make ties likely, which checks that argmax picks the lowest index.
    x = np.random.default_rng(1).integers(0, 4, (6, 16)).astype(np.float32)

    def run(x):
        amax, s, mx = np.empty(6, np.int32), np.empty(6, np.float32), np.empty(6, np.float32)
        _row_stats_kernel[(6,)](x, amax, s, mx, N=16)
        return amax, s, mx

    ref = lambda x: (x.argmax(1).astype(np.int32), x.sum(1), x.max(1))  # noqa: E731
    check_kernel(run, (x,), ref, modes=(mode,))


@tegula.jit
def _scale(x, FACTOR: tl.constexpr):
    if FACTOR == 1:
        return x, 0
    return x * FACTOR, 1


@tegula.jit
def _control_flow_kernel(x_ptr, out_ptr, flags_ptr, n_blocks, thresh, BLOCK: tl.constexpr,
                         FACTOR: tl.constexpr):  # fmt: skip
    total = 0.0  # A literal that becomes a loop-carried f32.
    count = 0
    for i in range(n_blocks):
        x = tl.load(x_ptr + i * BLOCK + tl.arange(0, BLOCK))
        s = tl.sum(x, axis=0)
        if s > thresh:
            total += s
            count += 1
        else:
            total -= 1.0
    y, used = _scale(total, FACTOR)
    acc = 0
    for j in tl.static_range(3):
        acc += j * used
    tl.store(out_ptr, y)
    tl.store(flags_ptr + tl.arange(0, 2), tl.where(tl.arange(0, 2) == 0, count, acc))


def test_control_flow(mode):
    x = np.random.default_rng(2).standard_normal((5, 32)).astype(np.float32)

    def run(x, factor):
        out, flags = np.empty(1, np.float32), np.empty(2, np.int32)
        _control_flow_kernel[(1,)](x, out, flags, 5, 0.5, BLOCK=32, FACTOR=factor)
        return out, flags

    def ref(x, factor):
        s = x.astype(np.float64).sum(1)
        total = np.where(s > 0.5, s, -1.0).sum()
        used = int(factor != 1)
        return np.array([total * factor], np.float32), np.array([(s > 0.5).sum(), 3 * used],
                                                                  np.int32)  # fmt: skip

    for factor in (1, 3):
        check_kernel(run, (x, factor), ref, modes=(mode,), atol=1e-5)

    module = _control_flow_kernel.ir(x, np.empty(1, np.float32), np.empty(2, np.int32), 5, 0.5,
                                     BLOCK=32, FACTOR=3)  # fmt: skip
    top = [op.name for op in module.body.ops]
    assert top.count("for") == 1, "static_range must unroll; only range() emits a for op"
    loop = next(op for op in module.body.ops if op.name == "for")
    assert [r.type for r in loop.results] == [tegula.compiler.ir.f32, tegula.compiler.ir.i32]
    assert any(op.name == "if" and len(op.results) == 2 for op in loop.walk())


@tegula.jit
def _desc_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_bk, stride_cm,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):  # fmt: skip
    pid_n, pid_m = tl.program_id(0), tl.program_id(1)
    a = tl.make_tensor_descriptor(a_ptr, [M, K], [stride_am, 1], [BM, BK])
    b = tl.make_tensor_descriptor(b_ptr, [K, N], [stride_bk, 1], [BK, BN])
    c = tl.make_tensor_descriptor(c_ptr, [M, N], [stride_cm, 1], [BM, BN])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc = tl.dot(a.load([pid_m * BM, k]), b.load([k, pid_n * BN]), acc)
    c.store([pid_m * BM, pid_n * BN], acc.to(c.dtype))


@pytest.mark.parametrize("dtype", [F32, F16])
def test_descriptor_matmul(rng, dtype):
    m, k, n = 50, 40, 33  # Ragged in every dimension: loads zero-fill, stores skip.

    def run(a, b):
        c = np.full((m, n + 3), 7, dtype)  # The padding columns must stay untouched.
        grid = (tegula.cdiv(n, 16), tegula.cdiv(m, 32))
        _desc_matmul_kernel[grid](a, b, c, m, n, k, k, n, n + 3, BM=32, BN=16, BK=16)
        return c

    def ref(a, b):
        c = np.full((m, n + 3), 7, dtype)
        c[:, :n] = (a.astype(np.float32) @ b.astype(np.float32)).astype(dtype)
        return c

    # Compiled descriptors arrive in M4.
    check_kernel(run, (randn(rng, (m, k), dtype), randn(rng, (k, n), dtype)), ref,
                 modes=("interpret",), atol=1e-4 if dtype is F32 else None)  # fmt: skip


@tegula.jit
def _oob_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))  # oob-line
    tl.store(out_ptr + tl.arange(0, BLOCK), x)


def test_out_of_bounds_load_names_kernel_line():
    x = np.zeros(10, np.float32)
    with execution_mode("interpret"), pytest.raises(IndexError) as e:
        _oob_kernel[(1,)](x, np.zeros(16, np.float32), BLOCK=16)
    msg = str(e.value)
    assert "out-of-bounds load" in msg
    assert "# oob-line" in msg and __file__ in msg


@tegula.jit
def _welford_combine(mean_a, m2_a, n_a, mean_b, m2_b, n_b):
    n = n_a + n_b
    delta = mean_b - mean_a
    frac = tl.where(n == 0, 0.0, n_b / n)
    return mean_a + delta * frac, m2_a + m2_b + delta * delta * n_a * frac, n


@tegula.jit
def _welford_kernel(x_ptr, mean_ptr, var_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0)
    count = mask.to(tl.float32)
    mean, m2, n = tl.reduce((x, tl.zeros_like(x), count), 0, _welford_combine)
    tl.store(mean_ptr + row, mean)
    tl.store(var_ptr + row, m2 / n)


def test_welford_tuple_reduce(mode, rng):
    x = (rng.standard_normal((3, 100)) * 3 + 5).astype(np.float32)

    def run(x):
        mean, var = np.empty(3, np.float32), np.empty(3, np.float32)
        _welford_kernel[(3,)](x, mean, var, 100, BLOCK=128)
        return mean, var

    ref = lambda x: (x.mean(1, dtype=np.float64).astype(np.float32),  # noqa: E731
                     x.var(1, dtype=np.float64).astype(np.float32))
    check_kernel(run, (x,), ref, modes=(mode,), atol=1e-4)
