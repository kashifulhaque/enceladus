"""Differential tests: kernels against NumPy references, in every available mode.

With `ENCELADUS_VERIFY=1` (set in conftest), each interpreted launch also compiles the kernel
to MSL and raises the compiler's errors, so these tests cover the compiler for every kernel
they run.
"""

from __future__ import annotations

import math

import ml_dtypes
import numpy as np
import pytest
from conftest import check_kernel, execution_mode, load_example

import enceladus
import enceladus.language as tl

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
# (3, 20) runs a 32-wide block on 128 threads, so whole lanes see only `other=-inf`.
@pytest.mark.parametrize("shape", [(4, 128), (37, 1000), (3, 20)])
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


# test_matmul_shapes skips the ragged pointer-tile shapes in the interpreter, so this one
# covers them there.
@pytest.mark.parametrize("dtype", [F32, F16, BF16])
def test_matmul(mode, rng, dtype):
    ex = load_example("04_matmul")
    m, k, n = 100, 70, 90
    a, b = randn(rng, (m, k), dtype), randn(rng, (k, n), dtype)
    check_kernel(ex.matmul, (a, b), ex.reference, modes=(mode,),
                 atol=1e-4 if dtype is F32 else None)  # fmt: skip


# The plan's matmul shapes; the largest runs compiled only (the interpreter takes seconds).
MATMUL_SHAPES = [(64, 64, 64), (513, 513, 513), (1000, 300, 777), (2048, 2048, 2048)]


@pytest.mark.parametrize("dtype", [F32, F16, BF16])
@pytest.mark.parametrize("mkn", MATMUL_SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("variant", ["desc", "pointer"])
def test_matmul_shapes(mode, rng, dtype, mkn, variant):
    m, k, n = mkn
    if mode == "interpret" and (m * n * k > 1 << 27 or variant == "pointer" and m > 64):
        pytest.skip("too slow for the interpreter; compiled mode covers it")
    ex = load_example("04_matmul")
    a, b = randn(rng, (m, k), dtype), randn(rng, (k, n), dtype)
    run = ex.matmul_desc if variant == "desc" else ex.matmul
    tol = 1e-4 * math.sqrt(k) if dtype is F32 else None
    check_kernel(run, (a, b), ex.reference, modes=(mode,), atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", [F32, F16])
def test_matmul_fused_epilogue(mode, rng, dtype):
    ex = load_example("07_matmul_fused")
    a, b = randn(rng, (200, 96), dtype), randn(rng, (96, 130), dtype)
    bias = randn(rng, 130, dtype)
    check_kernel(ex.matmul_bias_gelu, (a, b, bias), ex.reference, modes=(mode,),
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


@enceladus.jit
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


# Magnitudes where fast Metal math breaks: its `tanh` gives 0 at 44 and NaN from 45, and
# its `sin` and `cos` give 0 around 1e7. In float16, 1e7 becomes infinity.
LARGE = np.array([10.0, 44.0, 45.0, 100.0, 3e4, 1e7, np.inf])


@pytest.mark.parametrize("dtype", [F32, F16])
@pytest.mark.parametrize("op", list(UNARY))
def test_unary_ops(mode, rng, op, dtype):
    positive = op in ("log", "log2", "sqrt", "rsqrt")
    big = LARGE if positive else np.concatenate([LARGE, -LARGE])
    with np.errstate(over="ignore"):
        x = np.concatenate([rng.uniform(0.1 if positive else -3.0, 3.0, 300), big]).astype(dtype)

    def run(x):
        out = np.empty_like(x)
        _unary_kernel[(1,)](x, out, x.size, OP=op, BLOCK=512)
        return out

    def ref(x):
        with np.errstate(over="ignore", invalid="ignore"):
            return UNARY[op](x.astype(np.float64)).astype(dtype)

    check_kernel(run, (x,), ref, modes=(mode,))


BINARY = {
    "add": lambda x, y: x + y, "sub": lambda x, y: x - y, "mul": lambda x, y: x * y,
    "div": lambda x, y: x / y, "mod": np.fmod, "maximum": np.maximum, "minimum": np.minimum,
    "fma": lambda x, y: x * y + x, "clamp": lambda x, y: np.clip(x, -0.5, 0.5),
    "where": lambda x, y: np.where(x > y, x, 2 * y), "mixed": lambda x, y: x + y,
    "scalar": lambda x, y: 2 * x - 1,
}  # fmt: skip


@enceladus.jit
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


@enceladus.jit
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


@enceladus.jit
def _cast_kernel(x_ptr, out_ptr, DT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(x_ptr + offs).to(DT))


# float16 overflows to inf from 65520, and 2**30 + 2**22 + 1 rounds to bfloat16 through
# float32 in NumPy (ml_dtypes), which gives 2**30 rather than 2**30 + 2**23.
CAST_VALUES = [65504, 65505, 65519, 65520, -65519, 2049, 2051, 2**30 + 2**22 + 1]


@pytest.mark.parametrize("dst", [F16, BF16], ids=["f16", "bf16"])
@pytest.mark.parametrize("src", [np.int64, np.uint64, np.int32])
def test_int_to_half_casts(mode, src, dst):
    x = np.array([abs(v) if src is np.uint64 else v for v in CAST_VALUES], src)

    def run(x):
        out = np.zeros(x.size, dst)
        _cast_kernel[(1,)](x, out, DT=tl.float16 if dst is F16 else tl.bfloat16, BLOCK=8)
        return out

    with np.errstate(over="ignore"):
        check_kernel(run, (x,), lambda x: x.astype(dst), modes=(mode,), atol=0, rtol=0)


SHAPE_OPS = {
    "trans": lambda x: x.T, "T": lambda x: x.T, "reshape": lambda x: x.reshape(16, 8),
    "row_bcast": lambda x: x - x.max(1, keepdims=True),
    "col_bcast": lambda x: x / x.sum(0, keepdims=True),
    "expand": lambda x: np.broadcast_to(x.min(1)[:, None], x.shape),
    "argmin": lambda x: x.argmin(0).astype(x.dtype),
    "sum_all": lambda x: np.full(2, x.astype(np.float64).sum()),
}  # fmt: skip


@enceladus.jit
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


@enceladus.jit
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


@enceladus.jit
def _offsets_kernel(x_ptr, out_ptr, big, OP: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    if OP == "int64":  # 2**32 truncates to 0 in 32 bits.
        p = (x_ptr - big) + offs.to(tl.int64)
        p = p + (offs.to(tl.int64) * 0 + big)
    elif OP == "uint32_sub":  # Negating in uint32 wraps around.
        k = tl.program_id(0) + BLOCK
        p = (x_ptr + k) - k.to(tl.uint32) + offs
    elif OP == "int8_sub":  # -(-128) is -128 in int8.
        k = (tl.program_id(0) - 128).to(tl.int8)
        p = (x_ptr - 128) - k + offs
    elif OP == "int16_sub":  # The same for a tile of int16 at -32768.
        k = (offs * 0 + tl.program_id(0) - 32768).to(tl.int16)
        p = (x_ptr - 32768 + offs) - k
    else:  # Negating an int1 in int1 gives 1, not -1.
        p = (x_ptr + 1) + offs - (offs >= 0)
    tl.store(out_ptr + offs, tl.load(p))


@pytest.mark.parametrize("op", ["int64", "uint32_sub", "int8_sub", "int16_sub", "bool_sub"])
def test_pointer_offsets_keep_their_value(mode, op):
    x = np.arange(32, dtype=np.float32)

    def run(x):
        out = np.zeros_like(x)
        _offsets_kernel[(1,)](x, out, 1 << 32, OP=op, BLOCK=32)
        return out

    check_kernel(run, (x,), lambda x: x, modes=(mode,))


@enceladus.jit
def _add_combine(a, b):
    return a + b


@enceladus.jit
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


@enceladus.jit
def _scale(x, FACTOR: tl.constexpr):
    if FACTOR == 1:
        return x, 0
    return x * FACTOR, 1


@enceladus.jit
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
    assert [r.type for r in loop.results] == [enceladus.compiler.ir.f32, enceladus.compiler.ir.i32]
    assert any(op.name == "if" and len(op.results) == 2 for op in loop.walk())


@enceladus.jit
def _carried_kernel(out_ptr, n, B: tl.constexpr):
    s = 0
    t = tl.zeros((B,), tl.int32)
    p, q = 1, 2
    for _ in range(n):
        t_new = tl.full((B,), s, tl.int32)  # a uniform tile that reads `s` before its update
        s = s + 1
        t = t_new
        p, q = q, p + q
    tl.store(out_ptr + tl.arange(0, B), t + p * 1000 + q * 100000)


@pytest.mark.parametrize("n", [0, 1, 3])
def test_loop_carried_values_read_the_previous_iteration(mode, n):
    def ref(n):
        t, s, p, q = 0, 0, 1, 2
        for _ in range(n):
            t, s, p, q = s, s + 1, q, p + q
        return np.full(16, t + p * 1000 + q * 100000, np.int32)

    def run(n):
        out = np.empty(16, np.int32)
        _carried_kernel[(1,)](out, n, B=16)
        return out

    check_kernel(run, (n,), ref, modes=(mode,))


@enceladus.jit
def _range64_kernel(out_ptr, base, lo, hi, step, STEP: tl.constexpr):
    acc = 0
    count = 0
    for i in range(base + lo, base + hi, step if STEP == 0 else STEP):
        acc += (i - base).to(tl.int32) * 1000
        count += 1
    tl.store(out_ptr, acc + count)


# (lo, hi, step) as offsets from a base above 2**32. A zero `const_step` passes the step at
# run time.
@pytest.mark.parametrize("lo, hi, step", [(0, 30, 1), (3, 30, 7), (30, 0, -1), (29, 0, -4),
                                          (0, 0, 1), (5, 0, 2), (0, 5, -1)])  # fmt: skip
@pytest.mark.parametrize("const_step", [False, True])
def test_64_bit_loops_follow_python_range(mode, lo, hi, step, const_step):
    # Metal's compiler service crashed on some loops that count in a `long`.
    base = 5_000_000_000

    def run(lo, hi, step):
        out = np.zeros(1, np.int32)
        _range64_kernel[(1,)](out, base, lo, hi, step, STEP=step if const_step else 0)
        return out

    def ref(lo, hi, step):
        r = range(lo, hi, step)
        return np.array([sum(r) * 1000 + len(r)], np.int32)

    check_kernel(run, (lo, hi, step), ref, modes=(mode,))


# A separate kernel, so the recompile count of `_range64_kernel` stays below the warning.
_range_u64_kernel = enceladus.jit(_range64_kernel.fn)


@pytest.mark.parametrize("lo, hi, step", [(0, 30, 1), (3, 30, 7), (5, 0, 2)])
@pytest.mark.parametrize("const_step", [False, True])
def test_uint64_loops_follow_python_range(mode, lo, hi, step, const_step):
    # Unsigned bounds above 2**63 count in a uint64, and the arguments type as u64.
    base = (1 << 63) + 5_000_000_000

    def run(lo, hi, step):
        out = np.zeros(1, np.int32)
        _range_u64_kernel[(1,)](out, base, np.uint64(lo), np.uint64(hi), np.uint64(step),
                              STEP=step if const_step else 0)  # fmt: skip
        return out

    def ref(lo, hi, step):
        r = range(lo, hi, step)
        return np.array([sum(r) * 1000 + len(r)], np.int32)

    check_kernel(run, (lo, hi, step), ref, modes=(mode,))


# `and` and `or` return an operand, as in Python. `x` is 42, read only when it's needed.
BOOL_OPS = {
    "and": lambda n, m, f, x: n and m, "or": lambda n, m, f, x: n or m,
    "and_const": lambda n, m, f, x: n and 5, "or_const": lambda n, m, f, x: n or 5,
    "const_left": lambda n, m, f, x: 0 or n, "chain": lambda n, m, f, x: n and m and 9,
    "mixed": lambda n, m, f, x: n and (m + 1) * 2 or 11,
    "lazy_load": lambda n, m, f, x: n or x,
    "if_test": lambda n, m, f, x: 5 if (n and f) or not m else 6,
}  # fmt: skip


@enceladus.jit(do_not_specialize=["n", "m"])
def _bool_op_kernel(out_ptr, x_ptr, n, m, f, OP: tl.constexpr):
    if OP == "and":
        r = n and m
    elif OP == "or":
        r = n or m
    elif OP == "and_const":
        r = n and 5
    elif OP == "or_const":
        r = n or 5
    elif OP == "const_left":
        r = 0 or n
    elif OP == "chain":
        r = n and m and 9
    elif OP == "mixed":
        r = n and (m + 1) * 2 or 11
    elif OP == "lazy_load":
        r = n or tl.load(x_ptr)
    else:  # Only truth values matter here, so the operand types can differ.
        r = 5 if (n and f) or not m else 6
    tl.store(out_ptr, r)


@pytest.mark.parametrize("op", list(BOOL_OPS))
def test_bool_ops_return_operands(mode, op):
    nm = np.array([[3, 7], [0, 7], [2, 0], [0, 0]], np.int32)

    def run(nm):
        x, outs = np.array([42], np.int32), [np.zeros(1, np.int32) for _ in nm]
        for out, (n, m) in zip(outs, nm, strict=True):
            _bool_op_kernel[(1,)](out, x, int(n), int(m), float(n) / 2, OP=op)
        return np.concatenate(outs)

    def ref(nm):
        return np.array([BOOL_OPS[op](int(n), int(m), n / 2, 42) for n, m in nm], np.int32)

    check_kernel(run, (nm,), ref, modes=(mode,))



@enceladus.jit(do_not_specialize=["n"])
def _guarded_bias_kernel(x_ptr, bias_ptr: tl.constexpr, out_ptr, n, HAS_BIAS: tl.constexpr,
                         FORM: tl.constexpr, BLOCK: tl.constexpr):  # fmt: skip
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    if FORM == "and":
        if n > 0 and HAS_BIAS:
            x += tl.load(bias_ptr + offs)
    elif FORM == "or":
        if n <= 0 or not HAS_BIAS:
            x += 1.0
        else:
            x += tl.load(bias_ptr + offs)
    else:
        x += tl.load(bias_ptr + offs) if n > 0 and HAS_BIAS else 1.0
    tl.store(out_ptr + offs, x)


@pytest.mark.parametrize("form", ["and", "or", "ifexp"])
def test_constexpr_right_operand_skips_the_guarded_code(mode, form):
    # With HAS_BIAS=False the test is decided at compile time, so the load from the
    # `None` bias pointer must not be compiled.
    def run(x):
        out = np.empty_like(x)
        _guarded_bias_kernel[(1,)](x, None, out, 3, HAS_BIAS=False, FORM=form, BLOCK=16)
        return out

    check_kernel(run, (np.arange(16, dtype=np.float32),),
                 lambda x: x if form == "and" else x + 1, modes=(mode,))  # fmt: skip


@enceladus.jit
def _desc_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_b, stride_cm,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                        TRANS_B: tl.constexpr):  # fmt: skip
    pid_n, pid_m = tl.program_id(0), tl.program_id(1)
    a = tl.make_tensor_descriptor(a_ptr, [M, K], [stride_am, 1], [BM, BK])
    if TRANS_B:  # b is stored N x K; read it transposed
        b = tl.make_tensor_descriptor(b_ptr, [N, K], [stride_b, 1], [BN, BK])
    else:
        b = tl.make_tensor_descriptor(b_ptr, [K, N], [stride_b, 1], [BK, BN])
    c = tl.make_tensor_descriptor(c_ptr, [M, N], [stride_cm, 1], [BM, BN])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        if TRANS_B:
            bt = tl.trans(b.load([pid_n * BN, k]))
        else:
            bt = b.load([k, pid_n * BN])
        acc = tl.dot(a.load([pid_m * BM, k]), bt, acc)
    c.store([pid_m * BM, pid_n * BN], acc.to(c.dtype))


@pytest.mark.parametrize("trans_b", [False, True])
@pytest.mark.parametrize("dtype", [F32, F16, BF16])
@pytest.mark.parametrize("mkn", [(64, 32, 32), (50, 40, 33)])
def test_descriptor_matmul(mode, rng, dtype, mkn, trans_b):
    m, k, n = mkn  # The ragged shape makes loads zero-fill and stores skip.

    def run(a, b):
        c = np.full((m, n + 3), 7, dtype)  # The padding columns must stay untouched.
        grid = (enceladus.cdiv(n, 16), enceladus.cdiv(m, 32))
        bb = np.ascontiguousarray(b.T) if trans_b else b
        _desc_matmul_kernel[grid](a, bb, c, m, n, k, k, bb.shape[1], n + 3, BM=32, BN=16,
                                  BK=16, TRANS_B=trans_b)  # fmt: skip
        return c

    def ref(a, b):
        c = np.full((m, n + 3), 7, dtype)
        c[:, :n] = (a.astype(np.float32) @ b.astype(np.float32)).astype(dtype)
        return c

    check_kernel(run, (randn(rng, (m, k), dtype), randn(rng, (k, n), dtype)), ref,
                 modes=(mode,), atol=1e-4 if dtype is F32 else None)  # fmt: skip


@enceladus.jit
def _int_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr,
                       BK: tl.constexpr, VARIANT: tl.constexpr):  # fmt: skip
    pid_n, pid_m = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    if VARIANT == "desc":
        a = tl.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [BM, BK])
        b = tl.make_tensor_descriptor(b_ptr, [K, N], [N, 1], [BK, BN])
        for k in range(0, K, BK):
            acc = tl.dot(a.load([pid_m * BM, k]), b.load([k, pid_n * BN]), acc)
    else:
        # "hoist" loads A (K <= BK) once, so the dot reads it from registers.
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :],
                    mask=(rm[:, None] < M) & (rk[None, :] < K), other=0)  # fmt: skip
        for k in range(0, K, BK):
            if VARIANT == "pointer":
                a = tl.load(a_ptr + rm[:, None] * K + (k + rk)[None, :],
                            mask=(rm[:, None] < M) & ((k + rk)[None, :] < K), other=0)  # fmt: skip
            b = tl.load(b_ptr + (k + rk)[:, None] * N + rn[None, :],
                        mask=((k + rk)[:, None] < K) & (rn[None, :] < N), other=0)  # fmt: skip
            acc = tl.dot(a, b, acc)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def _int_matmul(a, b, variant="desc", bm=32, bn=32, bk=32, num_warps=4):
    (m, k), n = a.shape, b.shape[1]
    c = np.zeros((m, n), np.int32)
    grid = (enceladus.cdiv(n, bn), enceladus.cdiv(m, bm))
    _int_matmul_kernel[grid](a, b, c, m, n, k, BM=bm, BN=bn, BK=bk, VARIANT=variant,
                             num_warps=num_warps)  # fmt: skip
    return c


def _int_matmul_ref(a, b, **_):
    return (a.astype(np.int64) @ b.astype(np.int64)).astype(np.int32)  # wraps like int32


@pytest.mark.parametrize("variant", ["desc", "pointer", "hoist"])
@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.int32])
@pytest.mark.parametrize("mkn", [(64, 64, 64), (50, 100, 33)])
def test_integer_dot(mode, rng, dtype, mkn, variant):
    # Full-range values: 16-bit and 32-bit products and sums wrap modulo 2^32.
    m, k, n = mkn
    info = np.iinfo(dtype)
    a = rng.integers(info.min, info.max, (m, k), dtype, endpoint=True)
    b = rng.integers(info.min, info.max, (k, n), dtype, endpoint=True)
    blocks = {"bm": 16, "bk": enceladus.next_power_of_2(k)} if variant == "hoist" else {}
    check_kernel(_int_matmul, (a, b), _int_matmul_ref, kwargs={"variant": variant, **blocks},
                 modes=(mode,))  # fmt: skip


@pytest.mark.parametrize("dtype, k, bk, variant", [
    (np.int8, 2048, 32, "desc"),
    # One block of K = 512 needs two float32 partial sums of 256 steps each.
    (np.uint8, 512, 512, "desc"),
    (np.uint8, 512, 512, "hoist"),
])  # fmt: skip
def test_integer_dot_is_exact_past_float32_precision(mode, dtype, k, bk, variant):
    # Every sum exceeds 2^24 and is odd, so float32 accumulation would round it.
    extreme = np.iinfo(dtype).min if dtype == np.int8 else np.iinfo(dtype).max
    a = np.full((8, k), extreme, dtype)
    b = np.full((k, 8), extreme, dtype)
    a[:, 0] = 1
    check_kernel(_int_matmul, (a, b), _int_matmul_ref, modes=(mode,),
                 kwargs={"variant": variant, "bm": 8, "bn": 8, "bk": bk, "num_warps": 1})


# Argument names that generated code also uses: the register loop index `r`, the MMA
# loop's `i`, `j`, `kk`, `fa`, and `fb`, the exchange buffer `buf`, the argmax tie flag
# `tk`, and the kernel parameters `lane`, `warp`, and `pid`. The rest mean something to
# MSL: `metal_stdlib` macros (`INT_MAX`, `M_SQRT2_F`, and `METAL_FUNC`) and an address
# space (`ray_data`). MSL also forbids a kernel named `main`.
@enceladus.jit
def main(x_ptr, out_ptr, r, i, j, kk, fa, fb, buf, tk, lane, warp, pid, INT_MAX, M_SQRT2_F,
         METAL_FUNC, ray_data, BLOCK: tl.constexpr):  # fmt: skip
    offs = tl.arange(0, BLOCK)
    y = tl.load(x_ptr + offs) + r + i + j + kk + fa + fb + buf + tk + lane + warp + pid
    tl.store(out_ptr + offs, y + INT_MAX + M_SQRT2_F + METAL_FUNC + ray_data)


@enceladus.jit
def _reserved_names_dot_kernel(fa, fb, c_ptr, i, j, kk, BM: tl.constexpr, BN: tl.constexpr,
                               BK: tl.constexpr):  # fmt: skip
    pid_n, pid_m = tl.program_id(0), tl.program_id(1)
    a = tl.make_tensor_descriptor(fa, [i, kk], [kk, 1], [BM, BK])
    b = tl.make_tensor_descriptor(fb, [kk, j], [j, 1], [BK, BN])
    c = tl.make_tensor_descriptor(c_ptr, [i, j], [j, 1], [BM, BN])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, kk, BK):
        acc = tl.dot(a.load([pid_m * BM, k]), b.load([k, pid_n * BN]), acc)
    c.store([pid_m * BM, pid_n * BN], acc)


@pytest.mark.parametrize("kind", ["elementwise", "dot"])
def test_argument_names_dont_clash_with_generated_code(mode, rng, kind):
    if kind == "elementwise":
        scalars = [float(1 << k) for k in range(15)]  # Any clash changes the sum.

        def run(x):
            out = np.empty_like(x)
            main[(1,)](x, out, *scalars, BLOCK=1024)  # 8 registers each
            return out

        check_kernel(run, (randn(rng, 1024, F32),), lambda x: x + sum(scalars), modes=(mode,))
        return
    m, n, k = 50, 40, 36

    def run(a, b):
        c = np.zeros((m, n), np.float32)
        grid = (enceladus.cdiv(n, 32), enceladus.cdiv(m, 32))
        _reserved_names_dot_kernel[grid](a, b, c, m, n, k, BM=32, BN=32, BK=16)
        return c

    check_kernel(run, (randn(rng, (m, k), F32), randn(rng, (k, n), F32)), lambda a, b: a @ b,
                 modes=(mode,), atol=1e-4)  # fmt: skip


@enceladus.jit
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


@enceladus.jit
def _welford_combine(mean_a, m2_a, n_a, mean_b, m2_b, n_b):
    n = n_a + n_b
    delta = mean_b - mean_a
    frac = tl.where(n == 0, 0.0, n_b / n)
    return mean_a + delta * frac, m2_a + m2_b + delta * delta * n_a * frac, n


@enceladus.jit
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


# ---------------------------------------------------------------------------
# Interpreter fidelity: literals, loop variables, NaN reductions, and refusals
# ---------------------------------------------------------------------------


@enceladus.jit
def _unsigned_literal_kernel(x_ptr, out_ptr, n, OP: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    if OP == "add":
        y = tl.load(x_ptr + offs) + -1
    elif OP == "other":
        y = tl.load(x_ptr + offs, mask=offs < n, other=-1)
    else:
        y = tl.full((BLOCK,), -1, x_ptr.dtype)
    tl.store(out_ptr + offs, y)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint32])
@pytest.mark.parametrize("op", ["add", "other", "full"])
def test_negative_literals_wrap_in_unsigned_types(mode, op, dtype):
    def run(x):
        out = np.empty_like(x)
        _unsigned_literal_kernel[(1,)](x, out, 5, OP=op, BLOCK=16)
        return out

    def ref(x):
        top = np.iinfo(dtype).max
        return {"add": x - dtype(1), "other": np.where(np.arange(16) < 5, x, top).astype(dtype),
                "full": np.full(16, top, dtype)}[op]  # fmt: skip

    check_kernel(run, (np.arange(16, dtype=dtype),), ref, modes=(mode,))


@enceladus.jit
def _literal_error_kernel(out_ptr, OP: tl.constexpr):
    if OP == "float64":
        tl.store(out_ptr, tl.zeros((1,), tl.float64))  # literal-error-line
    elif OP == "full_big":
        tl.store(out_ptr, tl.full((1,), 1e10, tl.int32))  # literal-error-line
    elif OP == "full_nan":
        tl.store(out_ptr, tl.full((1,), float("nan"), tl.int32))  # literal-error-line
    elif OP == "store_float":
        tl.store(out_ptr, 130.5)  # literal-error-line
    elif OP == "other_float":
        x = tl.load(out_ptr + 1, mask=False, other=-130.5)  # literal-error-line
        tl.store(out_ptr, x)
    else:
        tl.store(out_ptr, 300)  # literal-error-line


@pytest.mark.parametrize("op", ["float64", "full_big", "full_nan", "store_wide", "store_float",
                                "other_float"])  # fmt: skip
def test_literals_that_dont_fit_are_refused(mode, op, monkeypatch):
    # With verification off, the interpreter checks the kernel on its own.
    monkeypatch.setenv("ENCELADUS_VERIFY", "0")
    with execution_mode(mode), pytest.raises(enceladus.CompilationError) as e:
        _literal_error_kernel[(1,)](np.zeros(1, np.int8), OP=op)
    assert "# literal-error-line" in str(e.value) and __file__ in str(e.value)


@enceladus.jit
def _loop_sum(n):
    acc = 0
    for j in tl.range(0, n):
        acc += j * 1500000000  # wraps in int32, as on the GPU
    return acc


@enceladus.jit(do_not_specialize=["n"])
def _loop_variable_kernel(out_ptr, n):
    acc = 0  # a literal that the loop makes an int32 scalar, even when it runs 0 times
    for i in range(n):
        acc += i
        tl.store(out_ptr + 4 + i, i.to(tl.float32) * 0.5)
    tl.store(out_ptr, acc.to(tl.float32))
    tl.store(out_ptr + 1, _loop_sum(n).to(tl.float32))
    big = 0
    for i in range(0, 1 << 34, 1 << 32):  # an int64 loop variable
        big += i
    tl.store(out_ptr + 2, (big >> 32).to(tl.float32))


@pytest.mark.parametrize("n", [0, 3])
def test_loop_variables_are_typed_scalars(mode, n):
    def run():
        out = np.zeros(8, np.float32)
        _loop_variable_kernel[(1,)](out, n)
        return out

    def ref():
        out = np.zeros(8, np.float32)
        out[0] = sum(range(n))
        out[1] = (np.arange(n, dtype=np.int32) * np.int32(1500000000)).sum(dtype=np.int32)
        out[2] = 6
        out[4:4 + n] = np.arange(n) * 0.5
        return out

    check_kernel(run, (), ref, modes=(mode,))


@enceladus.jit
def _nan_argmax_kernel(x_ptr, val_ptr, idx_ptr, amin_ptr, N: tl.constexpr):
    row = tl.program_id(0)
    x = tl.load(x_ptr + row * N + tl.arange(0, N))
    v, i = tl.max(x, 0, return_indices=True)
    tl.store(val_ptr + row, v)
    tl.store(idx_ptr + row, i)
    tl.store(amin_ptr + row, tl.argmin(x, 0))


def test_argmax_ignores_nan_like_max(mode):
    nan = np.nan
    x = np.array([[nan, 1, 3, 2, nan, 0, 3, 1], [nan] * 8, [0, 1, nan, 2, 5, 0, 3, 1],
                  [1, nan, nan, nan, nan, nan, nan, 0]], np.float32)  # fmt: skip

    def run(x):
        v, i, a = np.empty(4, np.float32), np.empty(4, np.int32), np.empty(4, np.int32)
        _nan_argmax_kernel[(4,)](x, v, i, a, N=8)
        return v, i, a

    def ref(x):
        def first(f):  # NaNs don't count unless the whole row is NaN; then the index is 0.
            return np.array([0 if np.isnan(r).all() else f(r) for r in x], np.int32)

        vals = [np.nan if np.isnan(r).all() else np.nanmax(r) for r in x]
        return np.array(vals, np.float32), first(np.nanargmax), first(np.nanargmin)

    check_kernel(run, (x,), ref, modes=(mode,))


@enceladus.jit(do_not_specialize=["n"])
def _chain_kernel(x_ptr, out_ptr, count_ptr, n, TILE: tl.constexpr):
    if TILE:
        x = tl.load(x_ptr + tl.arange(0, 8))
        tl.store(out_ptr + tl.arange(0, 8), (0 < n < x).to(tl.int32))
    else:
        tl.store(out_ptr, (0 <= n < 4).to(tl.int32))
        # Python semantics: the atomic runs only when `n < 0` is true.
        tl.store(out_ptr + 1, (n < 0 < tl.atomic_add(count_ptr, 1)).to(tl.int32))


@pytest.mark.parametrize("n", [-1, 2, 5])
def test_scalar_comparison_chains_short_circuit(n, monkeypatch):
    monkeypatch.setenv("ENCELADUS_VERIFY", "0")
    out, count = np.zeros(8, np.int32), np.zeros(1, np.int32)
    with execution_mode("interpret"):
        _chain_kernel[(1,)](np.zeros(8, np.float32), out, count, n, TILE=False)
    assert out[0] == (0 <= n < 4) and out[1] == 0 and count[0] == (n < 0)


def test_tile_comparison_chains_are_refused(monkeypatch):
    # Python would return the tile `n < x` unchecked, because `0 < n` is a scalar.
    monkeypatch.setenv("ENCELADUS_VERIFY", "0")
    with execution_mode("interpret"), pytest.raises(enceladus.CompilationError, match=r"\) & \("):
        _chain_kernel[(1,)](np.ones(8, np.float32), np.zeros(8, np.int32),
                            np.zeros(1, np.int32), 2, TILE=True)  # fmt: skip


@enceladus.jit
def _pointer_where_kernel(x_ptr, y_ptr, out_ptr, SAME: tl.constexpr):
    offs = tl.arange(0, 8)
    other = x_ptr + 8 + offs if SAME else y_ptr + offs
    tl.store(out_ptr + offs, tl.load(tl.where(offs % 2 == 0, x_ptr + offs, other)))


def test_where_selects_pointers_with_one_base(monkeypatch):
    monkeypatch.setenv("ENCELADUS_VERIFY", "0")
    x, out = np.arange(16, dtype=np.float32), np.zeros(8, np.float32)
    with execution_mode("interpret"):
        _pointer_where_kernel[(1,)](x, x, out, SAME=True)
        np.testing.assert_array_equal(out, np.where(np.arange(8) % 2 == 0, x[:8], x[8:]))
        with pytest.raises(enceladus.CompilationError, match="same base pointer"):
            _pointer_where_kernel[(1,)](x, x, out, SAME=False)


@enceladus.jit
def _small_k_dot_kernel(x_ptr, out_ptr):
    rm, rk = tl.arange(0, 16), tl.arange(0, 4)
    a = tl.load(x_ptr + rm[:, None] * 4 + rk[None, :])
    b = tl.load(x_ptr + rk[:, None] * 16 + rm[None, :])
    tl.store(out_ptr + rm[:, None] * 16 + rm[None, :], tl.dot(a, b))  # refused-line


def test_interpreter_warns_about_kernels_that_compiled_mode_refuses(monkeypatch):
    # Codegen refuses a K block of 4; the interpreter runs it, warning once per kernel.
    x, out = np.ones(64, np.float32), np.zeros(256, np.float32)
    monkeypatch.delenv("ENCELADUS_VERIFY", raising=False)
    with execution_mode("interpret"), pytest.warns(UserWarning, match="K block") as rec:
        _small_k_dot_kernel[(1,)](x, out)
        _small_k_dot_kernel[(1,)](x, out)
    assert len(rec) == 1 and rec[0].filename == __file__
    np.testing.assert_array_equal(out, np.full(256, 4, np.float32))
    monkeypatch.setenv("ENCELADUS_VERIFY", "1")
    with execution_mode("interpret"), pytest.raises(enceladus.CompilationError) as e:
        _small_k_dot_kernel[(1,)](x, out)
    assert "# refused-line" in str(e.value)


@enceladus.jit
def _barrier_kernel(x_ptr, buf_ptr, out_ptr, rounds, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    base = tl.program_id(0) * BLOCK
    x = tl.load(x_ptr + base + offs)
    for _ in range(rounds):
        tl.store(buf_ptr + base + offs, x)
        tl.debug_barrier()  # the reversed load reads other threads' stores
        x = tl.load(buf_ptr + base + (BLOCK - 1 - offs)) + 1
        tl.debug_barrier()  # every thread loads before the next round overwrites
    tl.store(out_ptr + base + offs, x)


def test_debug_barrier_orders_stores_before_loads_by_other_threads(mode):
    # Without the barriers, the GPU loads stale values for about half of the elements.
    def run(x):
        out = np.empty_like(x)
        _barrier_kernel[(16,)](x, np.zeros_like(x), out, 8, BLOCK=1024)
        return out

    check_kernel(run, (np.arange(16 * 1024, dtype=np.int32),), lambda x: x + 8, modes=(mode,))
