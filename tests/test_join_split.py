"""Differential tests for `tl.join` and `tl.split` across layouts and producers."""

import ml_dtypes
import numpy as np
import pytest
from conftest import check_kernel

import enceladus
import enceladus.language as tl

F32, F16, BF16, I32 = np.float32, np.float16, ml_dtypes.bfloat16, np.int32


@pytest.fixture
def rng_np():
    return np.random.default_rng(0)


@enceladus.jit
def _join_split(x_ptr, y_ptr, out_ptr, CASE: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    rm, rn = tl.arange(0, M), tl.arange(0, N)
    offs = rm[:, None] * N + rn[None, :]
    if CASE == "interleave":
        # Loaded operands: the new dimension is a register bit, so the join is free.
        x = tl.load(x_ptr + offs)
        y = tl.load(y_ptr + offs)
        r2 = tl.arange(0, 2 * N)
        tl.store(out_ptr + rm[:, None] * (2 * N) + r2[None, :],
                 tl.reshape(tl.join(x, y), (M, 2 * N)))
    elif CASE == "deinterleave":
        r2 = tl.arange(0, 2 * N)
        x = tl.load(x_ptr + rm[:, None] * (2 * N) + r2[None, :])
        a, b = tl.split(tl.reshape(x, (M, N, 2)))
        tl.store(out_ptr + offs, a)
        tl.store(out_ptr + M * N + offs, b)
    elif CASE == "lane_split":
        # After the transpose, a lane bit selects the last coordinate, so the split
        # exchanges data through threadgroup memory.
        x = tl.load(x_ptr + tl.arange(0, 2)[:, None] * M + rm[None, :])
        a, b = x.T.split()
        tl.store(out_ptr + rm, a)
        tl.store(out_ptr + M + rm, b)
    elif CASE == "tiny_split":
        # Fewer elements than threads: the loaded tile has no register bits.
        x = tl.load(x_ptr + rm[:, None] * 2 + tl.arange(0, 2)[None, :])
        a, b = tl.split(x)
        tl.store(out_ptr + rm, b)
        tl.store(out_ptr + M + rm, a)
    elif CASE == "reduce":
        x = tl.load(x_ptr + offs)
        j = tl.join(tl.sum(x, axis=1), tl.max(x, axis=1))
        tl.store(out_ptr + rm[:, None] * 2 + tl.arange(0, 2)[None, :], j)
        lo, hi = tl.split(j)
        tl.store(out_ptr + 2 * M + rm, hi - lo)
    elif CASE == "loop":
        x = tl.load(x_ptr + offs)
        acc = tl.zeros((M, N, 2), x.dtype)
        for i in range(3):
            acc = acc + tl.join(x, x * i)
        lo, hi = tl.split(acc)
        tl.store(out_ptr + offs, hi)
        tl.store(out_ptr + M * N + offs, lo)
    elif CASE == "cheap":
        # Values rebuilt in each consumer's layout, including one where a lane bit selects
        # the joined operand (16 x 2 elements over 128 threads).
        r16, r256 = tl.arange(0, 16), tl.arange(0, 256)
        c2 = tl.arange(0, 2)[None, :]
        tl.store(out_ptr + r16[:, None] * 2 + c2, tl.join(r16, r16 * 3 + 1).to(x_ptr.dtype))
        tl.store(out_ptr + 32 + r256[:, None] * 2 + c2, tl.join(r256 - 7, r256).to(x_ptr.dtype))
        e, o = tl.split(tl.reshape(tl.arange(0, 2 * N), (N, 2)))
        tl.store(out_ptr + 544 + rn, (e * 2 + o).to(x_ptr.dtype))
    else:  # "scalar"
        v = tl.load(x_ptr)
        tl.store(out_ptr + tl.arange(0, 2), tl.join(v, 5))
        a, b = tl.split(tl.load(x_ptr + 1 + tl.arange(0, 2)))
        tl.store(out_ptr + 2, b)
        tl.store(out_ptr + 3, a)


def _reference(case, x, y, m, n):
    if case == "interleave":
        return np.stack([x, y], -1).reshape(m, 2 * n)
    if case == "deinterleave":
        return x.reshape(m, n, 2).transpose(2, 0, 1)
    if case == "lane_split":
        return x
    if case == "tiny_split":
        return x.T[::-1]
    if case == "reduce":
        x64 = x.astype(np.float64)
        s, mx = x64.sum(1), x64.max(1)
        return np.concatenate([np.stack([s, mx], -1).ravel(), mx - s]).astype(x.dtype)
    if case == "loop":
        x64 = x.astype(np.float64)
        return np.stack([3 * x64, 3 * x64]).astype(x.dtype)
    if case == "cheap":
        r16, r256, r = np.arange(16), np.arange(256), np.arange(2 * n)
        return np.concatenate([np.stack([r16, 3 * r16 + 1], -1).ravel(),
                               np.stack([r256 - 7, r256], -1).ravel(),
                               r[0::2] * 2 + r[1::2]]).astype(x.dtype)  # fmt: skip
    return np.array([x[0], 5, x[2], x[1]], x.dtype)


# (case, x shape, y shape, M, N)
CASES = {
    "interleave": ((8, 64), (8, 64), 8, 64),
    "deinterleave": ((16, 64), None, 16, 32),
    "lane_split": ((2, 64), None, 64, 1),
    "tiny_split": ((16, 2), None, 16, 1),
    "reduce": ((16, 32), None, 16, 32),
    "loop": ((4, 32), None, 4, 32),
    "cheap": ((1,), None, 1, 8),
    "scalar": ((3,), None, 1, 1),
}


@pytest.mark.parametrize("dtype", [F32, F16, I32])
@pytest.mark.parametrize("case", list(CASES))
def test_join_and_split(rng_np, case, dtype):
    xs, ys, m, n = CASES[case]
    if dtype is I32 and case == "reduce":
        pytest.skip("covered by the float dtypes")
    gen = (lambda s: rng_np.integers(-50, 50, s)) if dtype is I32 else rng_np.standard_normal
    x = gen(xs).astype(dtype)
    y = gen(ys).astype(dtype) if ys else x

    def run(x, y):
        out = np.zeros(_reference(case, x, y, m, n).shape, dtype)
        _join_split[(1,)](x, y, out, CASE=case, M=m, N=n)
        return out

    atol = 0.05 if case == "reduce" and dtype is F16 else None
    check_kernel(run, (x, y), lambda x, y: _reference(case, x, y, m, n), atol=atol)


@enceladus.jit
def _dot_join_split(a_ptr, b_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rm, rn, rk = tl.arange(0, M), tl.arange(0, N), tl.arange(0, K)
    a = tl.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(b_ptr + rk[:, None] * N + rn[None, :])
    acc = tl.dot(a, b)
    # The element bit of the accumulator selects the last coordinate: both ops are free.
    lo, hi = tl.split(tl.reshape(acc, (M, N // 2, 2)))
    swapped = tl.reshape(tl.join(hi, lo * 2), (M, N))
    tl.store(out_ptr + rm[:, None] * N + rn[None, :], swapped)
    offs = (rm[:, None] * N + rn[None, :])[:, :, None] * 2 + tl.arange(0, 2)[None, None, :]
    tl.store(out_ptr + M * N + offs, tl.join(acc, acc + 1))


@pytest.mark.parametrize("dtype", [F32, F16, BF16])
@pytest.mark.parametrize("num_warps", [1, 4])
def test_join_and_split_of_dot_results(rng_np, dtype, num_warps):
    m, n, k = 32, 32, 16
    a = rng_np.standard_normal((m, k)).astype(dtype)
    b = rng_np.standard_normal((k, n)).astype(dtype)

    def run(a, b):
        out = np.zeros(3 * m * n, np.float32)
        _dot_join_split[(1,)](a, b, out, M=m, N=n, K=k, num_warps=num_warps)
        return out

    def reference(a, b):
        c = a.astype(np.float64) @ b.astype(np.float64)
        s = c.reshape(m, n // 2, 2)
        swapped = np.stack([s[..., 1], 2 * s[..., 0]], -1).reshape(m, n)
        return np.concatenate([swapped.ravel(), np.stack([c, c + 1], -1).ravel()]).astype(F32)

    check_kernel(run, (a, b), reference, atol=1e-3, rtol=1e-3)
    ck = _dot_join_split.warmup(a, b, np.zeros(3 * m * n, np.float32), M=m, N=n, K=k,
                                num_warps=num_warps)  # fmt: skip
    # The operands stage through threadgroup memory; the join and split don't add more.
    assert "buf[" not in ck.msl
