"""Constant folding and algebraic identities: compiled results must match the interpreter."""

import numpy as np
import pytest
from conftest import check_kernel

import enceladus
import enceladus.language as tl
from enceladus.compiler import ir
from enceladus.compiler.passes.simplify import simplify

B = 8
I32_EDGES = np.array([-(2**31), 2**31 - 1, 0, -1, 1, 7, -7, 12345], np.int32)
F32_EDGES = np.array([np.nan, np.inf, -np.inf, -0.0, 0.0, 1e38, -1.5, 3.0], np.float32)
INT_ROWS, FLOAT_ROWS = 26, 22


@enceladus.jit
def _fold_edges(xi_ptr, xf_ptr, n, oi_ptr, of_ptr, B: tl.constexpr):
    r = tl.arange(0, B)
    xi = tl.load(xi_ptr + r)
    xf = tl.load(xf_ptr + r)
    imin = tl.full((B,), -2147483648, tl.int32)
    imax = tl.full((B,), 2147483647, tl.int32)
    one = tl.full((B,), 1, tl.int32)
    ints = (
        imin + (one - 2), imax * 2, imin - one, -imax, imin // 7, (one - 8) % 3, imin >> 31,
        one << 31, tl.full((B,), -1, tl.int32).to(tl.uint32).to(tl.int64).to(tl.int32),
        tl.full((B,), 300, tl.int32).to(tl.int8).to(tl.int32),
        tl.full((B,), 2.5e9, tl.float32).to(tl.uint32).to(tl.int32),
        tl.full((B,), -3.99, tl.float32).to(tl.int32),
        (tl.full((B,), 4294967295, tl.uint32) > 1).to(tl.int32),
        (imax > imin).to(tl.int32),
        tl.where(imax > 0, xi, 5), tl.where(imax < 0, xi, 5),
        # Identities on runtime values, including INT_MIN and overflow.
        xi * 0, xi + 0, xi * 1, xi - xi, xi ^ xi, xi | 0, xi & -1, xi // 1, xi % 1, xi << 0,
    )  # fmt: skip
    for k in tl.static_range(len(ints)):
        tl.store(oi_ptr + k * B + r, ints[k])
    fmax = tl.full((B,), 3.4e38, tl.float32)
    nan = tl.full((B,), float("nan"), tl.float32)
    nz = tl.full((B,), -0.0, tl.float32)
    xh = xf.to(tl.float16)
    floats = (
        # `x * 0.0` and `x - x` must stay: NaN and infinity break them.
        xf * 0.0, xf - xf, xf + (-0.0), xf - 0.0, xf * 1.0, xf / 1.0,
        fmax * 10.0, nz + 0.0, nz * 5.0, nan + 1.0,
        (nan == nan).to(tl.float32), (nan != nan).to(tl.float32),
        tl.full((B,), 70000.0, tl.float32).to(tl.float16).to(tl.float32),
        tl.full((B,), 1.0 / 3.0, tl.float32).to(tl.float16).to(tl.float32),
        (tl.full((B,), 65504.0, tl.float16) * 2.0).to(tl.float32),
        tl.maximum(nan, 2.0), tl.floor(tl.full((B,), -2.5, tl.float32)),
        tl.full((B,), 3, tl.int32).to(tl.float32) / nz,
        tl.full((B,), 16777217, tl.int32).to(tl.float32),
        xh.to(tl.float32).to(tl.float16).to(tl.float32),
        tl.where(nan == nan, xf, 2.0), (nz - 0.0) * -1.0,
    )  # fmt: skip
    for k in tl.static_range(len(floats)):
        v = floats[k]
        # Compare bits, so -0.0 differs from 0.0; every NaN counts as the same NaN.
        bits = tl.where(v != v, 0x7FC00000, v.to(tl.int32, bitcast=True))
        tl.store(of_ptr + k * B + r, bits)
    # Branches on conditions that fold to constants.
    if n * 0 == 0:
        tl.store(oi_ptr + len(ints) * B + r, xi + 1)
    else:
        tl.store(oi_ptr + len(ints) * B + r, xi - 1)
    if n - n != 0:
        w = -xi
    else:
        w = xi * 3
    tl.store(oi_ptr + (len(ints) + 1) * B + r, w)


def _run(xi, xf, n):
    oi = np.zeros((INT_ROWS + 2, B), np.int32)
    of = np.zeros((FLOAT_ROWS, B), np.int32)
    _fold_edges[(1,)](xi, xf, n, oi, of, B=B)
    return oi, of


def _bits(v):
    v = np.asarray(v, np.float32)
    b = v.view(np.int32).copy()
    b[np.isnan(v)] = 0x7FC00000
    return b


def _reference(xi, xf, n):
    full = lambda v, dt=np.int32: np.full(B, v, dt)  # noqa: E731
    with np.errstate(all="ignore"):
        x64 = xi.astype(np.int64)
        ints = [
            full(2**31 - 1), full(-2), full(2**31 - 1), full(-(2**31) + 1), full(-306783378),
            full(-1), full(-1), full(-(2**31)), full(-1), full(44), full(-1794967296), full(-3),
            full(1), full(1), xi, full(5),
            full(0), xi, xi, full(0), full(0), xi, xi, xi, full(0), xi,
            xi + 1, (x64 * 3).astype(np.int32),
        ]
        f = xf
        floats = [
            f * np.float32(0), f - f, f, f, f, f,
            full(np.inf, np.float32), full(0.0, np.float32), full(-0.0, np.float32),
            full(np.nan, np.float32), full(0.0, np.float32), full(1.0, np.float32),
            full(np.inf, np.float32),
            full(np.float32(np.float16(np.float32(1 / 3))), np.float32),
            full(np.inf, np.float32), full(2.0, np.float32), full(-3.0, np.float32),
            full(-np.inf, np.float32), full(16777216.0, np.float32),
            f.astype(np.float16).astype(np.float32), full(2.0, np.float32),
            full(0.0, np.float32),
        ]
    return np.stack(ints), np.stack([_bits(v) for v in floats])


def test_folding_matches_the_interpreter_on_edge_values():
    check_kernel(_run, (I32_EDGES, F32_EDGES, 5), _reference)


@enceladus.jit
def _all_constant(out_ptr, n, B: tl.constexpr):
    r = tl.arange(0, B)
    c = tl.full((B,), 7, tl.int32)
    v = ((c * 3 - 1) // 4).to(tl.float32) + 0.5
    if (n * 0 + 2) * 1 > 1:
        tl.store(out_ptr + r, tl.where(v > 5.0, v, -v))
    else:
        tl.store(out_ptr + r, v)


def test_constant_expressions_fold_away():
    mod = _all_constant.ir(np.zeros(B, np.float32), 3, B=B)
    simplify(mod)
    names = [op.name for op in mod.walk()]
    assert not {"binary", "cmp", "cast", "select", "if"} & set(names), names
    out = np.zeros(B, np.float32)
    _all_constant[(1,)](out, 3, B=B)
    np.testing.assert_array_equal(out, np.full(B, 5.5, np.float32))


@enceladus.jit
def _identities(x_ptr, out_ptr, B: tl.constexpr, INT: tl.constexpr):
    r = tl.arange(0, B)
    x = tl.load(x_ptr + r)
    if INT:
        y = ((x * 1 + 0) | 0) & -1
    else:
        y = (x * 1.0 - 0.0) / 1.0
    tl.store(out_ptr + r, y)


@pytest.mark.parametrize("dtype", [np.float16, np.int8, np.uint32])
def test_identities_keep_the_dtype(dtype):
    # Identities on narrow and unsigned types, where a replacement of the wrong type would
    # fail verification or store the wrong width.
    x = (np.arange(B) - 3).astype(dtype)

    def run(x):
        out = np.zeros_like(x)
        _identities[(1,)](x, out, B=B, INT=np.dtype(dtype).kind in "iu")
        return out

    check_kernel(run, (x,), lambda x: x)
    mod = _identities.ir(x, x, B=B, INT=np.dtype(dtype).kind in "iu")
    simplify(mod)
    assert not any(op.name == "binary" for op in mod.walk())


@enceladus.jit
def _loop_invariants(x_ptr, out_ptr, n, B: tl.constexpr):
    r = tl.arange(0, B)
    x = tl.load(x_ptr + r)
    acc = tl.zeros((B,), tl.float32)
    s = 1
    for i in range(n):
        for j in range(2):
            # `n * 3 + 1` is invariant in both loops, `i * 2` only in the inner one, and
            # `s` changes every iteration.
            acc += x * (n * 3 + 1) + i * 2 + j
        s = s * 2 + 1
    tl.store(out_ptr + r, acc + s)


def test_loop_invariant_scalars_leave_their_loops():
    x = np.arange(B, dtype=np.float32)
    n = 3

    def run(x):
        out = np.zeros(B, np.float32)
        _loop_invariants[(1,)](x, out, n, B=B)
        return out

    def reference(x):
        acc = sum(x * (3 * n + 1) + 2 * i + j for i in range(n) for j in range(2))
        return (acc + 2 ** (n + 1) - 1).astype(np.float32)

    check_kernel(run, (x,), reference)
    mod = _loop_invariants.ir(x, x, n, B=B)
    simplify(mod)
    outer = next(op for op in mod.body.ops if op.name == "for")
    inner = next(op for op in outer.regions[0].block.ops if op.name == "for")
    scalar_ops = lambda loop: [op.attrs.get("op") for op in loop.regions[0].block.ops  # noqa: E731
                               if op.name == "binary" and not ir.shape_of(op.result.type)]
    assert scalar_ops(inner) == []  # `i * 2` and its conversion moved to the outer loop
    assert scalar_ops(outer) == ["mul", "mul", "add"]  # `i * 2` and `s * 2 + 1` stay
