"""Tests for AxisInfo, the `tl.multiple_of` and `tl.max_contiguous` hints, and the vector
loads and stores that codegen emits from their facts."""

import numpy as np
import pytest
from conftest import check_kernel, execution_mode

import enceladus
import enceladus.language as tl
from enceladus.compiler import ir
from enceladus.compiler.errors import CompilationError, DeviceAssertionError
from enceladus.compiler.passes.axis_info import MAX_DIV, AxisAnalysis


@enceladus.jit
def _probe(p, q, n, CASE: tl.constexpr):
    r = tl.arange(0, 16)
    if CASE == "mul":
        v = r * 8
    elif CASE == "expand":
        v = r[:, None] + tl.zeros((16, 16), tl.int32)
    elif CASE == "loop_scaled":
        v = r
        for _ in range(n):
            v = v * 2
    elif CASE == "loop_shifted":
        v = r
        for _ in range(n):
            v = v + 16
    elif CASE == "hints":
        v = tl.max_contiguous(tl.multiple_of(tl.load(q + r), 16), 16)
    elif CASE == "loop_counter":
        v = r
        for k in range(0, n, 32):
            v = k + r
    elif CASE == "lt":
        v = tl.program_id(0) * 16 + r < n
    else:
        v = tl.program_id(0) * 16 + r <= n
    tl.store(p + tl.zeros(v.shape, tl.int32), v)


# (case, n, contiguity, divisibility, constancy) of the stored value.
_PROBES = [
    ("mul", 3, (1,), (8,), (1,)),  # every value, not just the first, is a multiple of 8
    ("expand", 3, (16, 1), (MAX_DIV, 1), (1, 16)),  # 0, 1, 2, ... divide by nothing
    ("loop_scaled", 3, (1,), (1,), (1,)),  # the first pass keeps r's contiguity
    ("loop_shifted", 3, (16,), (16,), (1,)),
    ("hints", 3, (16,), (16,), (1,)),
    ("loop_counter", 3, (16,), (32,), (1,)),
    ("lt", 64, (1,), (1,), (16,)),  # x < n is constant on blocks that divide n
    ("lt", 65, (1,), (1,), (1,)),
    ("le", 64, (1,), (1,), (1,)),  # 63 <= 64 but 64 <= 64 and 65 > 64 in one block
]


@pytest.mark.parametrize(("case", "n", "contig", "div", "const"), _PROBES)
def test_axis_info_facts_hold_for_every_value(case, n, contig, div, const):
    m = _probe.ir(ir.PointerType(ir.i32), ir.PointerType(ir.i32), n, CASE=case)
    store = next(op for op in m.walk() if op.name == "store")
    info = AxisAnalysis(m).get(store.operands[1])
    assert (info.contiguity, info.divisibility, info.constancy) == (contig, div, const)


@enceladus.jit
def _scale_rows(x_ptr, y_ptr, M, N, sx, sy, BM: tl.constexpr, BN: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    x = tl.load(x_ptr + rm[:, None] * sx + rn[None, :], mask=mask, other=3)
    tl.store(y_ptr + rm[:, None] * sy + rn[None, :], x * 2 + 1, mask=mask)


def _vector_accesses(msl: str) -> tuple[int, int]:
    """Returns the number of vector loads and vector stores in a kernel's source."""
    return msl.count("*(device const "), msl.count("*(device ") - msl.count("*(device const ")


# (case, M, N, x row stride, x offset, vector loads, vector stores)
_ROWS = [
    ("aligned", 20, 64, 64, 0, 1, 1),
    ("odd_stride", 20, 64, 65, 0, 0, 1),  # rows of x start at unaligned addresses
    ("offset_view", 20, 64, 64, 1, 0, 1),  # x starts one element into its buffer
    ("ragged_cols", 20, 50, 64, 0, 0, 0),  # the mask changes inside a vector
]


@pytest.mark.parametrize("dtype", [np.float32, np.float16, np.int8, np.int64])
@pytest.mark.parametrize(("case", "m", "n", "sx", "off", "vloads", "vstores"), _ROWS)
def test_vector_accesses_match_scalar_ones(rng_np, dtype, case, m, n, sx, off, vloads, vstores):
    base = (rng_np.standard_normal(m * sx + 8) * 20).astype(dtype)
    x = base[off:off + m * sx].reshape(m, sx)

    def run(x):
        y = np.zeros((m, 64), dtype)
        _scale_rows[(enceladus.cdiv(m, 8),)](x, y, m, n, sx, 64, BM=8, BN=64)
        return y

    def reference(x):
        y = np.zeros((m, 64), dtype)
        y[:, :n] = x[:, :n] * dtype(2) + dtype(1)
        return y

    check_kernel(run, (x,), reference)
    ck = _scale_rows.warmup(x, np.zeros((m, 64), dtype), m, n, sx, 64, BM=8, BN=64)
    assert _vector_accesses(ck.msl) == (vloads, vstores)


@enceladus.jit(do_not_specialize=["start"])
def _hinted_copy(x_ptr, y_ptr, i_ptr, start, CASE: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.arange(0, BLOCK)
    if CASE == "scalar":
        offs = tl.multiple_of(start, BLOCK) + r
    elif CASE == "tile":
        offs = tl.multiple_of(start + r, BLOCK)
    else:  # gather through an index tile that holds runs of consecutive indices
        offs = tl.max_contiguous(tl.multiple_of(tl.load(i_ptr + r), 4), 4)
    tl.store(y_ptr + r, tl.load(x_ptr + offs) + 1.0)


@pytest.mark.parametrize("case", ["scalar", "tile", "gather"])
def test_hints_enable_vector_loads_and_are_checked_in_debug_mode(mode, rng_np, monkeypatch,
                                                                 case):  # fmt: skip
    # One SIMD group of 32 threads, each with 4 consecutive elements of the 128.
    x = rng_np.standard_normal(512).astype(np.float32)
    starts = rng_np.choice(np.arange(0, 508, 4), 32)
    good = (starts[:, None] + np.arange(4)).reshape(-1).astype(np.int32)
    bad = good.copy()
    bad[5] += 1  # breaks the run of indices 4 to 7
    start_good, start_bad = 128, 130

    def run(x, idx, start):
        y = np.zeros(128, np.float32)
        _hinted_copy[(1,)](x, y, idx, start, CASE=case, BLOCK=128, num_warps=1)
        return y

    if mode == "interpret":
        monkeypatch.setenv("ENCELADUS_DEBUG", "1")  # checks the promises
    offs = good if case == "gather" else start_good + np.arange(128)
    check_kernel(run, (x, good, start_good), lambda x, i, s: x[offs] + 1.0, modes=(mode,))
    if mode == "compiled":
        ck = _hinted_copy.warmup(x, x, good, start_good, CASE=case, BLOCK=128, num_warps=1)
        assert _vector_accesses(ck.msl) == ((2 if case == "gather" else 1), 1)
        return
    with execution_mode("interpret"), pytest.raises(DeviceAssertionError, match="tl.m"):
        run(x, bad, start_bad)


@enceladus.jit
def _misused_hint(p, CASE: tl.constexpr):
    r = tl.arange(0, 16)
    if CASE == "rank":
        v = tl.multiple_of(r, (16, 16))
    else:
        v = tl.max_contiguous(r.to(tl.float32), 16)
    tl.store(p + r, v)


@pytest.mark.parametrize(("case", "phrase", "line"), [
    ("rank", "needs one value per dimension", 4),
    ("float", "needs an integer or pointer", 6),
])  # fmt: skip
def test_hint_misuse_is_refused(case, phrase, line):
    with pytest.raises(CompilationError, match=phrase) as e:
        _misused_hint.ir(ir.PointerType(ir.f32), CASE=case)
    assert e.value.loc.line == _misused_hint.fn.__code__.co_firstlineno + line


@pytest.fixture
def rng_np():
    return np.random.default_rng(0)
