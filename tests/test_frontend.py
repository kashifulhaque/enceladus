"""Frontend tests: source-located errors, type promotion, specialization, and the verifier."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from conftest import MODES, check_kernel, load_example

import enceladus
import enceladus.language as tl
from enceladus.compiler import ir
from enceladus.compiler.semantic import computation_dtype

X = np.zeros(64, np.float32)

# ---------------------------------------------------------------------------
# Frontend errors. Each kernel marks its offending line with `# error`. test_errors.py
# covers the errors that a launch reports.
# ---------------------------------------------------------------------------


@enceladus.jit
def _non_pow2_zeros(x_ptr, BLOCK: tl.constexpr):
    acc = tl.zeros((BLOCK, 3), dtype=tl.float32)  # error
    tl.store(x_ptr + tl.arange(0, BLOCK), tl.sum(acc, axis=1))


@enceladus.jit
def _break_in_loop(x_ptr, n):
    for i in range(n):
        if i > 3:
            break  # error
    tl.store(x_ptr, 0.0)


@enceladus.jit
def _if_on_tile(x_ptr, BLOCK: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))
    if x > 0:  # error
        x = -x
    tl.store(x_ptr + tl.arange(0, BLOCK), x)


@enceladus.jit
def _loop_type_change(x_ptr, n, BLOCK: tl.constexpr):
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for _ in range(n):  # error
        acc = (acc + 1.0).to(tl.float16)
    tl.store(x_ptr + tl.arange(0, BLOCK), acc)


@enceladus.jit
def _or_changes_type(x_ptr, n):
    tl.store(x_ptr, n or 2.5)  # error


@enceladus.jit
def _returns_value(x_ptr):
    return tl.load(x_ptr)  # error


@enceladus.jit
def _recursive(x):
    return _recursive(x)  # error


@enceladus.jit
def _calls_recursive(x_ptr):
    tl.store(x_ptr, _recursive(1.0))


ERROR_CASES = [
    (_non_pow2_zeros, {"BLOCK": 64}, "dimension 1"),
    (_break_in_loop, {}, "`break` isn't supported"),
    (_if_on_tile, {"BLOCK": 64}, "tl.where"),
    (_loop_type_change, {"BLOCK": 64}, "loop-carried variable `acc`"),
    (_or_changes_type, {}, "both need the same type"),
    (_returns_value, {}, "can't return values"),
    (_calls_recursive, {}, "recursion isn't supported"),
]


@pytest.mark.parametrize("kernel, constexprs, phrase", ERROR_CASES,
                         ids=[c[0].__name__ for c in ERROR_CASES])  # fmt: skip
def test_error_points_at_source_line(kernel, constexprs, phrase):
    n_runtime = sum(not p.is_constexpr for p in kernel.params)
    args = [X, 64, 64][:n_runtime]
    with pytest.raises(enceladus.CompilationError) as e:
        kernel.ir(*args, **constexprs)
    msg = str(e.value)
    assert phrase in msg
    # The error names file:line of the marked line, shows that line, and puts a caret under it.
    fn = _recursive if kernel is _calls_recursive else kernel.fn
    lines, start = inspect.getsourcelines(fn)
    offset = next(i for i, text in enumerate(lines) if text.rstrip().endswith("# error"))
    assert f"{__file__}:{start + offset}:" in msg
    assert lines[offset].rstrip() in msg.splitlines()
    assert msg.splitlines()[-1].strip() == "^"


# ---------------------------------------------------------------------------
# Python semantics in compiled kernels. Each case once gave a wrong answer without an error.
# ---------------------------------------------------------------------------


@enceladus.jit
def _loop_var_after_loop(out_ptr, n):
    i = 100
    for i in range(n):  # noqa: B007 - `i` is read after the loop.
        pass
    tl.store(out_ptr, i)  # Python: n - 1, or 100 when the loop doesn't run.


@enceladus.jit
def _negate_mask(out_ptr, x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = tl.load(x_ptr + offs) > 0
    tl.store(out_ptr + offs, -m + 10 * ~m)  # `-` computes in int32; `~` is logical.


@enceladus.jit
def _unsigned_loop_bound(out_ptr, n_ptr):
    n = tl.load(n_ptr)
    count = 0
    for _ in range(0, n, 1_000_000_000):  # An int32 counter would see n < 0.
        count += 1
    tl.store(out_ptr, count)


@enceladus.jit
def _chained_compare(out_ptr, x_ptr, a, b):
    tl.store(out_ptr, (a < b <= tl.load(x_ptr)).to(tl.int32))


@enceladus.jit
def _where_same_base(out_ptr, x_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.load(tl.where(offs % 2 == 0, x_ptr + offs, x_ptr + 50 + offs)))


def _launch(kernel, out, *args, **kwargs):
    kernel[(1,)](out, *args, **kwargs)
    return out


_I32 = np.zeros(1, np.int32)
_ARANGE = np.arange(-32, 96, dtype=np.int32)
SEMANTICS_CASES = [
    ("loop_var_after_loop", lambda: _launch(_loop_var_after_loop, _I32.copy(), 5), [4], MODES),
    ("loop_var_after_empty_loop", lambda: _launch(_loop_var_after_loop, _I32.copy(), 0), [100],
     MODES),
    ("negate_mask", lambda: _launch(_negate_mask, np.zeros(64, np.int32), _ARANGE, BLOCK=64),
     np.where(_ARANGE[:64] > 0, -1, 10), MODES),
    ("uint32_loop_bound", lambda: _launch(_unsigned_loop_bound, _I32.copy(),
                                          np.array([3_000_000_000], np.uint32)), [3], MODES),
    ("chained_compare", lambda: _launch(_chained_compare, _I32.copy(), _ARANGE[40:], 1, 2), [1],
     MODES),
    ("chained_compare_short_circuits", lambda: _launch(_chained_compare, _I32.copy(),
                                                       _ARANGE[40:], 3, 2), [0], MODES),
    # The interpreter doesn't support tl.where on pointers.
    ("where_same_base_pointers", lambda: _launch(_where_same_base, np.zeros(64, np.int32),
                                                 _ARANGE, BLOCK=64),
     np.where(np.arange(64) % 2 == 0, _ARANGE[:64], _ARANGE[50:114]), ("compiled",)),
]  # fmt: skip


@pytest.mark.parametrize("run, expected, modes", [c[1:] for c in SEMANTICS_CASES],
                         ids=[c[0] for c in SEMANTICS_CASES])  # fmt: skip
def test_compiled_matches_python_semantics(run, expected, modes):
    check_kernel(run, (), lambda: np.asarray(expected, np.int32), modes=modes)


# ---------------------------------------------------------------------------
# Type promotion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op, a, b, expected",
    [
        ("add", tl.float16, tl.bfloat16, tl.float32),
        ("add", tl.int32, tl.float16, tl.float16),
        ("mul", tl.int8, 3, tl.int8),  # An int literal adopts the tile's integer type.
        ("mul", tl.int8, 1000, tl.int32),  # ... unless it doesn't fit.
        ("add", tl.int32, 0.5, tl.float32),
        ("add", tl.float16, 2, tl.float16),
        ("add", tl.uint32, tl.int32, tl.uint32),
        ("add", tl.int64, tl.uint32, tl.int64),
        ("add", tl.int1, tl.int1, tl.int32),
        ("and", tl.int1, tl.int1, tl.int1),
        ("div", tl.int32, tl.int32, tl.float32),
        ("lt", tl.int16, tl.int64, tl.int64),
    ],
)
def test_promotion(op, a, b, expected):
    assert computation_dtype(op, a, b) is expected
    assert computation_dtype(op, b, a) is expected


def test_promotion_rejects_float_bitwise_and_floordiv():
    with pytest.raises(enceladus.CompilationError, match="integer operands"):
        computation_dtype("and", tl.float32, tl.int32)
    with pytest.raises(enceladus.CompilationError, match="tl.floor"):
        computation_dtype("floordiv", tl.float32, 2)


# ---------------------------------------------------------------------------
# Specialization and IR structure
# ---------------------------------------------------------------------------


@enceladus.jit(do_not_specialize=["m"])
def _facts_kernel(x_ptr, n, m, one, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    tl.store(x_ptr + offs * one, 1.0, mask=offs < n + m)


def test_specialization_facts_are_attributes():
    module = _facts_kernel.ir(X, 32, 48, 1, BLOCK=64)
    func = module.func
    assert func.attrs["arg_names"] == ["x_ptr", "n", "m", "one"]  # Facts never drop args.
    facts = dict(zip(func.attrs["arg_names"], func.attrs["arg_attrs"], strict=True))
    assert facts["x_ptr"] == {"divisibility": 16}
    assert facts["n"] == {"divisibility": 16}
    assert facts["m"] == {}  # do_not_specialize
    assert facts["one"] == {"equal_to_1": True}
    assert [a.type for a in module.body.args] == [ir.PointerType(ir.f32), ir.i32, ir.i32, ir.i32]
    assert _facts_kernel.ir(X, 1 << 40, 3, 5, BLOCK=64).body.args[1].type == ir.i64


def test_descriptor_requires_unit_last_stride():
    ex_kernel = _desc_kernel
    ex_kernel.ir(X, 8, 8, 1, BLOCK=8)  # A specialized stride == 1 is accepted.
    with pytest.raises(enceladus.CompilationError, match="last stride"):
        ex_kernel.ir(X, 8, 8, 2, BLOCK=8)


@enceladus.jit
def _desc_kernel(x_ptr, m, n, stride_n, BLOCK: tl.constexpr):
    d = tl.make_tensor_descriptor(x_ptr, [m, n], [n, stride_n], [BLOCK, BLOCK])
    d.store([0, 0], d.load([0, 0]) * 2)


def _corrupt_use_before_def(m: ir.Module) -> None:
    ops = m.body.ops
    i = next(k for k, op in enumerate(ops) if op.operands and op.operands[0].defining_op)
    j = ops.index(ops[i].operands[0].defining_op)
    ops[i], ops[j] = ops[j], ops[i]


def _corrupt_type(m: ir.Module) -> None:
    op = next(op for op in m.walk() if op.name == "binary")
    op.results[0].type = ir.TileType((8,), ir.f16)


def _corrupt_terminator(m: ir.Module) -> None:
    m.body.ops.pop()


@pytest.mark.parametrize("corrupt, phrase", [
    (_corrupt_use_before_def, "isn't defined before this use"),
    (_corrupt_type, "types must match"),
    (_corrupt_terminator, "must end with `return`"),
])  # fmt: skip
def test_verifier_rejects_malformed_ir(corrupt, phrase):
    ex = load_example("01_vector_add")
    module = ex.add_kernel.ir(X, X, X, 64, BLOCK=64)
    ir.verify(module)
    corrupt(module)
    with pytest.raises(enceladus.CompilationError, match=phrase):
        ir.verify(module)
