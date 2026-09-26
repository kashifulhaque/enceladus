"""Frontend tests: source-located errors, type promotion, specialization, and the verifier."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from conftest import load_example

import enceladus
import enceladus.language as tl
from enceladus.compiler import ir
from enceladus.compiler.semantic import computation_dtype

X = np.zeros(64, np.float32)

# ---------------------------------------------------------------------------
# Errors for common mistakes. Each kernel marks its offending line with `# error`.
# ---------------------------------------------------------------------------


@enceladus.jit
def _missing_constexpr(x_ptr, BLOCK):
    offs = tl.arange(0, BLOCK)  # error
    tl.store(x_ptr + offs, 1.0)


@enceladus.jit
def _non_pow2_arange(x_ptr):
    offs = tl.arange(0, 100)  # error
    tl.store(x_ptr + offs, 1.0)


@enceladus.jit
def _non_pow2_zeros(x_ptr, BLOCK: tl.constexpr):
    acc = tl.zeros((BLOCK, 3), dtype=tl.float32)  # error
    tl.store(x_ptr + tl.arange(0, BLOCK), tl.sum(acc, axis=1))


@enceladus.jit
def _while_loop(x_ptr, n):
    i = 0
    while i < n:  # error
        i += 1
    tl.store(x_ptr, i)


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
    (_missing_constexpr, {}, "BLOCK: tl.constexpr"),
    (_non_pow2_arange, {}, "power of two"),
    (_non_pow2_zeros, {"BLOCK": 64}, "dimension 1"),
    (_while_loop, {}, "`while` loops aren't supported"),
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
