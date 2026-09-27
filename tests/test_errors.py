"""Error messages for common mistakes, and regressions for errors that lost their location.

Each kernel marks its offending line with `# error`. A good error names that line and
says what to do, so each case asserts the line and a phrase from the suggested fix.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import types

import numpy as np
import pytest
from conftest import execution_mode, mode_available

import enceladus
import enceladus.language as tl
from enceladus.compiler.codegen import msl

X = np.zeros(64, np.float32)


@enceladus.jit
def _missing_constexpr(x_ptr, BLOCK):
    tl.store(x_ptr + tl.arange(0, BLOCK), 1.0)  # error


@enceladus.jit
def _non_pow2_arange(x_ptr):
    tl.store(x_ptr + tl.arange(0, 48), 1.0)  # error


@enceladus.jit
def _while_loop(x_ptr, n):
    i = 0
    while i < n:  # error
        i += 1
    tl.store(x_ptr, i)


@enceladus.jit
def _and_on_tiles(x_ptr, BLOCK: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))
    tl.store(x_ptr + tl.arange(0, BLOCK), x, mask=(x > 0) and (x < 1))  # error


@enceladus.jit
def _fp16_atomic_add(p, BLOCK: tl.constexpr):
    tl.atomic_add(p + tl.arange(0, BLOCK), 1.0)  # error


@enceladus.jit
def _pow_on_tile(x_ptr, BLOCK: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))
    tl.store(x_ptr + tl.arange(0, BLOCK), x**2)  # error


@enceladus.jit
def _print_runtime(x_ptr, BLOCK: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))
    print(x)  # error


@enceladus.jit
def _small_dot_k(x_ptr, M: tl.constexpr, K: tl.constexpr):
    rm = tl.arange(0, M)
    rk = tl.arange(0, K)
    a = tl.load(x_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(x_ptr + rk[:, None] * M + rm[None, :])
    tl.store(x_ptr + rm[:, None] * M + rm[None, :], tl.dot(a, b))  # error


@enceladus.jit
def _int_dot_float_acc(x_ptr, out_ptr, N: tl.constexpr):
    r = tl.arange(0, N)
    x = tl.load(x_ptr + r[:, None] * N + r[None, :])
    acc = tl.zeros((N, N), dtype=tl.float32)
    acc = tl.dot(x, x, acc)  # error
    tl.store(out_ptr + r[:, None] * N + r[None, :], acc)


@enceladus.jit
def _tg_overflow(x_ptr, y_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    rm = tl.arange(0, M)
    rn = tl.arange(0, N)
    x = tl.load(x_ptr + rm[:, None] * N + rn[None, :])
    y = tl.load(y_ptr + rn[:, None] * M + rm[None, :])
    tl.store(out_ptr + rm[:, None] * N + rn[None, :], x + tl.trans(y))  # error


@enceladus.jit
def _literal_out_of_range_merge(x_ptr, u_ptr, flag):
    u = tl.load(u_ptr)
    if flag > 0:  # error
        s = u
    else:
        s = 300  # Doesn't fit in u's uint8.
    tl.store(x_ptr, s)


@enceladus.jit
def _full_out_of_range(x_ptr, BLOCK: tl.constexpr):
    tl.store(x_ptr + tl.arange(0, BLOCK), tl.full((BLOCK,), 1e10, tl.int32))  # error


@enceladus.jit
def _loop_var_changes_type(x_ptr, n):
    i = 0.5
    for i in range(n):  # noqa: B007 - `i` is read after the loop.
        pass
    tl.store(x_ptr, i)  # error


@enceladus.jit
def _chained_compare_on_tile(x_ptr, BLOCK: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))
    tl.store(x_ptr + tl.arange(0, BLOCK), x, mask=0 < x < 1)  # error


@enceladus.jit
def _where_different_bases(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    p = tl.where(offs < 8, x_ptr + offs, y_ptr + offs)  # error
    tl.store(x_ptr + offs, tl.load(p))


@enceladus.jit
def _dict_literal(x_ptr):
    scale = {"a": 2.0}  # error
    tl.store(x_ptr, scale["a"])


@enceladus.jit
def _shape_0_tile_as_scalar(x_ptr):
    tl.store(x_ptr, tl.full((), 1.0, tl.float32))  # error


class _Scaler:
    def __call__(self, v):
        return v * 2.0


# A module whose attributes the kernel cache can't track: changes to `ns.X` or to the
# `_Scaler` object's state wouldn't recompile the kernels that read them.
_cfg = types.ModuleType("_cfg")
_cfg.ns = types.SimpleNamespace(X=2.0)
_cfg.scale = _Scaler()
_SCALE = _Scaler()


@enceladus.jit
def _namespace_attribute(x_ptr):
    tl.store(x_ptr, _cfg.ns.X)  # error


def _helper():
    return 1.0


_helper.X = 2.0


@enceladus.jit
def _function_attribute(x_ptr):
    tl.store(x_ptr, _helper.X)  # error


@enceladus.jit
def _callable_instance(x_ptr):
    tl.store(x_ptr, _SCALE(1.0))  # error


@enceladus.jit
def _callable_instance_attribute(x_ptr):
    tl.store(x_ptr, _cfg.scale(1.0))  # error


def _tg_overflow_launch():
    x = np.zeros((128, 128), np.float32)
    _tg_overflow.warmup(x, x, x, M=128, N=128)


# (kernel, launch, modes, phrase). "interpret" runs the NumPy interpreter without building
# the IR first, so it checks the interpreter's own refusals.
KERNEL_CASES = [
    (_missing_constexpr, lambda: _missing_constexpr[(1,)](X, 16), ("interpret", "compiled"),
     "`BLOCK: tl.constexpr`"),
    (_non_pow2_arange, lambda: _non_pow2_arange[(1,)](X), ("interpret", "compiled"),
     "Round the block size up to a power of two"),
    (_while_loop, lambda: _while_loop[(1,)](X, 3), ("compiled",), "Use `for i in range(...)`"),
    (_and_on_tiles, lambda: _and_on_tiles[(1,)](X, BLOCK=16), ("interpret", "compiled"),
     "use `&`"),
    (_fp16_atomic_add, lambda: _fp16_atomic_add[(1,)](np.zeros(16, np.float16), BLOCK=16),
     ("interpret", "compiled"), "Accumulate in a float32 buffer"),
    (_pow_on_tile, lambda: _pow_on_tile[(1,)](X, BLOCK=16), ("interpret", "compiled"),
     "Multiply explicitly"),
    (_print_runtime, lambda: _print_runtime[(1,)](X, BLOCK=16), ("compiled",),
     "Use tl.device_print"),
    (_small_dot_k, lambda: _small_dot_k[(1,)](np.zeros((16, 16), np.float32), M=16, K=4),
     ("compiled",), "Use a K block size of at least 8"),
    (_tg_overflow, _tg_overflow_launch, ("compiled",), "Use smaller blocks"),
    (_int_dot_float_acc,
     lambda: _int_dot_float_acc[(1,)](np.zeros((16, 16), np.int8), np.zeros((16, 16), np.float32),
                                     N=16),
     ("interpret", "compiled"), "`tl.zeros(..., dtype=tl.int32)`"),
    (_literal_out_of_range_merge,
     lambda: _literal_out_of_range_merge[(1,)](X, np.zeros(1, np.uint8), 0), ("compiled",),
     "Convert the tl.uint8 value with `.to(tl.int32)`"),
    (_full_out_of_range, lambda: _full_out_of_range[(1,)](np.zeros(16, np.int32), BLOCK=16),
     ("interpret", "compiled"), "Use a value in that range"),
    (_loop_var_changes_type, lambda: _loop_var_changes_type[(1,)](X, 3), ("compiled",),
     "Give the loop variable a different name"),
    (_chained_compare_on_tile, lambda: _chained_compare_on_tile[(1,)](X, BLOCK=16), ("compiled",),
     "`(a < b) & (b < c)`"),
    (_where_different_bases, lambda: _where_different_bases[(1,)](X, X, BLOCK=16), ("compiled",),
     "derive from `x_ptr` and `y_ptr`"),
    (_dict_literal, lambda: _dict_literal[(1,)](X), ("compiled",), "Use a compile-time tuple"),
    (_shape_0_tile_as_scalar, lambda: _shape_0_tile_as_scalar[(1,)](X), ("compiled",),
     "Use a scalar instead"),
    (_namespace_attribute, lambda: _namespace_attribute[(1,)](X), ("compiled",),
     "pass the value as a tl.constexpr argument"),
    (_function_attribute, lambda: _function_attribute[(1,)](X), ("compiled",),
     "Read attributes only of modules, classes, and constant values"),
    (_callable_instance, lambda: _callable_instance[(1,)](X), ("compiled",),
     "Use a plain function"),
    (_callable_instance_attribute, lambda: _callable_instance_attribute[(1,)](X),
     ("compiled",), "Use a plain function"),
]


@pytest.mark.parametrize(
    "kernel, launch, mode, phrase",
    [pytest.param(k, launch, m, phrase, id=f"{k.__name__[1:]}-{m}")
     for k, launch, modes, phrase in KERNEL_CASES for m in modes],
)  # fmt: skip
def test_error_names_the_line_and_the_fix(kernel, launch, mode, phrase, monkeypatch):
    if not mode_available(mode):
        pytest.skip(f"{mode} execution isn't available")
    monkeypatch.setenv("ENCELADUS_VERIFY", "0")
    with execution_mode(mode), pytest.raises(enceladus.CompilationError) as e:
        launch()
    msg = str(e.value)
    assert phrase in msg
    lines, start = inspect.getsourcelines(kernel.fn)
    offset = next(i for i, text in enumerate(lines) if text.rstrip().endswith("# error"))
    assert f"{__file__}:{start + offset}:" in msg
    assert lines[offset].rstrip() in msg.splitlines()


def _cpu_torch():
    torch = pytest.importorskip("torch")
    return torch.zeros(64)


# Mistakes in launch arguments name the kernel and the argument instead of a source line.
ARG_CASES = [
    ("cpu_torch", lambda: (_cpu_torch(),), {"BLOCK": 16}, TypeError,
     ["argument `x_ptr`", '.to("mps")']),
    ("float64", lambda: (np.zeros(64),), {"BLOCK": 16}, TypeError,
     ["argument `x_ptr`", "astype(np.float32)"]),
    ("missing_constexpr_value", lambda: (X,), {}, TypeError,
     ["missing a required argument: 'BLOCK'"]),
    ("unhashable_constexpr", lambda: (X,), {"BLOCK": {16: 16}}, TypeError,
     ["parameter `BLOCK`", "a tuple"]),
]


@enceladus.jit
def _fill(x_ptr, BLOCK: tl.constexpr):
    tl.store(x_ptr + tl.arange(0, 16), 1.0)


@pytest.mark.parametrize("name, args, kwargs, exc, phrases", ARG_CASES,
                         ids=[c[0] for c in ARG_CASES])  # fmt: skip
def test_bad_arguments_name_the_kernel_and_argument(mode, name, args, kwargs, exc, phrases):
    if name == "unhashable_constexpr" and mode == "interpret":
        pytest.skip("the interpreter doesn't hash constexpr values")
    with execution_mode(mode), pytest.raises(exc) as e:
        _fill[(1,)](*args(), **kwargs)
    msg = str(e.value)
    assert msg.startswith("_fill: ")
    for p in phrases:
        assert p in msg


@enceladus.jit
def _scaled(out_ptr, S: tl.constexpr):
    offs = tl.arange(0, 16)
    tl.store(out_ptr + offs, (offs + 2147483647) * S)


@enceladus.jit
def _first(out_ptr, S: tl.constexpr):
    tl.store(out_ptr + tl.arange(0, 16), tl.full((16,), S[0], tl.float32))


def test_constexprs_that_compare_equal_compile_separately():
    # `2 == 2.0` and `1 == True` in Python, but they give different kernels: int32 math wraps
    # around, float math doesn't. A launch must not reuse another type's kernel.
    if not mode_available("compiled"):
        pytest.skip("compiled execution isn't available")

    def run(kernel, mode, s):
        out = np.zeros(16, np.float32)
        with execution_mode(mode):
            kernel[(1,)](out, S=s)
        return out

    for s in (2, 2.0, 1, True):
        np.testing.assert_array_equal(run(_scaled, "compiled", s), run(_scaled, "interpret", s))
    # A list constexpr used to crash the compiled launch path with "unhashable type".
    for s in ((3,), [5]):
        np.testing.assert_array_equal(run(_first, "compiled", s), np.full(16, s[0], np.float32))


@enceladus.jit
def _increment(x_ptr, BLOCK: tl.constexpr):
    x = tl.load(x_ptr + tl.arange(0, BLOCK))  # crash
    tl.store(x_ptr + tl.arange(0, BLOCK), x + 1)


def test_compiler_crash_is_reported_at_the_kernel_line(monkeypatch):
    # A bug in codegen must still point at the kernel line, not only at compiler internals.
    if not mode_available("compiled"):
        pytest.skip("compiled execution isn't available")

    def broken_load(self, op, lay):
        raise KeyError("simulated compiler bug")

    monkeypatch.setattr(msl._Codegen, "load", broken_load)
    monkeypatch.setenv("ENCELADUS_ALWAYS_COMPILE", "1")
    with pytest.raises(enceladus.CompilationError) as e:
        _increment.warmup(X, BLOCK=16)
    msg = str(e.value)
    assert "bug in Enceladus" in msg and "simulated compiler bug" in msg
    assert "# crash" in msg.splitlines()[1]


_ASSERT_MODULE = '''
import enceladus
import enceladus.language as tl


@enceladus.jit
def moved_assert_kernel(x_ptr, n):
    x = tl.load(x_ptr)
    tl.device_assert(x < n, "too big")
'''


def _import_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_device_assert_location_survives_the_disk_cache(tmp_path, monkeypatch):
    # The same kernel source at another line, or in another file, must not reuse a cached
    # kernel that reports the old line.
    if not mode_available("compiled"):
        pytest.skip("compiled execution isn't available")
    monkeypatch.setenv("ENCELADUS_DEBUG", "1")
    monkeypatch.setenv("ENCELADUS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("ENCELADUS_ALWAYS_COMPILE", raising=False)
    x = np.full(1, 5.0, np.float32)
    for i, pad in enumerate(["", "\n\n\n"]):
        path = tmp_path / f"moved_{i}" / "kernels.py"
        path.parent.mkdir()
        path.write_text(pad + _ASSERT_MODULE)
        mod = _import_module(path, f"enceladus_test_moved_{i}")
        with execution_mode("compiled"), pytest.raises(enceladus.DeviceAssertionError) as e:
            mod.moved_assert_kernel[(1,)](x, 1)
            enceladus.synchronize()
        want = 1 + (pad + _ASSERT_MODULE).splitlines().index(
            '    tl.device_assert(x < n, "too big")')
        assert (e.value.loc.file, e.value.loc.line) == (str(path), want)
