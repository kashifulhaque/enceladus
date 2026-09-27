"""Tests for framework interop: PyTorch MPS tensors, MLX arrays, and DLPack export."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from conftest import check_kernel, load_example

import enceladus
import enceladus.language as tl
from enceladus.runtime import launcher

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():
    pytest.skip("PyTorch has no MPS device", allow_module_level=True)
try:
    import mlx.core as mx
except ImportError:
    mx = None

NO_MLX = pytest.mark.skipif(mx is None, reason="MLX isn't installed")
FRAMEWORKS = ["torch", pytest.param("mlx", marks=NO_MLX)]


def to_framework(a: np.ndarray, fw: str, pad: int):
    """Copies `a` into `fw` memory as a view that starts `pad` rows into a larger array.

    A nonzero `pad` gives the view a nonzero storage or byte offset.
    """
    big = np.zeros((a.shape[0] + pad, *a.shape[1:]), a.dtype)
    big[pad:] = a
    if fw == "torch":
        return torch.from_numpy(big).to("mps")[pad:]
    out = mx.array(big)[pad:]
    mx.eval(out)
    return out


def to_numpy(a) -> np.ndarray:
    if isinstance(a, (np.ndarray, int, float)):
        return a
    if isinstance(a, torch.Tensor):
        return a.cpu().numpy()
    return np.array(a)


def _vector_add(fw, pad, rng, dtype):
    ex = load_example("01_vector_add")
    x, y = (rng.standard_normal(10_007).astype(dtype) for _ in range(2))
    return ex.add, ex.reference, (x, y)


def _softmax(fw, pad, rng, dtype):
    ex = load_example("02_softmax")
    return ex.softmax, ex.reference, (rng.standard_normal((37, 300)).astype(dtype),)


def _matmul(fw, pad, rng, dtype):
    ex = load_example("04_matmul")
    a = rng.standard_normal((70, 50)).astype(dtype)
    b = rng.standard_normal((50, 90)).astype(dtype)
    return ex.matmul, ex.reference, (a, b)


def _matmul_desc(fw, pad, rng, dtype):
    ex = load_example("04_matmul")
    a = rng.standard_normal((100, 64)).astype(dtype)
    b = rng.standard_normal((64, 96)).astype(dtype)
    return ex.matmul_desc, ex.reference, (a, b)


def _layernorm(fw, pad, rng, dtype):
    ex = load_example("03_layernorm")
    x = rng.standard_normal((37, 300)).astype(dtype)
    w, b = (rng.standard_normal(300).astype(dtype) for _ in range(2))
    return ex.layernorm, ex.reference, (x, w, b)


def _fused_gelu(fw, pad, rng, dtype):
    ex = load_example("05_fused_gelu")
    x, bias = rng.standard_normal((37, 300)).astype(dtype), rng.standard_normal(300).astype(dtype)
    return ex.fused_gelu, ex.reference, (x, bias, 0.75)


def _rmsnorm(fw, pad, rng, dtype):
    ex = load_example("06_rmsnorm")
    x, w = rng.standard_normal((37, 300)).astype(dtype), rng.standard_normal(300).astype(dtype)
    return ex.rmsnorm, ex.reference, (x, w)


def _histogram(fw, pad, rng, dtype):
    ex = load_example("09_histogram")
    return ex.histogram, ex.reference, (rng.standard_normal(10_007).astype(dtype), 48, -3.0, 3.0)


def _cumsum(fw, pad, rng, dtype):
    ex = load_example("10_cumsum")
    return ex.cumsum, ex.reference, (rng.standard_normal((37, 300)).astype(dtype),)


@pytest.mark.parametrize("fw", FRAMEWORKS)
@pytest.mark.parametrize("pad", [0, 3])  # 3 rows gives offsets that aren't 16-byte aligned
@pytest.mark.parametrize("case", [_vector_add, _softmax, _layernorm, _matmul, _matmul_desc,
                                  _fused_gelu, _rmsnorm, _histogram, _cumsum])  # fmt: skip
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_examples_on_framework_arrays(fw, pad, case, dtype, monkeypatch):
    run, reference, host = case(fw, pad, np.random.default_rng(0), dtype)
    args = [to_framework(a, fw, pad) if isinstance(a, np.ndarray) else a for a in host]

    def run_np(*framework_args):
        out = run(*framework_args)
        outs = out if isinstance(out, tuple) else (out,)
        # Each output is the caller's kind of array.
        assert all(type(o) is type(framework_args[0]) for o in outs)
        return tuple(to_numpy(o) for o in outs) if isinstance(out, tuple) else to_numpy(out)

    synced = []
    real = launcher.launch_synced
    monkeypatch.setattr(launcher, "launch_synced", lambda *a: synced.append(1) or real(*a))
    check_kernel(run_np, args, lambda *a: reference(*[to_numpy(x) for x in a]),
                 atol=5e-2 if dtype == np.float16 else 1e-4, rtol=1e-2)  # fmt: skip
    # PyTorch launches run on PyTorch's stream; MLX launches are synchronous by design.
    assert bool(synced) == (fw == "mlx")


@enceladus.jit
def _scalars_kernel(out_ptr, fout_ptr, a, b, c, flag):
    tl.store(out_ptr, a.to(tl.int64))
    tl.store(out_ptr + 1, b)
    tl.store(out_ptr + 2, flag.to(tl.int64))
    tl.store(fout_ptr, c)


@pytest.mark.parametrize("a, b, c, flag", [(-(1 << 31), (1 << 62) + 5, 1.5, True),
                                           ((1 << 31) - 1, -(1 << 63), -2.25, False)])  # fmt: skip
def test_jit_scalar_types_bind_on_torch_path(a, b, c, flag):
    out = torch.zeros(3, dtype=torch.int64, device="mps")
    fout = torch.zeros(1, device="mps")
    _scalars_kernel[(1,)](out, fout, a, b, c, flag)
    assert out.cpu().tolist() == [a, b, int(flag)]
    assert fout.item() == c


RAW_SCALARS = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device ulong* o [[buffer(0)]], constant char& a [[buffer(1)]],
              constant short& b [[buffer(2)]], constant int& c [[buffer(3)]],
              constant long& d [[buffer(4)]], constant uchar& e [[buffer(5)]],
              constant ushort& f [[buffer(6)]], constant uint& g [[buffer(7)]],
              constant ulong& h [[buffer(8)]], constant half& i [[buffer(9)]],
              constant float& j [[buffer(10)]], constant bool& l [[buffer(11)]],
              constant bfloat& m [[buffer(12)]], uint t [[thread_position_in_grid]]) {
    if (t != 0) return;
    o[0] = as_type<ulong>(long(a)); o[1] = as_type<ulong>(long(b)); o[2] = as_type<ulong>(long(c));
    o[3] = as_type<ulong>(d); o[4] = e; o[5] = f; o[6] = g; o[7] = h;
    o[8] = as_type<ushort>(i); o[9] = as_type<uint>(j); o[10] = l ? 1 : 0;
    o[11] = as_type<ushort>(m);
}
"""


def test_every_abi_scalar_type_binds_on_torch_path():
    k = enceladus.metal_kernel(RAW_SCALARS, "k")
    # Extreme values of each type catch a wrong width or a missing signed wrap.
    vals = (-128, -32768, -(1 << 31), -(1 << 63), 255, 65535, (1 << 32) - 1, (1 << 64) - 1,
            -1.5, 3.25, True, 0.5)  # fmt: skip
    got = torch.zeros(12, dtype=torch.uint64, device="mps")
    k[(1,), (32,)](got, *vals)
    native = np.zeros(12, np.uint64)
    k[(1,), (32,)](native, *vals)
    mask = (1 << 64) - 1
    expected = [v & mask for v in vals[:8]] + [
        int(np.float16(-1.5).view(np.uint16)), int(np.float32(3.25).view(np.uint32)), 1,
        0x3F00,  # bfloat16 0.5
    ]  # fmt: skip
    assert [int(v) & mask for v in got.cpu().tolist()] == expected
    assert native.tolist() == expected


def test_torch_ordering_has_no_stale_data():
    """Torch writes, Enceladus reads and writes, torch reads: 200 times, with no syncs."""
    ex = load_example("01_vector_add")
    n = 1 << 20
    x = torch.empty(n, device="mps")
    ones = torch.ones(n, device="mps")
    out = torch.empty(n, device="mps")
    sums = torch.empty(200, device="mps")
    for i in range(200):
        x.fill_(float(i))  # a torch op writes x
        ex.add_kernel[(n // 1024,)](x, ones, out, n, BLOCK=1024)  # Enceladus reads x
        sums[i] = out[::4097].sum()  # a torch op reads Enceladus's output
    count = len(range(0, n, 4097))
    expected = (torch.arange(200, dtype=torch.float32) + 1) * count
    assert torch.equal(sums.cpu(), expected)


@pytest.mark.parametrize("fw", FRAMEWORKS)
@pytest.mark.parametrize("versioned", [False, True])
def test_dlpack_export_shares_memory(fw, versioned):
    ex = load_example("01_vector_add")
    base = enceladus.zeros(1030)
    t = base[6:]  # a view with a nonzero byte offset
    # A pending launch: the export must wait for it.
    ex.add_kernel[(1,)](enceladus.arange(0, 1024, dtype="float32"), enceladus.ones(1024), t,
                        1024, BLOCK=1024)  # fmt: skip
    if fw == "torch":
        cap = t.__dlpack__(max_version=(1, 0) if versioned else None)
        imported = torch.utils.dlpack.from_dlpack(cap)
        assert imported.device.type == "mps"
        np.testing.assert_array_equal(imported.cpu().numpy(), np.arange(1024) + 1)
        imported.mul_(2)  # a write through the import is visible to Enceladus
        torch.mps.synchronize()
    else:
        imported = mx.from_dlpack(t)
        np.testing.assert_array_equal(np.array(imported), np.arange(1024) + 1)
        base.numpy()[6] = -5.0  # and an Enceladus-side write is visible to MLX
    expected = (np.arange(1024) + 1.0) * (2 if fw == "torch" else 1)
    if fw == "mlx":
        expected[0] = -5.0
        np.testing.assert_array_equal(np.array(imported), expected)
    np.testing.assert_array_equal(t.numpy(), expected)


def test_mixed_arguments_fall_back_and_log_once(caplog):
    ex = load_example("01_vector_add")
    n = 5000
    x = torch.randn(n, device="mps")
    y = enceladus.randn(n, seed=1)
    with caplog.at_level(logging.WARNING, logger="enceladus"):
        for _ in range(2):
            out = torch.zeros(n, device="mps")
            ex.add_kernel[(enceladus.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
            # The result is visible to torch without an explicit sync.
            np.testing.assert_allclose(out.cpu().numpy(), x.cpu().numpy() + y.numpy())
    fallbacks = [r for r in caplog.records if "mixes PyTorch tensors" in r.getMessage()]
    assert len(fallbacks) == 1


@NO_MLX
def test_mlx_refusals():
    ex = load_example("01_vector_add")
    f64 = mx.zeros((4,), dtype=mx.float64, stream=mx.cpu)
    with pytest.raises(TypeError, match="float64"):
        ex.add_kernel[(1,)](f64, f64, f64, 4, BLOCK=1024)
    out = mx.broadcast_to(mx.array([0.0]), (4,))
    with pytest.raises(ValueError, match="broadcast MLX array"):
        ex.add_kernel[(1,)](mx.ones((4,)), mx.ones((4,)), out, 4, BLOCK=1024)


def _matmul_fused(fw, pad, rng, dtype):
    ex = load_example("07_matmul_fused")
    a = rng.standard_normal((100, 64)).astype(dtype)
    b = rng.standard_normal((64, 96)).astype(dtype)
    return ex.matmul_bias_gelu, ex.reference, (a, b, rng.standard_normal(96).astype(dtype))


def _attention(fw, pad, rng, dtype):
    ex = load_example("08_flash_attention")
    q, k, v = (rng.standard_normal((1, 2, 64, 32)).astype(dtype) for _ in range(3))
    return ex.attention, ex.reference, (q, k, v)


@pytest.mark.parametrize("fw", FRAMEWORKS)
@pytest.mark.parametrize("case", [_matmul_fused, _attention])
def test_fused_examples_allocate_outputs_in_the_input_framework(fw, case, monkeypatch):
    # The wrappers allocate their outputs themselves; they must not assume NumPy.
    test_examples_on_framework_arrays(fw, 0, case, np.float16, monkeypatch)


def test_torch_launch_refuses_a_grid_that_overflows_thread_positions():
    ex = load_example("01_vector_add")
    x = torch.ones(8, device="mps")
    # 1024 threads per program: 2^22 programs is 2^32 threads, which Metal runs as none.
    with pytest.raises(ValueError, match="exceeds Metal's limit"):
        ex.add_kernel[(1 << 22,)](x, x, x, 8, BLOCK=1024, num_warps=32)
