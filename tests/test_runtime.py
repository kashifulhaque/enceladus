"""Tests for the native runtime, buffers, streams, and raw MSL kernels."""

import ml_dtypes
import numpy as np
import pytest

import tegula
from tegula.runtime.device import PAGE_SIZE

VADD = """
#include <metal_stdlib>
using namespace metal;
kernel void vadd(device const float* x [[buffer(0)]], device const float* y [[buffer(1)]],
                 device float* o [[buffer(2)]], constant uint& n [[buffer(3)]],
                 uint i [[thread_position_in_grid]]) {
    if (i < n) o[i] = x[i] + y[i];
}
"""


@pytest.fixture(scope="module")
def vadd():
    return tegula.metal_kernel(VADD, "vadd")


def page_aligned(n: int, dtype=np.float32) -> np.ndarray:
    nbytes = n * np.dtype(dtype).itemsize
    raw = np.empty(nbytes + PAGE_SIZE, np.uint8)
    start = -raw.ctypes.data % PAGE_SIZE
    return raw[start : start + nbytes].view(dtype)


def make_args(kind: str, n: int):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)
    if kind == "tensor":
        return tegula.from_numpy(x), tegula.from_numpy(y), tegula.zeros(n)
    if kind == "tensor_view":
        # Views with nonzero element offsets inside a larger buffer.
        big = [tegula.zeros(n + 7) for _ in range(3)]
        views = [b[7:] for b in big]
        views[0].copy_(x)
        views[1].copy_(y)
        return tuple(views)
    if kind == "numpy_aligned":
        xa, ya, oa = page_aligned(n), page_aligned(n), page_aligned(n)
        xa[:], ya[:], oa[:] = x, y, 0
        return xa, ya, oa
    if kind == "numpy_unaligned":
        xs, ys, os_ = (np.zeros(n + 3, np.float32)[3:] for _ in range(3))
        xs[:], ys[:] = x, y
        return xs, ys, os_
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["tensor", "tensor_view", "numpy_aligned", "numpy_unaligned"])
@pytest.mark.parametrize("n", [1, 1000, (1 << 20) + 3])
def test_raw_vector_add(vadd, kind, n):
    x, y, out = make_args(kind, n)
    vadd[(-(-n // 256),), (256,)](x, y, out, n)
    got = out.numpy() if isinstance(out, tegula.Tensor) else out
    np.testing.assert_array_equal(got, np.asarray(x) + np.asarray(y))


def test_zero_copy_aliasing_both_directions():
    fill = tegula.metal_kernel(
        """
        #include <metal_stdlib>
        using namespace metal;
        kernel void scale(device float* a [[buffer(0)]], constant float& s [[buffer(1)]],
                          constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
            if (i < n) a[i] = a[i] * s;
        }
        """,
        "scale",
    )
    host = page_aligned(4096)
    host[:] = np.arange(4096, dtype=np.float32)
    t = tegula.from_numpy(host)
    assert t.data_ptr == host.ctypes.data  # shares memory, no copy
    fill[(16,), (256,)](t, 2.0, 4096)
    tegula.synchronize()  # tensor-only launches are asynchronous
    np.testing.assert_array_equal(host, np.arange(4096) * 2)  # GPU write -> host
    host[:] = 1.0
    fill[(16,), (256,)](t, 3.0, 4096)
    np.testing.assert_array_equal(t.numpy(), 3.0)  # host write -> GPU
    # A NumPy argument that doesn't start on a page is also wrapped in place.
    fill[(16,), (256,)](host[5:4005], 0.5, 4000)
    np.testing.assert_array_equal(host[:5], 3.0)
    np.testing.assert_array_equal(host[5:4005], 1.5)


def test_scalar_arguments_bind_by_declared_type():
    k = tegula.metal_kernel(
        """
        #include <metal_stdlib>
        using namespace metal;
        kernel void scalars(device float* o [[buffer(0)]], constant half& h [[buffer(1)]],
                            constant int& i [[buffer(2)]], constant bfloat& b [[buffer(3)]],
                            constant ulong& l [[buffer(4)]], constant uchar& c [[buffer(5)]]) {
            o[0] = float(h); o[1] = float(i); o[2] = float(b);
            o[3] = float(l >> 40); o[4] = float(c);
        }
        """,
        "scalars",
    )
    out = tegula.zeros(5)
    k[(1,), (32,)](out, 1.5, -7, 3.25, 5 << 40, 200)
    expected = [1.5, -7, float(ml_dtypes.bfloat16(3.25)), 5, 200]
    np.testing.assert_array_equal(out.numpy(), expected)


def test_errors_raise_python_exceptions(vadd):
    with pytest.raises(tegula.MetalError, match=r"program_source:3:.*undeclared"):
        tegula.metal_kernel(
            "#include <metal_stdlib>\nkernel void bad(device float* o [[buffer(0)]]) {\n"
            "  o[0] = nope;\n}\n",
            "bad",
        )
    with pytest.raises(tegula.MetalError, match="no kernel function named 'missing'"):
        tegula.metal_kernel(VADD, "missing")
    x = tegula.zeros(8)
    with pytest.raises(TypeError, match=r"takes 4 arguments \(x, y, o, n\), but 3"):
        vadd[(1,), (32,)](x, x, x)
    with pytest.raises(ValueError, match="exceeds the pipeline limit"):
        vadd[(1,), (2048,)]
    with pytest.raises(TypeError, match="float64"):
        vadd[(1,), (32,)](np.zeros(8), x, x, 8)
    tegula.synchronize()  # the stream is still usable


def test_stream_batches_and_flushes_on_threshold(vadd):
    stream = tegula.get_device().stream
    old = stream.flush_every
    stream.flush_every = 4
    try:
        n = 256
        x, y = tegula.ones(n), tegula.ones(n)
        outs = [tegula.zeros(n) for _ in range(6)]
        tegula.synchronize()
        for i in range(3):
            vadd[(1,), (256,)](x, y, outs[i], n)
        assert stream.pending == 3  # tensor-only launches don't wait
        vadd[(1,), (256,)](x, y, outs[3], n)
        assert stream.pending == 0  # the 4th dispatch committed the batch
        vadd[(1,), (256,)](outs[3], y, outs[4], n)  # depends on a committed batch
        vadd[(1,), (256,)](outs[4], y, outs[5], n)  # depends on the open batch
        np.testing.assert_array_equal(outs[5].numpy(), 4.0)
        assert stream.pending == 0
        # A launch with a NumPy argument synchronizes before it returns.
        host = np.zeros(n, np.float32)
        vadd[(1,), (256,)](x, y, host, n)
        assert stream.pending == 0 and host[0] == 2.0
    finally:
        stream.flush_every = old
