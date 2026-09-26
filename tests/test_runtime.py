"""Tests for the native runtime, buffers, streams, and raw MSL kernels."""

import gc
import os
import subprocess
import sys
import textwrap

import ml_dtypes
import numpy as np
import pytest

import enceladus
import enceladus.language as tl
from enceladus.runtime.device import PAGE_SIZE

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
    return enceladus.metal_kernel(VADD, "vadd")


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
        return enceladus.from_numpy(x), enceladus.from_numpy(y), enceladus.zeros(n)
    if kind == "tensor_view":
        # Views with nonzero element offsets inside a larger buffer.
        big = [enceladus.zeros(n + 7) for _ in range(3)]
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
    got = out.numpy() if isinstance(out, enceladus.Tensor) else out
    np.testing.assert_array_equal(got, np.asarray(x) + np.asarray(y))


def test_zero_copy_aliasing_both_directions():
    fill = enceladus.metal_kernel(
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
    t = enceladus.from_numpy(host)
    assert t.data_ptr == host.ctypes.data  # shares memory, no copy
    fill[(16,), (256,)](t, 2.0, 4096)
    enceladus.synchronize()  # tensor-only launches are asynchronous
    np.testing.assert_array_equal(host, np.arange(4096) * 2)  # GPU write -> host
    host[:] = 1.0
    fill[(16,), (256,)](t, 3.0, 4096)
    np.testing.assert_array_equal(t.numpy(), 3.0)  # host write -> GPU
    # A NumPy argument that doesn't start on a page is also wrapped in place.
    fill[(16,), (256,)](host[5:4005], 0.5, 4000)
    np.testing.assert_array_equal(host[:5], 3.0)
    np.testing.assert_array_equal(host[5:4005], 1.5)


def test_scalar_arguments_bind_by_declared_type():
    k = enceladus.metal_kernel(
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
    out = enceladus.zeros(5)
    k[(1,), (32,)](out, 1.5, -7, 3.25, 5 << 40, 200)
    expected = [1.5, -7, float(ml_dtypes.bfloat16(3.25)), 5, 200]
    np.testing.assert_array_equal(out.numpy(), expected)


def test_errors_raise_python_exceptions(vadd):
    with pytest.raises(enceladus.MetalError, match=r"program_source:3:.*undeclared"):
        enceladus.metal_kernel(
            "#include <metal_stdlib>\nkernel void bad(device float* o [[buffer(0)]]) {\n"
            "  o[0] = nope;\n}\n",
            "bad",
        )
    with pytest.raises(enceladus.MetalError, match="no kernel function named 'missing'"):
        enceladus.metal_kernel(VADD, "missing")
    x = enceladus.zeros(8)
    with pytest.raises(TypeError, match=r"takes 4 arguments \(x, y, o, n\), but 3"):
        vadd[(1,), (32,)](x, x, x)
    with pytest.raises(ValueError, match="exceeds the pipeline limit"):
        vadd[(1,), (2048,)]
    with pytest.raises(TypeError, match="float64"):
        vadd[(1,), (32,)](np.zeros(8), x, x, 8)
    enceladus.synchronize()  # the stream is still usable


def test_stream_batches_and_flushes_on_threshold(vadd):
    stream = enceladus.get_device().stream
    old = stream.flush_every
    stream.flush_every = 4
    try:
        n = 256
        x, y = enceladus.ones(n), enceladus.ones(n)
        outs = [enceladus.zeros(n) for _ in range(6)]
        enceladus.synchronize()
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


SLOW_FILL = """
#include <metal_stdlib>
using namespace metal;
kernel void slow_fill(device float* o [[buffer(0)]], constant int& iters [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    float acc = 0;
    for (int k = 0; k < iters; ++k) acc = fma(acc, 0.999f, 1.0f);
    o[i] = acc > 0 ? 7.0f : 0.0f;
}
"""


@enceladus.jit
def _slow_fill(o_ptr, iters, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for _ in range(iters):
        acc = acc * 0.999 + 1.0
    tl.store(o_ptr + offs, acc)  # stores the loop result, so the compiler keeps the loop


@pytest.mark.parametrize("path", ["raw", "jit"])
def test_async_launch_keeps_borrowed_host_memory_alive(path):
    n = 1 << 20
    if path == "raw":
        k = enceladus.metal_kernel(SLOW_FILL, "slow_fill")
        launch = lambda t: k[(n // 256,), (256,)](t, 20000)  # noqa: E731
    else:
        launch = lambda t: _slow_fill[(n // 1024,)](t, 20000, BLOCK=1024)  # noqa: E731
    launch(enceladus.zeros(n))  # compiles first, so the compiler allocates nothing later
    enceladus.synchronize()
    t = enceladus.from_numpy(page_aligned(n))
    launch(t)
    enceladus.get_device().stream.flush()
    # Dropping the tensor must not free the NumPy memory that the GPU still writes: the
    # next allocation of the same size reuses those pages.
    del t
    gc.collect()
    fresh = np.zeros(n * 4 + PAGE_SIZE, np.uint8)
    enceladus.synchronize()
    assert not fresh.any()


@pytest.mark.skipif(not os.path.exists("/usr/lib/libgmalloc.dylib"),
                    reason="needs Guard Malloc to catch the use after free")  # fmt: skip
def test_stream_outlives_a_freed_kernel():
    script = textwrap.dedent(f"""
        import gc, enceladus
        out = enceladus.zeros(1024)
        k = enceladus.metal_kernel({VADD!r}, "vadd")
        k[(4,), (256,)](out, out, out, 1024)
        del k
        gc.collect()
        enceladus.synchronize()
    """)
    env = {**os.environ, "DYLD_INSERT_LIBRARIES": "/usr/lib/libgmalloc.dylib"}
    r = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True,
                       timeout=60)  # fmt: skip
    assert r.returncode == 0, r.stderr.decode()[-2000:]
