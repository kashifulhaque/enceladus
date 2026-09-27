"""Tests for the native runtime, buffers, streams, and raw MSL kernels."""

import ctypes
import dataclasses
import gc
import os
import re
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest
from conftest import execution_mode

import enceladus
import enceladus.language as tl
from enceladus.runtime import launcher
from enceladus.runtime import stream as stream_module
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
    # Metal computes thread positions in 32 bits: 2^32 threads would silently run none.
    with pytest.raises(ValueError, match="exceeds Metal's limit"):
        vadd[(1 << 27,), (32,)]
    vadd[((1 << 27) - 1,), (32,)]  # the largest grid is accepted
    with pytest.raises(ValueError, match="at least 1"):
        vadd[(1,), (0,)]
    with pytest.raises(ValueError, match="exceeds Metal's limit"):  # the native check
        enceladus.get_device().stream.native.dispatch(
            vadd.pipeline, vadd._plan((True, True, True, False)), [x.buffer] * 3, None,
            bytes(4), (1 << 27,), (32, 1, 1),
        )  # fmt: skip
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


INC = """
#include <metal_stdlib>
kernel void inc(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) { o[i] += 1; }
"""

_cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
_cf.CFGetRetainCount.restype = ctypes.c_long
_cf.CFGetRetainCount.argtypes = [ctypes.c_void_p]


def _retains(t: enceladus.Tensor) -> int:
    return _cf.CFGetRetainCount(t.buffer.handle)


@enceladus.jit
def _add_one(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=m) + 1.0, mask=m)


def test_launches_release_their_buffers():
    t = enceladus.zeros(1024)
    _add_one[(1,)](t, t, 1024, BLOCK=1024)
    enceladus.synchronize()
    base = _retains(t)
    # Every @jit launch reads the buffer's address, which once leaked one retain each.
    for _ in range(50):
        _add_one[(1,)](t, t, 1024, BLOCK=1024)
    repr(t)
    assert _retains(t) - base < 5
    # A committed command buffer retains its buffers. Later flushes release the ones
    # that completed, without waiting for a sync.
    stream = enceladus.get_device().stream
    other = enceladus.zeros(1024)
    _add_one[(1,)](t, t, 1024, BLOCK=1024)
    stream.flush()
    deadline = time.monotonic() + 5
    while _retains(t) > base and time.monotonic() < deadline:
        _add_one[(1,)](other, other, 1024, BLOCK=1024)
        stream.flush()
        time.sleep(0.001)
    assert _retains(t) <= base
    enceladus.synchronize()


_THREADS_SCRIPT = f"""
import threading
import numpy as np
import enceladus
k = enceladus.metal_kernel({INC!r}, "inc")
stop = threading.Event()
def syncer():
    while not stop.is_set():
        enceladus.synchronize()
def launch(t):
    for _ in range(20000):
        k[(4,), (256,)](t)
ts = [enceladus.zeros(1024) for _ in range(2)]
syncers = [threading.Thread(target=syncer) for _ in range(2)]
launchers = [threading.Thread(target=launch, args=(t,)) for t in ts]
for th in syncers + launchers:
    th.start()
for th in launchers:
    th.join()
stop.set()
for th in syncers:
    th.join()
for t in ts:
    np.testing.assert_array_equal(t.numpy(), 20000)
"""


def test_concurrent_launches_and_syncs():
    # It runs in its own process, so that a crash or a hang fails only this test.
    r = subprocess.run([sys.executable, "-c", _THREADS_SCRIPT], capture_output=True,
                       timeout=30)  # fmt: skip
    assert r.returncode == 0, r.stderr.decode()[-2000:]


class _BigBuffer:
    """Stands in for a buffer of 16 GB, which the launch must refuse before binding."""

    ptr, nbytes = 0, 1 << 34


@pytest.mark.parametrize("case", ["grid", "pipeline_threads", "read_only"])
def test_jit_refuses_launches_it_would_run_wrong(case):
    x, out = enceladus.ones(8), enceladus.zeros(8)
    if case == "grid":
        # 128 threads per program: 2^25 programs is 2^32 threads.
        with pytest.raises(ValueError, match="exceeds Metal's limit"):
            _add_one[(1 << 25,)](x, out, 8, BLOCK=8)
    elif case == "pipeline_threads":
        ck = _add_one.warmup(x, out, 8, BLOCK=8)
        small = SimpleNamespace(max_total_threads_per_threadgroup=64, name=ck.name)
        with pytest.raises(ValueError, match="allows at most 64"):
            dataclasses.replace(ck, pipeline=small).launch((1, 1, 1), (x, out, 8))
    else:
        # A kernel may read a read-only array, but it must not write one.
        ro = np.frombuffer(bytes(32), np.float32)
        host = np.zeros(8, np.float32)
        _add_one[(1,)](ro, host, 8, BLOCK=8)
        np.testing.assert_array_equal(host, 1.0)
        with pytest.raises(ValueError, match="read-only NumPy array"):
            _add_one[(1,)](x, ro, 8, BLOCK=8)
        with pytest.raises(ValueError, match="read-only NumPy array"):
            _add_one[(1,)](x, np.broadcast_to(np.zeros(1, np.float32), (8,)), 8, BLOCK=8)
        assert not np.frombuffer(ro.tobytes(), np.float32).any()


@enceladus.jit
def _masked_tail(x_ptr, y_ptr, n, STEP: tl.constexpr, BLOCK: tl.constexpr):
    # Program p covers elements p * STEP onward; program 2 passes 2^31. Each program
    # reads and writes a small window, so small arrays serve every program.
    pid = tl.program_id(0)
    offs = pid * STEP + tl.arange(0, BLOCK)
    local = offs - pid * STEP
    tl.store(y_ptr + pid * BLOCK + local, tl.load(x_ptr + local) + 1.0, mask=offs < n)


# The register array of the `local` offsets, by the C type of its elements.
_LOCAL_TYPE = re.compile(r"\b(\w+) local_\d+\[\d+\];")


def test_arrays_past_2_31_elements_compile_a_64_bit_variant():
    x = np.arange(16, dtype=np.float32)
    n = (1 << 31) + 3  # program 2 keeps 3 elements, if its mask computes in 64 bits
    huge = enceladus.Tensor(_BigBuffer(), ((1 << 31) + 1,), "float32")
    one = np.zeros(1, np.float32)
    view = np.lib.stride_tricks.as_strided(one, shape=(2, 8), strides=(8 << 30, 4))
    small = _masked_tail.warmup(x, x, n, STEP=1 << 30, BLOCK=16)
    assert not small.idx64 and _LOCAL_TYPE.search(small.msl)[1] == "int"
    for big_arg in (huge, view):
        big = _masked_tail.warmup(big_arg, x, n, STEP=1 << 30, BLOCK=16)
        assert big.idx64 and _LOCAL_TYPE.search(big.msl)[1] == "long"
    # The 64-bit variant computes the exact offsets, so the mask of program 2 keeps only
    # offsets 2^31 to 2^31 + 2. Launched on small arrays, it shows that directly.
    y = np.zeros(48, np.float32)
    big.launch((3, 1, 1), (x, y, n))
    enceladus.synchronize()
    expected = np.concatenate([x + 1, x + 1, np.where(np.arange(16) < 3, x + 1, 0)])
    np.testing.assert_array_equal(y, expected)
    # A kernel with 32-bit offsets refuses the array.
    with pytest.raises(ValueError, match="spans 2147483649 elements"):
        small.launch((1, 1, 1), (huge, y, n))
    edge = np.lib.stride_tricks.as_strided(one, shape=(1 << 31,), strides=(4,))
    assert not launcher.needs_idx64(edge)  # exactly 2^31 elements fit in 32-bit offsets


@enceladus.jit
def _bump(x_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(x_ptr + offs, tl.load(x_ptr + offs, mask=m) + 1, mask=m)


@pytest.mark.slow
def test_kernel_addresses_every_byte_of_a_2_gb_array():
    n = (1 << 31) + 77
    x = enceladus.zeros(n, dtype="uint8")
    try:
        x.numpy()[-5:] = np.arange(5, dtype=np.uint8)
        _bump[(enceladus.cdiv(n, 4096),)](x, n, BLOCK=4096)
        got = x.numpy()
        assert (got[:-5] == 1).all()
        np.testing.assert_array_equal(got[-5:], np.arange(1, 6, dtype=np.uint8))
    finally:
        del x
        gc.collect()


def test_raw_kernel_refuses_to_write_read_only_arrays(vadd):
    ro = np.frombuffer(np.arange(8, dtype=np.float32).tobytes(), np.float32)
    out = np.zeros(8, np.float32)
    vadd[(1,), (32,)](ro, ro, out, 8)  # `device const` inputs may be read-only
    np.testing.assert_array_equal(out, np.arange(8) * 2)
    with pytest.raises(ValueError, match=r"argument 2 \('o'\) is a read-only"):
        vadd[(1,), (32,)](out, out, ro, 8)


@enceladus.jit
def _scale(out_ptr, a):
    tl.store(out_ptr + tl.arange(0, 4), tl.full((4,), 1.0, tl.float32) * a)


@pytest.mark.filterwarnings("ignore:overflow encountered")
@pytest.mark.parametrize("mode", ["interpret", "compiled"])
@pytest.mark.parametrize("value", [1e40, -1e40, np.float64(1e39), 2.5])
def test_float_scalars_convert_like_c(mode, value):
    out = np.zeros(4, np.float32)
    with execution_mode(mode):
        _scale[(1,)](out, value)
    expected = np.float32(value) if abs(value) < 3.4e38 else np.copysign(np.inf, value)
    np.testing.assert_array_equal(out, expected)


def test_empty_views_stay_inside_their_buffer():
    for t, key in [(enceladus.zeros(8), slice(8, None)), (enceladus.zeros(0), slice(None)),
                   (enceladus.zeros((4, 0)), slice(1, None))]:  # fmt: skip
        v = t[key]
        assert v.numel == 0 and v.buffer is t.buffer
        assert 0 <= v.byte_offset <= t.buffer.nbytes


@enceladus.jit
def _checked_copy(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m)
    tl.device_assert(x >= 0, "negative input", mask=m)
    tl.store(out_ptr + offs, x, mask=m)


class _NativeProxy:
    """Wraps the native stream to inject a failure the GPU won't produce on demand."""

    def __init__(self, native, fail_sync=False, lost_sentinels=0):
        self._native, self._fail_sync, self._lost = native, fail_sync, lost_sentinels

    def __getattr__(self, name):
        return getattr(self._native, name)

    @property
    def log_sentinels(self):
        return self._native.log_sentinels + self._lost

    def sync(self):
        self._native.sync()
        if self._fail_sync:
            raise enceladus.MetalError("injected command buffer failure")


def test_failed_sync_leaves_no_stale_device_assert(monkeypatch):
    monkeypatch.setenv("ENCELADUS_DEBUG", "1")
    stream = enceladus.get_device().stream
    out = enceladus.zeros(4)
    _checked_copy[(1,)](enceladus.full(4, -1.0), out, 4, BLOCK=4)  # the assert fails
    monkeypatch.setattr(stream, "native", _NativeProxy(stream.native, fail_sync=True))
    with pytest.raises(enceladus.MetalError, match="injected"):
        enceladus.synchronize()
    monkeypatch.setattr(stream, "native", stream.native._native)
    _checked_copy[(1,)](enceladus.ones(4), out, 4, BLOCK=4)
    enceladus.synchronize()  # reports nothing: the failed sync reset the assert
    np.testing.assert_array_equal(out.numpy(), 1.0)


@enceladus.jit
def _print_pid(x_ptr):
    tl.device_print("pid", tl.program_id(0))


def test_a_lost_log_sentinel_delays_only_one_sync(monkeypatch, capfd):
    stream = enceladus.get_device().stream
    x = enceladus.zeros(1)
    _print_pid[(1,)](x)
    enceladus.synchronize()  # moves the stream to the logging queue
    monkeypatch.setattr(stream_module, "LOG_TIMEOUT", 0.2)
    monkeypatch.setattr(stream, "_lost_sentinels", 0)
    # Metal drops a sentinel, for example when the log buffer overflows.
    monkeypatch.setattr(stream, "native", _NativeProxy(stream.native, lost_sentinels=1))
    _print_pid[(1,)](x)
    enceladus.synchronize()
    assert "didn't arrive in time" in capfd.readouterr().err
    _print_pid[(3,)](x)
    enceladus.synchronize()
    err = capfd.readouterr().err
    assert "didn't arrive in time" not in err
    assert sum(line.startswith("pid") for line in err.splitlines()) == 3
