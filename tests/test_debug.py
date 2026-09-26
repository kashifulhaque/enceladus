"""Tests for the debugging tools: device printing, device asserts, explain, and capture."""

from __future__ import annotations

import inspect
import os
from collections import Counter

import numpy as np
import pytest
from conftest import execution_mode

import enceladus
import enceladus.language as tl


def _line_of(fn, needle: str) -> int:
    """Returns the file line number of the first line of `fn` that contains `needle`."""
    lines, first = inspect.getsourcelines(fn.fn)
    return first + next(i for i, text in enumerate(lines) if needle in text)


@enceladus.jit
def print_kernel(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=-1.0)
    tl.device_print("x", x, offs)
    # A row sum is a slice layout: 4 threads hold each result, and only one prints it.
    rows = tl.load(x_ptr + tl.arange(0, 2)[:, None] * 4 + tl.arange(0, 4)[None, :])
    tl.device_print("rowsum", tl.sum(rows, axis=1))
    tl.device_print("n", n, pid)


def _torch_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.backends.mps.is_available()


@pytest.mark.parametrize("mode", ["interpret", "compiled",
                                  pytest.param("torch", marks=pytest.mark.skipif(
                                      not _torch_available(), reason="no PyTorch MPS"))])
def test_device_print_lines_carry_program_ids_without_duplicates(mode, capfd):
    # 8 elements across 128 threads is a broadcast layout: 16 threads hold each element.
    x = np.arange(11, dtype=np.float32) * 0.5
    arg = x
    if mode == "torch":
        import torch

        arg = torch.from_numpy(x).to("mps")
    with execution_mode("interpret" if mode == "interpret" else "compiled"):
        print_kernel[(2,)](arg, 11, BLOCK=8)
        enceladus.synchronize()
    got = Counter(line for line in capfd.readouterr().err.splitlines() if line.startswith("pid"))
    want = Counter()
    for pid in range(2):
        for i in range(8):
            o = pid * 8 + i
            v = x[o] if o < 11 else -1.0
            want[f"pid ({pid}, 0, 0) idx ({i}) x: {v:f} {o}"] += 1
        for r in range(2):
            want[f"pid ({pid}, 0, 0) idx ({r}) rowsum: {x[4 * r:4 * r + 4].sum():f}"] += 1
        want[f"pid ({pid}, 0, 0) n: 11 {pid}"] += 1
    assert got == want


@enceladus.jit
def gather_kernel(src_ptr, idx_ptr, out_ptr, n_src, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    idx = tl.load(idx_ptr + offs, mask=m)
    ok = (idx >= 0) & (idx < n_src)
    tl.device_assert(ok, "gather index out of range", mask=m)
    tl.store(out_ptr + offs, tl.load(src_ptr + idx, mask=m & ok), mask=m)


@pytest.mark.parametrize("debug", [True, False])
@pytest.mark.parametrize("mode", ["interpret", "compiled"])
def test_device_assert_reports_the_failing_line_only_in_debug_mode(mode, debug, monkeypatch):
    if debug:
        monkeypatch.setenv("ENCELADUS_DEBUG", "1")
    else:
        monkeypatch.delenv("ENCELADUS_DEBUG", raising=False)
    src = np.arange(10, dtype=np.float32)
    bad = np.array([1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 0], dtype=np.int32)  # 12 is in program 1
    out = np.zeros(11, np.float32)
    with execution_mode(mode):
        if debug:
            with pytest.raises(enceladus.DeviceAssertionError) as info:
                gather_kernel[(3,)](src, bad, out, 10, 11, BLOCK=4)
                enceladus.synchronize()
            e = info.value
            assert e.loc.line == _line_of(gather_kernel, "tl.device_assert")
            assert e.program_id == (1, 0, 0)
            assert "gather index out of range" in str(e)
        # Valid indices never fail, and a reported failure doesn't linger to a later sync.
        good = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0], dtype=np.int32)
        gather_kernel[(3,)](src, good, out, 10, 11, BLOCK=4)
        enceladus.synchronize()
        np.testing.assert_array_equal(out, src[good])
        if not debug:
            gather_kernel[(3,)](src, bad, out, 10, 11, BLOCK=4)
            enceladus.synchronize()
    if mode == "compiled":
        ck = gather_kernel.warmup(src, good, out, 10, 11, BLOCK=4)
        assert (ck.assert_buffer_index is not None) == debug
        assert ("tg_assert_fail" in ck.msl) == debug


@enceladus.jit
def transposed_sum_kernel(a_ptr, b_ptr, out_ptr, N: tl.constexpr):
    r = tl.arange(0, N)
    a = tl.load(a_ptr + r[:, None] * N + r[None, :])
    bt = tl.load(b_ptr + r[:, None] + r[None, :] * N)  # contiguous down columns
    s = tl.sum(a + bt, axis=0)
    tl.store(out_ptr + r, s)


def test_explain_reports_conversions_and_threadgroup_memory_by_line():
    a = np.zeros((64, 64), np.float32)
    out = np.zeros(64, np.float32)
    text = transposed_sum_kernel.explain(a, a, out, grid=(1,), N=64)
    fname = os.path.basename(__file__)
    where = f"{fname}:{_line_of(transposed_sum_kernel, 's = tl.sum')}"
    conv = [line for line in text.splitlines() if "threadgroup memory exchange" in line]
    assert len(conv) == 1 and conv[0].strip().startswith(where) and "`bt`" in conv[0]
    assert "s = tl.sum(a + bt, axis=0)" in conv[0]
    reduce = [line for line in text.splitlines() if line.strip().startswith(f"{where}  reduce:")]
    assert reduce and int(reduce[0].split("reduce:")[1].split()[0]) > 0
    ck = transposed_sum_kernel.warmup(a, a, out, N=64)
    assert f"Threadgroup memory: {ck.threadgroup_memory_bytes} bytes" in text
    assert "dot backend: none" in text


def test_capture_explains_the_environment_requirement(monkeypatch, tmp_path):
    monkeypatch.delenv("MTL_CAPTURE_ENABLED", raising=False)
    with pytest.raises(RuntimeError, match="MTL_CAPTURE_ENABLED=1"):
        with enceladus.capture(tmp_path / "trace.gputrace"):
            pass
