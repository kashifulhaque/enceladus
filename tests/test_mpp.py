"""The Metal 4 `matmul2d` backend for `tl.dot` (`dot_backend="mpp"`) and its fallback."""

import logging

import numpy as np
import pytest
from conftest import check_kernel, load_example

import enceladus
import enceladus.language as tl
from enceladus.runtime.dot_backend import mpp_supported

pytestmark = pytest.mark.skipif(not mpp_supported(), reason="the device lacks Metal 4 matmul2d")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
@pytest.mark.parametrize("mnk", [(128, 128, 64), (200, 130, 100)])
@pytest.mark.parametrize("example", ["matmul", "fused"])
def test_examples_on_mpp(mode, rng, example, mnk, dtype):
    m, n, k = mnk
    a = rng.standard_normal((m, k)).astype(dtype)
    b = rng.standard_normal((k, n)).astype(dtype)
    if example == "matmul":
        ex = load_example("04_matmul")
        run, args, kernel = ex.matmul_desc, (a, b), ex.matmul_desc_kernel
        warm = (a, b, a, m, n, k, k, n, n)
    else:
        ex = load_example("07_matmul_fused")
        bias = rng.standard_normal(n).astype(dtype)
        run, args, kernel = ex.matmul_bias_gelu, (a, b, bias), ex.matmul_bias_gelu_kernel
        warm = (a, b, bias, a, m, n, k, k, n, n)
    check_kernel(run, args, ex.reference, kwargs={"dot_backend": "mpp"}, modes=(mode,))
    if mode == "compiled":
        ck = kernel.warmup(*warm, BM=64, BN=64, BK=32, dot_backend="mpp")
        assert ck.dot_backend == "mpp" and ck.language_version == (4, 0)
        assert "matmul2d" in ck.msl and "simdgroup_multiply_accumulate" not in ck.msl


@enceladus.jit
def _variants(a_ptr, b_ptr, r_ptr, c_ptr, n_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr,
              BK: tl.constexpr, CASE: tl.constexpr):  # fmt: skip
    """`a @ b` three ways: transposed A, transposed B, and a loop that must stay a loop."""
    pid_n, pid_m = tl.program_id(0), tl.program_id(1)
    a = tl.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [BM, BK])
    at = tl.make_tensor_descriptor(a_ptr, [K, M], [M, 1], [BK, BM])
    b = tl.make_tensor_descriptor(b_ptr, [K, N], [N, 1], [BK, BN])
    bt = tl.make_tensor_descriptor(b_ptr, [N, K], [K, 1], [BN, BK])
    r = tl.make_tensor_descriptor(r_ptr, [M, N], [N, 1], [BM, BN])
    c = tl.make_tensor_descriptor(c_ptr, [M, N], [N, 1], [BM, BN])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    count = 0
    for k in range(0, K, BK):
        if CASE == "trans_a":
            acc = tl.dot(tl.trans(at.load([k, pid_m * BM])), b.load([k, pid_n * BN]), acc)
        elif CASE == "trans_b":
            acc = tl.dot(a.load([pid_m * BM, k]), tl.trans(bt.load([pid_n * BN, k])), acc)
        else:
            acc = tl.dot(a.load([pid_m * BM, k]), b.load([k, pid_n * BN]), acc)
            count += 1  # a second carried value keeps the K loop
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    r2 = tl.load(r_ptr + rows[:, None] * N + cols[None, :], mask=mask, other=0.0)
    y = acc + r.load([pid_m * BM, pid_n * BN]).to(tl.float32) - 2.0 * r2.to(tl.float32)
    c.store([pid_m * BM, pid_n * BN], y.to(c.dtype))
    if pid_m == 0 and pid_n == 0:
        tl.store(n_ptr, count)


@pytest.mark.parametrize("case", ["trans_a", "trans_b", "manual"])
def test_transposed_operands_and_k_loop(rng, case):
    m, n, k = 150, 90, 72  # ragged in every dimension, and K isn't a multiple of BK
    a = rng.standard_normal((m, k)).astype(np.float16)
    b = rng.standard_normal((k, n)).astype(np.float16)
    r = rng.standard_normal((m, n)).astype(np.float16)
    a_in = np.ascontiguousarray(a.T) if case == "trans_a" else a
    b_in = np.ascontiguousarray(b.T) if case == "trans_b" else b
    meta = {"BM": 64, "BN": 32, "BK": 32, "CASE": case}

    def run(a_in, b_in, r):
        c, count = np.empty((m, n), np.float16), np.zeros(1, np.int32)
        _variants[(3, 3)](a_in, b_in, r, c, count, m, n, k, **meta, dot_backend="mpp")
        return c, count

    def reference(a_in, b_in, r):
        y = a.astype(np.float32) @ b.astype(np.float32) - r.astype(np.float32)
        return y.astype(np.float16), np.array([3 if case == "manual" else 0], np.int32)

    check_kernel(run, (a_in, b_in, r), reference, modes=("compiled",))
    ck = _variants.warmup(a_in, b_in, r, r, r, m, n, k, **meta, dot_backend="mpp")
    assert ck.dot_backend == "mpp"
    assert ("multiply_accumulate" in ck.msl) == (case == "manual")


@enceladus.jit
def _ineligible(a_ptr, b_ptr, c_ptr, s_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr,
                BK: tl.constexpr, CASE: tl.constexpr):  # fmt: skip
    pid_n, pid_m = tl.program_id(0), tl.program_id(1)
    a = tl.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [BM, BK])
    b = tl.make_tensor_descriptor(b_ptr, [K, N], [N, 1], [BK, BN])
    c = tl.make_tensor_descriptor(c_ptr, [M, N], [N, 1], [BM, BN])
    if CASE == "init":
        acc = c.load([pid_m * BM, pid_n * BN]).to(tl.float32)
    else:
        acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc = tl.dot(a.load([pid_m * BM, k]), b.load([k, pid_n * BN]), acc)
    if CASE == "sum":
        tl.store(s_ptr + pid_m * tl.num_programs(0) + pid_n, tl.sum(tl.sum(acc, 1), 0))
    c.store([pid_m * BM, pid_n * BN], acc.to(c.dtype))


@pytest.mark.parametrize(("case", "reason"), [
    ("init", "doesn't start from zeros"), ("sum", "feeds `reduce`"),
])  # fmt: skip
def test_ineligible_kernels_fall_back_to_simdgroup(rng, caplog, monkeypatch, tmp_path, case,
                                                   reason):  # fmt: skip
    monkeypatch.setenv("ENCELADUS_CACHE_DIR", str(tmp_path))  # compile, and log, afresh
    m, n, k = 100, 70, 40
    a = rng.integers(-2, 3, (m, k)).astype(np.float32)
    b = rng.integers(-2, 3, (k, n)).astype(np.float32)
    c0 = rng.integers(-2, 3, (m, n)).astype(np.float32)
    meta = {"BM": 64, "BN": 64, "BK": 32, "CASE": case}

    def run(a, b, c0):
        c, s = c0.copy(), np.zeros(4, np.float32)
        _ineligible[(2, 2)](a, b, c, s, m, n, k, **meta, dot_backend="mpp")
        return c, s

    def reference(a, b, c0):
        acc = a @ b + (c0 if case == "init" else 0)
        s = np.zeros(4, np.float32)
        if case == "sum":
            s = np.array([acc[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64].sum()
                          for i in range(2) for j in range(2)], np.float32)  # fmt: skip
        return acc, s

    with caplog.at_level(logging.DEBUG, logger="enceladus"):
        check_kernel(run, (a, b, c0), reference, modes=("compiled",))
    assert any(reason in r.getMessage() and "simdgroup" in r.getMessage()
               for r in caplog.records)  # fmt: skip
    ck = _ineligible.warmup(a, b, c0, c0, m, n, k, **meta, dot_backend="mpp")
    assert ck.dot_backend == "simdgroup" and ck.language_version == (3, 2)
    assert "matmul2d" not in ck.msl and "simdgroup_multiply_accumulate" in ck.msl


def test_mpp_kernel_reloads_from_the_disk_cache(rng, monkeypatch, tmp_path):
    """A cache hit must restore the language version that `matmul2d` needs."""
    from enceladus.runtime import cache

    monkeypatch.setenv("ENCELADUS_CACHE_DIR", str(tmp_path))
    ex = load_example("04_matmul")
    a = rng.standard_normal((96, 80)).astype(np.float16)
    b = rng.standard_normal((80, 64)).astype(np.float16)
    first = ex.matmul_desc(a, b, bm=32, bn=32, dot_backend="mpp")
    monkeypatch.setattr(cache, "_pipelines", {})
    monkeypatch.setattr(ex.matmul_desc_kernel, "_compiled", {})
    second = ex.matmul_desc(a, b, bm=32, bn=32, dot_backend="mpp")
    ck = next(iter(ex.matmul_desc_kernel._compiled.values()))
    assert ck.cache_dir is None  # loaded from the disk cache, not generated
    assert ck.language_version == (4, 0) and ck.dot_backend == "mpp"
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(second, ex.reference(a, b), atol=1e-2, rtol=2e-3)


def test_mpp_kernel_runs_on_pytorch_stream(rng):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("PyTorch MPS isn't available")
    from enceladus.runtime.launcher import TorchLaunch

    ex = load_example("04_matmul")
    a = rng.standard_normal((100, 70)).astype(np.float16)
    b = rng.standard_normal((70, 90)).astype(np.float16)
    c = ex.matmul_desc(torch.from_numpy(a).to("mps"), torch.from_numpy(b).to("mps"),
                       dot_backend="mpp")  # fmt: skip
    np.testing.assert_allclose(c.cpu().numpy(), ex.reference(a, b), atol=1e-2, rtol=2e-3)
    cks = [ck for ck in ex.matmul_desc_kernel._compiled.values() if ck.dot_backend == "mpp"]
    assert any(isinstance(ck._torch, TorchLaunch) for ck in cks)
