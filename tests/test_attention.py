"""Flash attention and the `tl.dot` paths it relies on: accumulator row reductions and
accumulators that feed a second `tl.dot` from registers."""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest
from conftest import check_kernel, load_example

import enceladus
import enceladus.language as tl

F32, F16, BF16 = np.float32, np.float16, ml_dtypes.bfloat16

# (dtype, seq_len, head_dim, causal, (BLOCK_M, BLOCK_N, num_warps)). Each case reaches a
# different code path: 8 or 16 query rows per SIMD group, keys staged through threadgroup
# memory or read from device memory (one SIMD group), and FP32 at head dimension 128,
# which loads one value fragment at a time.
ATTENTION_CASES = [
    (F16, 64, 64, False, (32, 32, 4)),
    (F16, 64, 64, True, (32, 32, 4)),
    (F16, 100, 64, False, (32, 32, 4)),
    (F16, 100, 64, True, (32, 32, 4)),
    (F16, 100, 128, False, (64, 32, 8)),
    (F16, 100, 128, True, (32, 16, 4)),
    (F16, 100, 64, True, (64, 16, 4)),
    (F16, 37, 32, False, (16, 32, 1)),
    (F32, 100, 128, True, (32, 32, 4)),
    (BF16, 100, 64, False, (32, 16, 4)),
]


@pytest.mark.parametrize("dtype,n,d,causal,cfg", ATTENTION_CASES,
                         ids=lambda v: getattr(v, "__name__", str(v)))  # fmt: skip
def test_flash_attention(mode, dtype, n, d, causal, cfg):
    ex = load_example("08_flash_attention")
    rng = np.random.default_rng(n + d)
    q, k, v = (rng.standard_normal((2, 2, n, d)).astype(np.float32).astype(dtype)
               for _ in range(3))  # fmt: skip
    bm, bn, nw = cfg
    kw = {"causal": causal, "sm_scale": 0.3 if causal else None}

    def run(q, k, v, causal, sm_scale):
        return ex.attention(q, k, v, causal, sm_scale, block_m=bm, block_n=bn, num_warps=nw)

    # FP16 rounds P before the second dot, which costs a few units in the last place.
    atol = {F16: 2e-3, F32: 1e-5, BF16: 2e-2}[dtype]
    check_kernel(run, (q, k, v), ex.reference, kwargs=kw, modes=(mode,), atol=atol)


@enceladus.jit
def _two_dots(a_ptr, b_ptr, c_ptr, o_ptr, M, K, N1, N2, BM: tl.constexpr, BK: tl.constexpr,
              BN1: tl.constexpr, BN2: tl.constexpr):  # fmt: skip
    a_d = tl.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [BM, BK])
    b_d = tl.make_tensor_descriptor(b_ptr, [K, N1], [N1, 1], [BK, BN1])
    c_d = tl.make_tensor_descriptor(c_ptr, [N1, N2], [N2, 1], [BN1, BN2])
    o_d = tl.make_tensor_descriptor(o_ptr, [M, N2], [N2, 1], [BM, BN2])
    a = a_d.load([0, 0])  # loaded once; stays in registers across the loop
    acc = tl.zeros((BM, BN2), tl.float32)
    for n in range(0, N1, BN1):
        p = tl.dot(a, b_d.load([0, n]))
        acc = tl.dot(p.to(tl.float16), c_d.load([n, 0]), acc)
    o_d.store([0, 0], acc)


@pytest.mark.parametrize("dot_warps", [(4, 1), (2, 2)])
def test_accumulator_feeds_a_second_dot(mode, dot_warps):
    """With WN = 1, `p` feeds the second dot from registers; with WN = 2, it's staged."""
    rng = np.random.default_rng(1)
    m, k, n1, n2 = 50, 20, 70, 45  # ragged in every dimension; three loop steps over N1
    a, b, c = (rng.standard_normal(s).astype(F16) for s in ((m, k), (k, n1), (n1, n2)))
    meta = {"BM": 64, "BK": 32, "BN1": 32, "BN2": 64}

    def run(a, b, c):
        o = np.empty((m, n2), F32)
        _two_dots[(1,)](a, b, c, o, m, k, n1, n2, **meta, dot_warps=dot_warps)
        return o

    def reference(a, b, c):
        p = (a.astype(F32) @ b.astype(F32)).astype(F16)
        return p.astype(F32) @ c.astype(F32)

    check_kernel(run, (a, b, c), reference, modes=(mode,), atol=2e-2, rtol=2e-3)
    if mode == "compiled":
        ck = _two_dots.warmup(a, b, c, a, m, k, n1, n2, **meta, dot_warps=dot_warps)
        # A staged left operand is the only thing that allocates a `tga` buffer.
        assert ("tga" in ck.msl) == (dot_warps[1] > 1)


@enceladus.jit
def _dot_row_max(a_ptr, b_ptr, out_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr,
                 BK: tl.constexpr):  # fmt: skip
    a_d = tl.make_tensor_descriptor(a_ptr, [M, K], [K, 1], [BM, BK])
    b_d = tl.make_tensor_descriptor(b_ptr, [K, N], [N, 1], [BK, BN])
    s = tl.dot(a_d.load([0, 0]), b_d.load([0, 0]))
    cols = tl.arange(0, BN)
    s = tl.where(cols[None, :] < N, s, float("-inf"))
    rows = tl.arange(0, BM)
    tl.store(out_ptr + rows, tl.max(s, 1), mask=rows < M)


@pytest.mark.parametrize("dot_warps", [(4, 1), (2, 2)])
def test_accumulator_row_reduction_stays_in_simd_groups(mode, dot_warps):
    rng = np.random.default_rng(2)
    m, n, k = 60, 30, 16
    a, b = rng.standard_normal((m, k)).astype(F16), rng.standard_normal((k, n)).astype(F16)
    meta = {"BM": 64, "BN": 32, "BK": 16}

    def run(a, b):
        out = np.empty(m, F32)
        _dot_row_max[(1,)](a, b, out, m, n, k, **meta, dot_warps=dot_warps)
        return out

    check_kernel(run, (a, b), lambda a, b: (a.astype(F32) @ b.astype(F32)).max(1),
                 modes=(mode,), atol=1e-4)  # fmt: skip
    if mode == "compiled":
        ck = _dot_row_max.warmup(a, b, a, m, n, k, **meta, dot_warps=dot_warps)
        # With WN = 1, each SIMD group owns whole rows: shuffles only. With WN = 2, the
        # row halves meet in threadgroup memory.
        no_tg = ck.threadgroup_memory_bytes == 0 and "threadgroup_barrier" not in ck.msl
        assert no_tg == (dot_warps[1] == 1)


def test_autotuning_as_the_first_dot_of_a_process(tmp_path):
    """Autotuning compiles from several threads. Each compile of a `tl.dot` kernel first
    checks the `simdgroup_matrix` lane layout once per process, and concurrent checks
    that launch on the shared stream without a lock crash the process."""
    import os
    import subprocess
    import sys

    from conftest import EXAMPLES

    path = str(EXAMPLES / "08_flash_attention.py")
    code = (
        "import importlib.util, enceladus\n"
        f"spec = importlib.util.spec_from_file_location('fa', {path!r})\n"
        "fa = importlib.util.module_from_spec(spec); spec.loader.exec_module(fa)\n"
        "q = enceladus.randn(1, 1, 64, 128, dtype='float16')\n"
        "fa.attention_tuned(q, q, q).numpy()\n"
    )
    env = dict(os.environ, ENCELADUS_CACHE_DIR=str(tmp_path), ENCELADUS_INTERPRET="0")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                       timeout=120)  # fmt: skip
    assert r.returncode == 0, r.stderr[-2000:]
