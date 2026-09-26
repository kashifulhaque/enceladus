"""Matmul with pointer tiles: `c = a @ b`, accumulating in float32.

M4 adds a tensor-descriptor variant. Run the demo with
`TEGULA_INTERPRET=1 uv run python examples/04_matmul.py`.
"""

import numpy as np

import tegula
import tegula.language as tl


@tegula.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):  # fmt: skip
    pid_n, pid_m = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] + k < K), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] + k < K) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


def matmul(a: np.ndarray, b: np.ndarray, bm: int = 32, bn: int = 32, bk: int = 32) -> np.ndarray:
    """Returns `a @ b` in the input dtype."""
    (m, k), (_, n) = a.shape, b.shape
    c = np.empty((m, n), a.dtype)
    s = [x // a.itemsize for x in (*a.strides, *b.strides, *c.strides)]
    grid = lambda meta: (tegula.cdiv(n, meta["BN"]), tegula.cdiv(m, meta["BM"]))  # noqa: E731
    matmul_kernel[grid](a, b, c, m, n, k, *s, BM=bm, BN=bn, BK=bk)
    return c


def reference(a: np.ndarray, b: np.ndarray, **_) -> np.ndarray:
    return (a.astype(np.float32) @ b.astype(np.float32)).astype(a.dtype)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = rng.standard_normal((100, 70)).astype(np.float32)
    b = rng.standard_normal((70, 90)).astype(np.float32)
    print("max abs error:", np.abs(matmul(a, b) - reference(a, b)).max())
