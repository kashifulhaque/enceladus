"""Matmul, `c = a @ b`, accumulating in float32, in two variants.

- `matmul_desc` uses tensor descriptors. `tl.dot` loads its fragments straight from
  device memory, and out-of-bounds tiles are handled for you. This is the fast path.
- `matmul` uses pointer tiles with masks. The operands go through threadgroup memory,
  which works for any tile but runs slower.

Run the demo with `uv run python examples/04_matmul.py`.
"""

import numpy as np

import enceladus
import enceladus.language as tl
from enceladus.configs import matmul_configs


@enceladus.jit
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
    grid = lambda meta: (enceladus.cdiv(n, meta["BN"]), enceladus.cdiv(m, meta["BM"]))  # noqa: E731
    matmul_kernel[grid](a, b, c, m, n, k, *s, BM=bm, BN=bn, BK=bk)
    return c


@enceladus.jit
def matmul_desc_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_bk, stride_cm,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):  # fmt: skip
    pid_n, pid_m = tl.program_id(0), tl.program_id(1)
    a = tl.make_tensor_descriptor(a_ptr, [M, K], [stride_am, 1], [BM, BK])
    b = tl.make_tensor_descriptor(b_ptr, [K, N], [stride_bk, 1], [BK, BN])
    c = tl.make_tensor_descriptor(c_ptr, [M, N], [stride_cm, 1], [BM, BN])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc = tl.dot(a.load([pid_m * BM, k]), b.load([k, pid_n * BN]), acc)
    c.store([pid_m * BM, pid_n * BN], acc.to(c.dtype))


def matmul_desc(a, b, c=None, bm: int = 64, bn: int = 64, bk: int = 32, num_warps: int = 4):
    """Returns `a @ b` in the input dtype. `a`, `b`, and `c` must be row-major."""
    (m, k), (_, n) = a.shape, b.shape
    if c is None:
        c = np.empty((m, n), a.dtype) if isinstance(a, np.ndarray) else enceladus.empty((m, n),
                                                                                   a.dtype)
    grid = (enceladus.cdiv(n, bn), enceladus.cdiv(m, bm))
    matmul_desc_kernel[grid](a, b, c, m, n, k, k, n, n, BM=bm, BN=bn, BK=bk,
                             num_warps=num_warps)  # fmt: skip
    return c


def _configs_for_dtype(configs, named):
    """Keeps the configs that `matmul_configs` lists for the dtype of `a`."""
    allowed = matmul_configs(named["a_ptr"].dtype)
    return [c for c in configs if c in allowed]


# The float16 list is a superset of the float32 one. Pruning by dtype keeps the float32
# spill-cliff shapes out of float32 tuning. The key includes the argument dtypes, so each
# dtype is tuned separately.
matmul_desc_tuned = enceladus.autotune(
    configs=matmul_configs("float16"),
    key=["M", "N", "K"],
    prune_configs_by={"early_config_prune": _configs_for_dtype},
)(matmul_desc_kernel)


def matmul_tuned(a, b, c=None):
    """Returns `a @ b` using the fastest configuration for this shape and dtype."""
    (m, k), (_, n) = a.shape, b.shape
    if c is None:
        c = np.empty((m, n), a.dtype) if isinstance(a, np.ndarray) else enceladus.empty((m, n),
                                                                                   a.dtype)
    grid = lambda meta: (enceladus.cdiv(n, meta["BN"]), enceladus.cdiv(m, meta["BM"]))  # noqa: E731
    matmul_desc_tuned[grid](a, b, c, m, n, k, k, n, n)
    return c


def reference(a: np.ndarray, b: np.ndarray, **_) -> np.ndarray:
    return (a.astype(np.float32) @ b.astype(np.float32)).astype(a.dtype)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = rng.standard_normal((100, 70)).astype(np.float32)
    b = rng.standard_normal((70, 90)).astype(np.float32)
    print("max abs error, pointer tiles:", np.abs(matmul(a, b) - reference(a, b)).max())
    print("max abs error, descriptors:", np.abs(matmul_desc(a, b) - reference(a, b)).max())
