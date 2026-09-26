"""Matmul with a fused epilogue: `gelu(a @ b + bias)`.

The bias add, the GELU, and the cast run on the accumulator registers before the store,
so the fused kernel costs about the same as a plain matmul.

Run the demo with `uv run python examples/07_matmul_fused.py`.
"""

import math

import numpy as np

import enceladus
import enceladus.language as tl


@enceladus.jit
def matmul_bias_gelu_kernel(a_ptr, b_ptr, bias_ptr, c_ptr, M, N, K, stride_am, stride_bk,
                            stride_cm, BM: tl.constexpr, BN: tl.constexpr,
                            BK: tl.constexpr):  # fmt: skip
    pid_n, pid_m = tl.program_id(0), tl.program_id(1)
    a = tl.make_tensor_descriptor(a_ptr, [M, K], [stride_am, 1], [BM, BK])
    b = tl.make_tensor_descriptor(b_ptr, [K, N], [stride_bk, 1], [BK, BN])
    c = tl.make_tensor_descriptor(c_ptr, [M, N], [stride_cm, 1], [BM, BN])
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        acc = tl.dot(a.load([pid_m * BM, k]), b.load([k, pid_n * BN]), acc)
    cols = pid_n * BN + tl.arange(0, BN)
    bias = tl.load(bias_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
    y = acc + bias[None, :]
    y = 0.5 * y * (1.0 + tl.erf(y * 0.7071067811865476))
    c.store([pid_m * BM, pid_n * BN], y.to(c.dtype))


def matmul_bias_gelu(a, b, bias, c=None, bm: int = 64, bn: int = 64, bk: int = 32):
    """Returns `gelu(a @ b + bias)` in the input dtype."""
    (m, k), (_, n) = a.shape, b.shape
    if c is None:
        c = np.empty((m, n), a.dtype) if isinstance(a, np.ndarray) else enceladus.empty((m, n),
                                                                                   a.dtype)
    grid = (enceladus.cdiv(n, bn), enceladus.cdiv(m, bm))
    matmul_bias_gelu_kernel[grid](a, b, bias, c, m, n, k, k, n, n, BM=bm, BN=bn, BK=bk)
    return c


_erf = np.vectorize(math.erf, otypes=[np.float64])


def reference(a, b, bias, **_):
    y = a.astype(np.float64) @ b.astype(np.float64) + bias.astype(np.float64)
    return (0.5 * y * (1 + _erf(y / math.sqrt(2)))).astype(a.dtype)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = rng.standard_normal((200, 96)).astype(np.float32)
    b = rng.standard_normal((96, 130)).astype(np.float32)
    bias = rng.standard_normal(130).astype(np.float32)
    print("max abs error:", np.abs(matmul_bias_gelu(a, b, bias) - reference(a, b, bias)).max())
