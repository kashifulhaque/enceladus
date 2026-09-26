"""Flash attention forward, `softmax(q @ k^T * sm_scale) @ v`, with an optional causal mask.

Each program computes `BLOCK_M` query rows of one (batch, head) pair. It loops over the
keys in blocks of `BLOCK_N` and keeps a running row maximum `m_i` and row sum `l_i`, so
the full attention matrix never exists in memory (the online softmax of FlashAttention).
The output accumulates in float32.

The kernel keeps everything in registers:

- `tl.dot(q, tl.trans(k))` loads `k` straight from device memory with transposed
  fragment loads, and `q` stays in registers across the loop.
- The row maximum and row sum of the scores reduce inside each SIMD group, because each
  SIMD group owns whole rows (`dot_warps=(num_warps, 1)`).
- `p.to(...)` feeds the second `tl.dot` directly from the score registers.

Run the demo with `uv run python examples/08_flash_attention.py`.
"""

import math

import numpy as np

import enceladus
import enceladus.language as tl
from enceladus.configs import attention_configs

LOG2E = 1.4426950408889634


@enceladus.jit
def attention_kernel(q_ptr, k_ptr, v_ptr, o_ptr, sm_scale, N_CTX, stride_h, stride_m,
                     HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                     CAUSAL: tl.constexpr):  # fmt: skip
    start_m = tl.program_id(0)
    base = tl.program_id(1) * stride_h
    q_desc = tl.make_tensor_descriptor(q_ptr + base, [N_CTX, HEAD_DIM], [stride_m, 1],
                                       [BLOCK_M, HEAD_DIM])  # fmt: skip
    k_desc = tl.make_tensor_descriptor(k_ptr + base, [N_CTX, HEAD_DIM], [stride_m, 1],
                                       [BLOCK_N, HEAD_DIM])  # fmt: skip
    v_desc = tl.make_tensor_descriptor(v_ptr + base, [N_CTX, HEAD_DIM], [stride_m, 1],
                                       [BLOCK_N, HEAD_DIM])  # fmt: skip
    o_desc = tl.make_tensor_descriptor(o_ptr + base, [N_CTX, HEAD_DIM], [stride_m, 1],
                                       [BLOCK_M, HEAD_DIM])  # fmt: skip
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
    # Scale by log2(e) so the softmax can use exp2.
    qk_scale = sm_scale * 1.4426950408889634
    q = q_desc.load([start_m * BLOCK_M, 0])
    hi = N_CTX
    if CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
    for start_n in range(0, hi, BLOCK_N):
        qk = tl.dot(q, tl.trans(k_desc.load([start_n, 0]))) * qk_scale
        cols = start_n + offs_n
        mask = cols[None, :] < N_CTX
        if CAUSAL:
            mask = mask & (offs_m[:, None] >= cols[None, :])
        qk = tl.where(mask, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp2(qk - m_ij[:, None])
        alpha = tl.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(q_desc.dtype), v_desc.load([start_n, 0]), acc)
        m_i = m_ij
    acc = acc / l_i[:, None]
    o_desc.store([start_m * BLOCK_M, 0], acc.to(o_desc.dtype))


def _alloc_like(q):
    return np.empty_like(q) if isinstance(q, np.ndarray) else enceladus.empty(q.shape, q.dtype)


def _check(q, k, v):
    if not (q.shape == k.shape == v.shape) or len(q.shape) != 4:
        raise ValueError("q, k, and v must all have shape (batch, heads, seq_len, head_dim)")
    if q.shape[-1] not in (16, 32, 64, 128):
        raise ValueError(f"head_dim must be 16, 32, 64, or 128, not {q.shape[-1]}")


def attention(q, k, v, causal: bool = False, sm_scale: float | None = None, o=None,
              block_m: int = 32, block_n: int = 32, num_warps: int = 4):  # fmt: skip
    """Returns `softmax(q @ k^T * sm_scale) @ v` for contiguous (batch, heads, seq, dim) inputs.

    Args:
        q: Queries, shape (batch, heads, seq_len, head_dim).
        k: Keys, with the same shape as `q`.
        v: Values, with the same shape as `q`.
        causal: Whether query `i` attends only to keys `0..i`.
        sm_scale: The softmax scale; defaults to `1 / sqrt(head_dim)`.
        o: An optional output array with the same shape and dtype as `q`.
        block_m: Query rows per program.
        block_n: Keys per loop iteration.
        num_warps: SIMD groups per program. Each owns `block_m / num_warps` query rows.
    """
    _check(q, k, v)
    z, h, n, d = q.shape
    o = _alloc_like(q) if o is None else o
    scale = 1.0 / math.sqrt(d) if sm_scale is None else sm_scale
    grid = (enceladus.cdiv(n, block_m), z * h)
    attention_kernel[grid](q, k, v, o, scale, n, n * d, d, HEAD_DIM=d, BLOCK_M=block_m,
                           BLOCK_N=block_n, CAUSAL=causal, num_warps=num_warps,
                           dot_warps=(num_warps, 1))  # fmt: skip
    return o


def _prune(configs, named):
    """Keeps the configs that `attention_configs` lists for this dtype and head dimension."""
    allowed = attention_configs(named["q_ptr"].dtype, named["HEAD_DIM"])
    return [c for c in configs if c in allowed]


attention_tuned_kernel = enceladus.autotune(
    configs=attention_configs("float16", 64) + attention_configs("float16", 128),
    key=["N_CTX", "HEAD_DIM", "CAUSAL"],
    prune_configs_by={"early_config_prune": _prune},
)(attention_kernel)


def attention_tuned(q, k, v, causal: bool = False, sm_scale: float | None = None, o=None):
    """Returns the same result as `attention`, using the fastest configuration for the shape."""
    _check(q, k, v)
    z, h, n, d = q.shape
    o = _alloc_like(q) if o is None else o
    scale = 1.0 / math.sqrt(d) if sm_scale is None else sm_scale
    grid = lambda meta: (enceladus.cdiv(n, meta["BLOCK_M"]), z * h)  # noqa: E731
    attention_tuned_kernel[grid](q, k, v, o, scale, n, n * d, d, HEAD_DIM=d, CAUSAL=causal)
    return o


def reference(q, k, v, causal: bool = False, sm_scale: float | None = None, **_):
    """Returns the attention output computed in float64 with NumPy."""
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d) if sm_scale is None else sm_scale
    s = np.einsum("zhmd,zhnd->zhmn", q.astype(np.float64), k.astype(np.float64)) * scale
    if causal:
        n = q.shape[2]
        s = np.where(np.tril(np.ones((n, n), bool)), s, -np.inf)
    s = np.exp(s - s.max(-1, keepdims=True))
    p = s / s.sum(-1, keepdims=True)
    return np.einsum("zhmn,zhnd->zhmd", p, v.astype(np.float64)).astype(q.dtype)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal((2, 3, 100, 64)).astype(np.float16) for _ in range(3))
    for causal in (False, True):
        err = np.abs(attention(q, k, v, causal).astype(np.float32)
                     - reference(q, k, v, causal).astype(np.float32)).max()  # fmt: skip
        print(f"max abs error, causal={causal}: {err:.2e}")
