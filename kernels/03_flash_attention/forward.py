"""
Flash Attention - Forward Pass

Learning goals:
- Combine matmul + online softmax into a fused kernel
- Understand why tiling over seq_len avoids O(N^2) HBM usage
- See how BlockSpec drives the KV iteration loop

Algorithm (per query block i):
  For each KV block j:
    S_ij = Q_i @ K_j^T / sqrt(d)              # [Bq, Bkv]
    m_new = max(m_old, rowmax(S_ij))
    P_ij = exp(S_ij - m_new)
    l_new = l_old * exp(m_old - m_new) + rowsum(P_ij)
    O_i = O_i * exp(m_old - m_new) + P_ij @ V_j
  O_i = O_i / l_new

This is the core insight of Flash Attention:
  - O(N*d) VMEM instead of O(N^2)
  - MXU stays busy the whole time
"""

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
from functools import partial


def flash_attention_kernel(
    q_ref,   # [Bq, d]
    k_ref,   # [Bkv, d]  — full KV dimension via grid
    v_ref,   # [Bkv, d]
    o_ref,   # [Bq, d]
    *,
    seq_len: int,
    bq: int,
    bkv: int,
    sm_scale: float,
):
    """
    Single kernel invocation handles one (batch, head, q_block) tile.
    Iterates over KV blocks with lax.fori_loop.
    """
    import jax.lax as lax

    bq_actual, d = q_ref.shape
    num_kv_blocks = seq_len // bkv

    q = q_ref[...].astype(jnp.float32)

    # Running statistics
    m = jnp.full((bq_actual,), -jnp.inf, dtype=jnp.float32)
    l = jnp.zeros((bq_actual,), dtype=jnp.float32)
    o = jnp.zeros((bq_actual, d), dtype=jnp.float32)

    def kv_step(j, carry):
        m, l, o = carry
        # Load KV block j
        k = pl.load(k_ref, (pl.dslice(j * bkv, bkv), slice(None))).astype(jnp.float32)
        v = pl.load(v_ref, (pl.dslice(j * bkv, bkv), slice(None))).astype(jnp.float32)

        # Attention scores: [Bq, Bkv]
        s = jnp.dot(q, k.T) * sm_scale

        # Online softmax update
        m_new = jnp.maximum(m, jnp.max(s, axis=-1))
        exp_s = jnp.exp(s - m_new[:, None])
        l_new = l * jnp.exp(m - m_new) + jnp.sum(exp_s, axis=-1)
        o_new = o * jnp.exp(m - m_new)[:, None] + jnp.dot(exp_s, v)

        return m_new, l_new, o_new

    m, l, o = lax.fori_loop(0, num_kv_blocks, kv_step, (m, l, o))

    # Normalize
    o = o / l[:, None]
    o_ref[...] = o.astype(o_ref.dtype)


def flash_attention(
    q: jax.Array,  # [batch, heads, seq_q, d]
    k: jax.Array,  # [batch, heads, seq_kv, d]
    v: jax.Array,  # [batch, heads, seq_kv, d]
    bq: int = 128,
    bkv: int = 128,
) -> jax.Array:
    batch, heads, seq_q, d = q.shape
    _, _, seq_kv, _ = k.shape
    sm_scale = d ** -0.5

    assert seq_q % bq == 0
    assert seq_kv % bkv == 0

    # Reshape: merge batch*heads for grid
    q_2d = q.reshape(batch * heads, seq_q, d)
    k_2d = k.reshape(batch * heads, seq_kv, d)
    v_2d = v.reshape(batch * heads, seq_kv, d)

    kernel = partial(
        flash_attention_kernel,
        seq_len=seq_kv,
        bq=bq,
        bkv=bkv,
        sm_scale=sm_scale,
    )

    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(q_2d.shape, q_2d.dtype),
        grid=(batch * heads, seq_q // bq),
        in_specs=[
            pl.BlockSpec((1, bq, d), lambda bh, i: (bh, i, 0)),
            # K and V: full seq_kv passed in, kernel iterates internally
            pl.BlockSpec((1, seq_kv, d), lambda bh, i: (bh, 0, 0)),
            pl.BlockSpec((1, seq_kv, d), lambda bh, i: (bh, 0, 0)),
        ],
        out_specs=pl.BlockSpec((1, bq, d), lambda bh, i: (bh, i, 0)),
    )(q_2d, k_2d, v_2d)

    return out.reshape(batch, heads, seq_q, d)


def reference_attention(q, k, v):
    """Standard attention for correctness check."""
    d = q.shape[-1]
    scale = d ** -0.5
    attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
    attn = jax.nn.softmax(attn, axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", attn, v)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../..")
    from utils.benchmark import benchmark

    batch, heads, seq_len, d = 1, 8, 1024, 64
    dtype = jnp.bfloat16
    key = jax.random.PRNGKey(0)
    q = jax.random.normal(key, (batch, heads, seq_len, d), dtype=dtype)
    k = jax.random.normal(key, (batch, heads, seq_len, d), dtype=dtype)
    v = jax.random.normal(key, (batch, heads, seq_len, d), dtype=dtype)

    # Correctness
    ref = reference_attention(q, k, v)
    out = flash_attention(q, k, v)
    jax.block_until_ready(out)
    max_err = jnp.max(jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))).item()
    print(f"Max error vs reference: {max_err:.6f}")

    # Benchmark
    benchmark(lambda: reference_attention(q, k, v), name="reference attention (XLA)")
    benchmark(lambda: flash_attention(q, k, v), name="flash attention (Pallas)")

    # Bonus: show memory savings
    n = seq_len
    naive_mem_mb = batch * heads * n * n * 2 / 1e6  # bf16 attention matrix
    flash_mem_mb = batch * heads * (n * d) * 2 / 1e6 * 3  # Q, K, V blocks only
    print(f"\nMemory: naive={naive_mem_mb:.1f}MB  flash={flash_mem_mb:.1f}MB  "
          f"ratio={naive_mem_mb/flash_mem_mb:.1f}x")
