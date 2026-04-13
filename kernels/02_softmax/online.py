"""
Softmax Kernel - Online (numerically stable, single-pass)

Learning goals:
- Use scratchpad (VMEM scratch) for intermediate max/sum accumulators
- Understand why naive two-pass softmax is bad for large sequences
- Online algorithm: track (max, sum) and update in one pass

Algorithm (per row):
  For each block of x:
    m_new = max(m_old, block_max)
    s_new = s_old * exp(m_old - m_new) + sum(exp(block - m_new))
  out = exp(x - m_final) / s_final
"""

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
from functools import partial


def softmax_kernel(x_ref, o_ref, *, seq_len: int, block_size: int):
    """
    Online softmax: 兩趟掃描，每趟每次只看 block_size 個元素。

    第一趟 (fori_loop): 邊掃邊更新 running_max 和 running_sum
      每個 block:
        m_new = max(m_old, block_max)
        s_new = s_old * exp(m_old - m_new) + sum(exp(block - m_new))

    第二趟 (fori_loop): 用最終 m, s 正規化並寫入輸出
      output_block = exp(block - m_final) / s_final
    """
    rows = x_ref.shape[0]          # rows_per_block，例如 16
    num_blocks = seq_len // block_size

    # --- 第一趟：算 global max 和 global sum ---
    def scan_body(j, carry):
        m, s = carry                                               # [rows], [rows]
        x_blk = x_ref[:, pl.dslice(j * block_size, block_size)]  # [rows, block_size]
        x_blk = x_blk.astype(jnp.float32)

        m_new = jnp.maximum(m, jnp.max(x_blk, axis=-1))          # [rows]
        exp_blk = jnp.exp(x_blk - m_new[:, None])
        s_new = s * jnp.exp(m - m_new) + jnp.sum(exp_blk, axis=-1)
        return m_new, s_new

    m_init = jnp.full((rows,), -jnp.inf, dtype=jnp.float32)
    s_init = jnp.zeros((rows,), dtype=jnp.float32)
    m_final, s_final = jax.lax.fori_loop(0, num_blocks, scan_body, (m_init, s_init))

    # --- 第二趟：正規化，寫回輸出 ---
    def norm_body(j, _):
        x_blk = x_ref[:, pl.dslice(j * block_size, block_size)].astype(jnp.float32)
        out_blk = jnp.exp(x_blk - m_final[:, None]) / s_final[:, None]
        o_ref[:, pl.dslice(j * block_size, block_size)] = out_blk.astype(o_ref.dtype)

    jax.lax.fori_loop(0, num_blocks, norm_body, None)


def softmax(x: jax.Array, rows_per_block: int = 16, block_size: int = 512) -> jax.Array:
    """
    x: [rows, seq_len]
    rows_per_block: 每個 block 處理幾行（bf16 需是 16 的倍數）
    block_size:     沿 seq 方向每次掃多少（必須是 128 的倍數）
    """
    rows, seq_len = x.shape
    assert rows % rows_per_block == 0
    assert seq_len % block_size == 0

    return pl.pallas_call(
        partial(softmax_kernel, seq_len=seq_len, block_size=block_size),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(rows // rows_per_block,),
        in_specs=[pl.BlockSpec((rows_per_block, seq_len), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((rows_per_block, seq_len), lambda i: (i, 0)),
    )(x)


# ---- Online softmax for two blocks (illustrates the core algorithm) ----

def online_softmax_two_pass_demo(x: jax.Array) -> dict:
    """
    Pure JAX demo of the online softmax algorithm.
    Shows the math before we move it into a kernel.
    """
    x = x.astype(jnp.float32)
    n = x.shape[-1]
    mid = n // 2
    x1, x2 = x[..., :mid], x[..., mid:]

    # Pass 1: first block
    m1 = jnp.max(x1, axis=-1, keepdims=True)
    s1 = jnp.sum(jnp.exp(x1 - m1), axis=-1, keepdims=True)

    # Pass 2: second block — update running stats
    m2 = jnp.maximum(m1, jnp.max(x2, axis=-1, keepdims=True))
    s2 = (s1 * jnp.exp(m1 - m2)
          + jnp.sum(jnp.exp(x2 - m2), axis=-1, keepdims=True))

    # Final normalization
    out = jnp.concatenate([
        jnp.exp(x1 - m2) / s2,
        jnp.exp(x2 - m2) / s2,
    ], axis=-1)
    return {"output": out, "global_max": m2.squeeze(-1), "global_sum": s2.squeeze(-1)}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../..")
    from utils.benchmark import benchmark

    rows, seq_len = 1024, 4096
    dtype = jnp.bfloat16
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (rows, seq_len), dtype=dtype)

    # Correctness
    ref = jax.nn.softmax(x, axis=-1)
    out = softmax(x)
    jax.block_until_ready(out)
    max_err = jnp.max(jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))).item()
    print(f"Max error vs jax.nn.softmax: {max_err:.6f}")

    # Demo: online algorithm correctness
    demo = online_softmax_two_pass_demo(x[:4])
    ref_small = jax.nn.softmax(x[:4].astype(jnp.float32), axis=-1)
    max_err_demo = jnp.max(jnp.abs(demo["output"] - ref_small)).item()
    print(f"Online softmax demo error:   {max_err_demo:.8f}")

    # Benchmark
    benchmark(lambda: jax.nn.softmax(x, axis=-1), name="jax.nn.softmax (XLA)")
    benchmark(softmax, x, name="pallas softmax")
