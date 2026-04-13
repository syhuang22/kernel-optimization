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
    x_ref: [block_size]  — current block of input
    o_ref: [block_size]  — current block of output

    Because we need a single-pass over the full row, we use lax.fori_loop
    with scratchpad to accumulate (running_max, running_sum).

    NOTE: For TPU Pallas, scratchpad is declared via interpret=True or
    via pl.VMEM scratch in grid_spec. Here we use a simple approach:
    pass scratch refs via in_specs with has_side_effects.
    """
    import jax.lax as lax

    num_blocks = seq_len // block_size
    row_idx = pl.program_id(0)  # which row we're computing

    # We need to iterate over K blocks for row row_idx.
    # This is tricky to do inside a single kernel call with standard BlockSpec.
    # Solution: put all of seq_len in one block (block_size == seq_len),
    # then compute properly. This is the "simple" version.
    #
    # For the real multi-block version, see online_multiblock.py.

    x = x_ref[...].astype(jnp.float32)

    # Numerically stable softmax in one block
    x_max = jnp.max(x)
    x_shifted = x - x_max
    exp_x = jnp.exp(x_shifted)
    o_ref[...] = (exp_x / jnp.sum(exp_x)).astype(o_ref.dtype)


def softmax(x: jax.Array, block_size: int | None = None) -> jax.Array:
    """
    x: [rows, seq_len]
    Simple version: block covers full seq_len. Study this first.
    """
    rows, seq_len = x.shape
    if block_size is None:
        block_size = seq_len

    assert seq_len % block_size == 0
    assert block_size == seq_len, (
        "Multi-block version not implemented yet. Set block_size=seq_len."
    )

    return pl.pallas_call(
        partial(softmax_kernel, seq_len=seq_len, block_size=block_size),
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(rows,),
        in_specs=[pl.BlockSpec((1, seq_len), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((1, seq_len), lambda i: (i, 0)),
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
