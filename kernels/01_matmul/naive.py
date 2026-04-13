"""
Matmul Kernel - Level 1: Naive (single block, no tiling)

Learning goals:
- Understand pallas_call API
- Understand BlockSpec and grid indexing
- Baseline to compare against tiled.py
"""

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl


def matmul_kernel(a_ref, b_ref, o_ref):
    # a_ref: [BM, BK], b_ref: [BK, BN], o_ref: [BM, BN]
    # pl.dot = MXU matmul, maps directly to TPU matrix unit
    o_ref[...] = pl.dot(a_ref[...], b_ref[...])


def matmul(a: jax.Array, b: jax.Array, bm: int = 128, bk: int = 128, bn: int = 128):
    m, k = a.shape
    _, n = b.shape
    assert m % bm == 0, f"M={m} not divisible by bm={bm}"
    assert k % bk == 0, f"K={k} not divisible by bk={bk}"
    assert n % bn == 0, f"N={n} not divisible by bn={bn}"

    return pl.pallas_call(
        matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), a.dtype),
        grid=(m // bm, n // bn),
        in_specs=[
            # For grid tile (i, j): a_ref gets row-block i, all of K
            pl.BlockSpec((bm, bk), lambda i, j: (i, 0)),
            # b_ref gets all of K, col-block j
            pl.BlockSpec((bk, bn), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
    )(a, b)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../..")
    from utils.benchmark import benchmark, roofline_matmul

    M, K, N = 1024, 1024, 1024
    dtype = jnp.bfloat16

    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (M, K), dtype=dtype)
    b = jax.random.normal(key, (K, N), dtype=dtype)

    # Correctness check
    ref = jnp.matmul(a, b)
    out = matmul(a, b)
    jax.block_until_ready(out)
    max_err = jnp.max(jnp.abs(out - ref)).item()
    print(f"Max error vs jnp.matmul: {max_err:.6f}")
    assert max_err < 1.0, "Too large — something is wrong"

    # Benchmark
    xla_ms, _ = benchmark(jnp.matmul, a, b, name="jnp.matmul (XLA)")
    pallas_ms, _ = benchmark(matmul, a, b, name="pallas naive")
    print(f"\nSpeedup: {xla_ms / pallas_ms:.2f}x  (>1 = Pallas faster)")

    roofline_matmul(M, K, N, dtype, pallas_ms)
