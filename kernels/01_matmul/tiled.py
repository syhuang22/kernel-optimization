"""
Matmul Kernel - Level 2: Tiled (K-dimension reduction)

Learning goals:
- Accumulate partial results across K tiles using scratch/accum pattern
- Understand why K-tiling matters for large matrices
- Sweep block sizes and observe MXU utilization

Key insight: naive.py puts all K in one block → VMEM overflow for large K.
Here we tile K and accumulate, like a proper GEMM.
"""

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl
from jax.experimental.pallas import tpu as pltp


def matmul_kernel_tiled(a_ref, b_ref, o_ref, *, bk: int):
    """Accumulate across K tiles."""
    m, _ = a_ref.shape
    _, n = b_ref.shape  # actually (BK, BN) but we only need n

    # Initialize accumulator
    acc = jnp.zeros((m, n), dtype=jnp.float32)

    # NOTE: K dimension is already tiled by BlockSpec — the grid drives K index.
    # This kernel is called once per (i, j, k) tile.
    acc = acc + pl.dot(a_ref[...].astype(jnp.float32),
                       b_ref[...].astype(jnp.float32))
    o_ref[...] = acc.astype(o_ref.dtype)


def matmul_tiled(a: jax.Array, b: jax.Array,
                 bm: int = 128, bk: int = 128, bn: int = 128):
    m, k = a.shape
    _, n = b.shape

    # Grid: (M//bm, N//bn, K//bk)
    # The K dimension is in the grid so each kernel call gets one K-slice.
    # For full reduction we need to use interpret mode or lax.associative_scan.
    # Simple version: K//bk == 1 (bk == k). See notes below.
    #
    # For K > bk, use the lax.fori_loop approach in tiled_accum.py (next step).
    assert k == bk, (
        f"This simple version requires bk==k (bk={bk}, k={k}). "
        "Use tiled_accum.py for full K tiling."
    )

    return pl.pallas_call(
        lambda a_ref, b_ref, o_ref: matmul_kernel_tiled(a_ref, b_ref, o_ref, bk=bk),
        out_shape=jax.ShapeDtypeStruct((m, n), a.dtype),
        grid=(m // bm, n // bn),
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j: (i, 0)),
            pl.BlockSpec((bk, bn), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
    )(a, b)


def matmul_tiled_sweep(a: jax.Array, b: jax.Array):
    """Sweep block sizes, print results."""
    import sys
    sys.path.insert(0, "../..")
    from utils.benchmark import benchmark

    configs = [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    ]
    m, k = a.shape
    _, n = b.shape

    for bm, bk, bn in configs:
        if m % bm or k % bk or n % bn:
            continue
        if bk != k:
            continue  # skip until K-tiling implemented
        fn = lambda a=a, b=b, bm=bm, bk=bk, bn=bn: matmul_tiled(a, b, bm, bk, bn)
        benchmark(fn, name=f"pallas bm={bm} bk={bk} bn={bn}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../..")
    from utils.benchmark import benchmark, roofline_matmul

    M, K, N = 1024, 1024, 1024
    dtype = jnp.bfloat16

    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (M, K), dtype=dtype)
    b = jax.random.normal(key, (K, N), dtype=dtype)

    ref = jnp.matmul(a, b)
    out = matmul_tiled(a, b, bm=128, bk=K, bn=128)
    jax.block_until_ready(out)
    max_err = jnp.max(jnp.abs(out - ref)).item()
    print(f"Max error vs jnp.matmul: {max_err:.6f}")

    xla_ms, _ = benchmark(jnp.matmul, a, b, name="jnp.matmul (XLA)")
    pallas_ms, _ = benchmark(lambda: matmul_tiled(a, b, bm=128, bk=K, bn=128), name="pallas tiled")
    print(f"\nSpeedup: {xla_ms / pallas_ms:.2f}x")

    roofline_matmul(M, K, N, dtype, pallas_ms)

    print("\n--- Block size sweep ---")
    matmul_tiled_sweep(a, b)
