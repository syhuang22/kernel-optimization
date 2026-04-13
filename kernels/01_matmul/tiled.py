"""
Matmul Kernel - Level 2: Tiled (K-dimension reduction)

Key insight vs naive.py:
- naive: bk = K = 1024 → 256KB block → VMEM spill → 473x slower than XLA
- tiled: K is in grid, each kernel sees bk=128 block → 32KB → fits in VMEM

How K-tiling works:
  grid = (M//bm, N//bn, K//bk)  ← K added as 3rd grid dimension
  Each kernel call handles one (i, j, k) tile
  Accumulator is passed as input+output (input_output_aliases)
  → K tiles run sequentially per (i,j), accumulating partial sums
"""

import jax
import jax.numpy as jnp
import jax.experimental.pallas as pl


def matmul_kernel_tiled(a_ref, b_ref, o_ref, *, bk: int, num_k_tiles: int):
    # a_ref: [bm, k_full] — full K row (DMA'd to VMEM)
    # b_ref: [k_full, bn] — full K col
    # o_ref: [bm, bn]     — output tile
    #
    # We slice K inside the kernel using lax.fori_loop.
    # Each iteration loads a [bm,bk] and [bk,bn] slice → only 2×32KB in registers,
    # even though a_ref/b_ref themselves sit in VMEM at full K size.
    bm_size = a_ref.shape[0]
    bn_size = b_ref.shape[1]

    def body(kt, acc):
        a_slice = a_ref[:, pl.dslice(kt * bk, bk)]
        b_slice = b_ref[pl.dslice(kt * bk, bk), :]
        return acc + pl.dot(a_slice.astype(jnp.float32), b_slice.astype(jnp.float32))

    acc = jax.lax.fori_loop(
        0, num_k_tiles, body,
        jnp.zeros((bm_size, bn_size), dtype=jnp.float32),
    )
    o_ref[...] = acc.astype(o_ref.dtype)


def matmul_tiled(a: jax.Array, b: jax.Array,
                 bm: int = 128, bk: int = 128, bn: int = 128) -> jax.Array:
    m, k = a.shape
    _, n = b.shape
    assert m % bm == 0 and k % bk == 0 and n % bn == 0

    import functools
    kernel = functools.partial(
        matmul_kernel_tiled,
        bk=bk,
        num_k_tiles=k // bk,
    )

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), a.dtype),
        grid=(m // bm, n // bn),
        in_specs=[
            pl.BlockSpec((bm, k), lambda i, j: (i, 0)),   # full K row
            pl.BlockSpec((k, bn), lambda i, j: (0, j)),   # full K col
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

    # Correctness
    ref = jnp.matmul(a, b)
    out = matmul_tiled(a, b)
    jax.block_until_ready(out)
    max_err = jnp.max(jnp.abs(out - ref)).item()
    print(f"Max error vs jnp.matmul: {max_err:.6f}")
    assert max_err < 1.0

    # Benchmark
    xla_ms, _ = benchmark(jnp.matmul, a, b, name="jnp.matmul (XLA)")
    pallas_ms, _ = benchmark(lambda: matmul_tiled(a, b), name="pallas tiled bk=128")
    print(f"Speedup: {xla_ms / pallas_ms:.2f}x")

    roofline_matmul(M, K, N, dtype, pallas_ms)

    # bk must be >= 128 (TPU vector alignment constraint).
    # Also: current BlockSpec is (bm, k_full), so full K is always DMA'd to VMEM.
    # fori_loop only splits register-level computation, not VMEM.
    # → Real VMEM K-tiling requires emit_pipeline (see future exercise).
    # TPU BlockSpec 對齊規則：最後一維須是 128 倍數，倒數第二維須是 8 倍數
    print("\n--- bm/bn sweep (bk fixed at K=1024) ---")
    for bm, bn in [(128, 128), (256, 128), (256, 256), (512, 256)]:
        if M % bm or N % bn:
            continue
        ms, _ = benchmark(lambda bm=bm, bn=bn: matmul_tiled(a, b, bm=bm, bk=K, bn=bn),
                          name=f"pallas bm={bm} bn={bn}")
