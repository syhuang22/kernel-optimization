"""Shared benchmarking utilities for TPU kernel experiments."""

import time
import jax
import jax.numpy as jnp


def benchmark(fn, *args, warmup=5, iters=20, name="kernel"):
    """Time a JAX function, returns (mean_ms, std_ms)."""
    # Warmup
    for _ in range(warmup):
        out = fn(*args)
    jax.block_until_ready(out)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    import statistics
    mean_ms = statistics.mean(times)
    std_ms = statistics.stdev(times)
    print(f"{name:30s}: {mean_ms:.3f} ms ± {std_ms:.3f} ms")
    return mean_ms, std_ms


def roofline_matmul(m, k, n, dtype, time_ms):
    """Print roofline analysis for a matmul."""
    flops = 2 * m * k * n
    bytes_read = (m * k + k * n) * jnp.dtype(dtype).itemsize
    bytes_write = m * n * jnp.dtype(dtype).itemsize
    total_bytes = bytes_read + bytes_write
    arithmetic_intensity = flops / total_bytes

    # TPU v5e / v7 roofline numbers (adjust as needed)
    peak_tflops = 918e12   # bf16 MXU
    peak_bw = 7.37e12      # v7 HBM BW in bytes/s

    compute_bound_ms = (flops / peak_tflops) * 1000
    memory_bound_ms = (total_bytes / peak_bw) * 1000
    roofline_ms = max(compute_bound_ms, memory_bound_ms)

    print(f"\n--- Roofline ({m}x{k}x{n} {dtype}) ---")
    print(f"  FLOPs:               {flops/1e9:.1f} GFLOPs")
    print(f"  Arithmetic intensity: {arithmetic_intensity:.1f} FLOPs/byte")
    print(f"  Roofline (compute):  {compute_bound_ms:.3f} ms")
    print(f"  Roofline (memory):   {memory_bound_ms:.3f} ms")
    print(f"  Roofline bound:      {roofline_ms:.3f} ms")
    print(f"  Actual:              {time_ms:.3f} ms")
    print(f"  Efficiency:          {roofline_ms / time_ms * 100:.1f}%")
