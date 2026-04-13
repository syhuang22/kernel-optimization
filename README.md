# kernel-optimization

TPU Pallas kernel learning lab. Three exercises, increasing difficulty.

## Structure

```
kernels/
  01_matmul/     — GridSpec, BlockSpec, tiling basics
  02_softmax/    — Scratchpad memory, online algorithm
  03_flash_attention/ — Fused attention, IO/compute overlap
utils/
  benchmark.py   — Timing + roofline analysis
notes/           — Personal observations per exercise
```

## Setup

```bash
conda activate nebula
source ~/tpu_env.sh
```

## Learning Path

1. `kernels/01_matmul/naive.py` — get a kernel running, understand the API
2. `kernels/01_matmul/tiled.py` — sweep block sizes, see MXU utilization
3. `kernels/02_softmax/online.py` — scratchpad, online softmax algorithm
4. `kernels/03_flash_attention/forward.py` — fused attention, the real deal

## Running

```bash
cd kernels/01_matmul
python naive.py
```
