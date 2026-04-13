# WAN 2.2 Kernel 優化洞察

## 現況 (2026-04-09 profile, dp=2/tp=4)

| Component | ms/step | % |
|---|---|---|
| **pallas_self_attn** | **1774ms** | **55.7%** |
| other fusions | 495ms | 15.5% |
| alltoall (MoE) | 339ms | 10.6% |
| copy_reshape | 333ms | 10.4% |
| gemm_fusion | 90ms | 2.8% |

**瓶頸很清楚：attention 吃掉超過一半的時間。**

---

## 為什麼 Attention 這麼貴

WAN 2.2 的 self-attention 參數：
- 40 layers，每層 44ms
- S=75600 tokens，H=10 heads，head_dim=128
- 已經用 Pallas splash attention，不是 naive XLA

理論上 attention 的 arithmetic intensity：
```
FLOPs = 4 × B × H × S² × d  (QK matmul + SV matmul)
Bytes = 2 × B × H × S × d × 3  (Q, K, V) + output

對 S=75600, d=128:
AI ≈ S/2 ≈ 37800 FLOP/B  → 遠超 v7 crossover (313)
→ compute-bound，MXU 利用率 72%
```

已經是 compute-bound，代表：
- **減少 FLOPs = 唯一有效方向**
- 更好的 scheduling (pipelining) 可以把 72% → 更高
- 換 sharding 策略幾乎沒用（CP=8 測過，不行）

---

## 為什麼 XLA 的 Flash Attention 和手寫 Pallas 差在哪

### XLA 做不到的事

XLA 看到 attention 的計算圖：
```
S = Q @ K^T        → [N, N] 存到 HBM
P = softmax(S)     → 從 HBM 讀，算，存回 HBM
O = P @ V          → 從 HBM 讀，算
```

因為 softmax 需要 global max/sum（data dependency），
XLA 的 fusion pass 無法合併這三步。

### Flash Attention 的洞察

用 online softmax，把計算圖換掉：
```
for each KV block j:
    S_j = Q @ K_j^T          只有 [Bq, Bkv]，住在 VMEM
    m_new = max(m_old, max(S_j))
    s_new = s_old × exp(m_old - m_new) + sum(exp(S_j - m_new))
    O += rescale + exp(S_j) @ V_j

→ HBM traffic: O(S×d) 不是 O(S²)
→ VMEM 只需要放一個 block，不需要整個 attention matrix
```

**XLA 無法自動做這件事，因為這需要算法層的重寫，不只是 op 調度。**

---

## 已做的優化

### FP8-QK (-7.2%, -3.6ms/layer)
- v7 native E4M3 FP8，QK matmul 從 bf16 → fp8
- MXU throughput 2x
- 直接 cast，不需要 scaling（attention score 的 range 可接受）
- SV 不值得做 FP8：head_dim=128 太小，cast overhead > 計算節省

### Direct layout (-1.3%, -0.7ms/layer)
- 消除 post-kernel 的 swapaxes/copy

---

## 下一步方向（按 ROI 排序）

### 1. Sparse Attention（最高 ROI，但有品質風險）
- 理論：attention 的 softmax mass 集中在少數 KV blocks
- 做法：block scoring → top-k gather → 只算 15% KV
- 預期：44ms → ~12ms/layer（-73%），整體 -54s
- 風險：訓練時沒用 sparse，inference 突然用可能影響品質

### 2. Software Pipelining（零品質風險）
- MXU 算 tile[i] 的 SV 時，VPU 可以同時做 softmax 更新
- 目前：MXU → VPU → MXU 串行
- 目標：MXU ‖ VPU overlap → 理論上接近 2x MXU utilization
- 難度：需要手動排 VLIW，`emit_pipeline` API

### 3. Padding Fix (+6%, +100ms/step)
- S=75600 不能被 128 整除（75600/128 = 590.625）
- 現在 pad 到 76032，浪費 6% compute
- 修法：在 model 層讓 S 變成 128 對齊（設計層問題）

### 4. convert_reduce_fusion (495ms, 15.5%)
- 目前還沒 profile 清楚是什麼
- 可能是 norm + dtype convert + reduce 的組合
- 值得下次 profile 時深挖

---

## 為什麼 Pallas elementwise 打不贏 XLA

測過 norm+adaln+linear：
- XLA: 0.62ms（已經 fuse 成單一 HLO fusion）
- Pallas: 0.96ms（+54%）

XLA 的 fusion pass 對 elementwise 已經最優。
Pallas custom_call dispatch overhead 讓它更慢。

**規則：只在 XLA 做不到的地方用 Pallas。**
- ✅ Flash Attention（算法層 fuse，XLA 做不到）
- ✅ Sparse Attention（動態 gather pattern，XLA 做不到）
- ❌ Norm / elementwise fusion（XLA 已經最優）

---

## Kernel 優化的本質（對 WAN 2.2 的應用）

Layer 3（時間層）是唯一能再動的地方：
- Layer 1（功能）：attention 計算正確 ✅
- Layer 2（資源）：sharding 已最優 (tp=4+dp=2) ✅
- Layer 3（時間）：MXU/VPU pipeline、sparse pattern — 還有空間

**WAN 2.2 的 kernel 優化 = 在 55.7% 的 attention 裡，擠出更多的 MXU 利用率，或減少總 FLOPs。**
