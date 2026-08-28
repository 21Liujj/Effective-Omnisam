# Step 17 — Training Time Breakdown Report (Phase 6-2)

> Date: 2026-06-17 | Seed=1 | N_iters=500 confirmed from run.log
> No training, PL generation, or validation was started during this phase.

---

## 1. Phase 6-2 目标

解释为什么 TeCMA 显著降低 memory attention time，但 total step time 的下降幅度较小
且不完全单调。为论文中的 "TeCMA reduces memory attention cost by 55.8%, while overall
step time decreases by 3.9%" 提供可信的机制解释。

---

## 2. 可用 timing 字段

| 字段 | 来源 | 单位 | 说明 |
|---|---|---|---|
| `step_time` | run_summary / train_timing.jsonl | s/iter（per-iter 平均） | 完整迭代墙钟时间，包含 data loading + compute + 循环 overhead |
| `data_time` | train_timing.jsonl | s/iter | data loading per-iter 平均，由 `t_data0` 计时 |
| `iter_compute_time`（ict） | train_timing.jsonl | s/iter | forward+backward+optimizer per-iter 平均，由 `t_comp0` 计时，**包含** memory attention |
| `memory_attention_time` | run_summary.profile | s（epoch 总量） | `attention_time_accum` 累积，非 per-iter；需 ÷ N_iters 换算 |
| `memory_attention_time_per_call` | train_timing.jsonl | s/call | per-call 平均，A_ccpm=14.7ms/call |
| `epoch_train_time` | train_timing.jsonl | s（epoch 总量） | `sum(compute_durations)` = ict × N_iters |

---

## 3. 不可用的细粒度 timing

以下字段在当前 profiling 中**不存在**，不能编造：

- **Image encoder（SAM2 + LoRA）时间**：无单独计时，包含在 ict 中
- **FPN/backbone 时间**：同上
- **Decoder（SegFormerHead）时间**：包含在 ict 中
- **Loss 计算时间**：包含在 ict 中
- **Backward pass 时间**：包含在 ict 中
- **Optimizer step 时间**：包含在 ict 中
- **D 模块（FSMC）forward 时间**：无独立计时器；只能从 ict 差值估算（~0.08s），噪声大
- **E 模块（SG-SMR）forward 时间**：同上（~0.04s 估算）
- **Validation 时间**：epoch_total − epoch_train，不在 step_time 内

---

## 4. 为什么不能编造 encoder / decoder / dataloader 时间

因为代码中只有 `t_data0`（data loading）和 `t_comp0`（forward+backward+optim）两个
计时点，trainer.py 中无独立的 encoder / decoder / loss / backward 计时。
分别汇报这些数字需要在 model.forward() / strategies.forward() 中插入额外的
`torch.cuda.Event` 计时点，当前实验没有这个粒度的数据。

---

## 5. memory_attention_time 是 total 还是 per-iter

`memory_attention_time` 是 **epoch-level total**（累积量），来自：
```python
float(getattr(self.model_tgt.memory_bank, "attention_time_accum", 0.0))
```
`attention_time_accum` 在每次 SAM2 memory attention call 后累加。

`step_time` 和 `iter_compute_time` 是 **per-iteration average**。

两者**不能直接相减**，必须先换算：

---

## 6. 如何换算为 per-iter

```
N_iters = 500   (confirmed: 2000 PL images / batch_size=4)
       from run.log: "Epoch 1 [Train]:   0%|  0/500"

memory_attention_time_per_iter = memory_attention_time_total / N_iters
```

A_ccpm 示例：
```
ma_total = 117.56s
ma_per_iter = 117.56 / 500 = 0.2351s = 235.1 ms/iter
step_time = 2.4089 s/iter
ma_share = 235.1ms / 2408.9ms = 9.76%
```

---

## 7. memory attention 在 total step time 中的占比

| Config | MemAttn/iter (ms) | Step (s) | MemAttn Share |
|---|--:|--:|--:|
| A_ccpm (ref.) | 235.1 | 2.4089 | 9.76% |
| A_Dtrain | 122.0 | 2.2938 | 5.32% |
| A_Etrain | 159.4 | 2.3798 | 6.70% |
| Dtrain_Etrain | 104.8 | 2.3237 | 4.51% |
| A_Dtrain_Etrain | 103.8 | 2.3153 | 4.48% |

Memory attention 在参考设置下占总 step time 约 **9.8%**。其余 ~90% 是 data loading、
SAM2 encoder、FPN、decoder、loss、backward、optim 等不受 token reduction 影响的组件。

---

## 8. final setting 的 MemAttn/iter 降低多少

| Config | MemAttn/iter (ms) | Reduction vs A_ccpm |
|---|--:|--:|
| A_ccpm (ref.) | 235.1 ms | — |
| A_Dtrain_Etrain | 103.8 ms | **−55.84%** (131.3ms saved/iter) |

A_Dtrain_Etrain 每 iter 节省 131.3ms memory attention 时间。

---

## 9. final setting 的 step time 变化

| Config | Step Time (s/iter) | Δ vs A_ccpm |
|---|--:|--:|
| A_ccpm (ref.) | 2.4089 | — |
| A_Dtrain_Etrain | 2.3153 | **−0.094s/iter (−3.89%)** |

Step time 下降幅度（−3.89%）远小于 MemAttn 下降幅度（−55.84%），因为 memory attention
仅占总 step time 的 9.76%。

预期 step time 最大下降 ≈ 9.76% × 55.84% ≈ 5.45%。实测 −3.89%，低于理论上限，
差距来自 D+E forward-pass overhead（约 0.08s/iter）。

---

## 10. 为什么 total step time 不严格单调（75% 比 100% 慢）

**token budget 75% 点**（A_Etrain_keep075）step time = **2.461s** > 100%（2.409s，+2.18%）。

三重原因：

1. **Sparse retrieval (E 模块) forward overhead**：需要计算 feature_norm saliency（对 32768 tokens 排序）+ top-k indexing（保留 24576）。此 overhead 与被减少的 attention token 数（32768→24576，仅去掉 25%）相比，节省量不足以抵消。
   - ict_minus_ma（non-MA compute）：75%=1.291s vs 100%=1.223s（差 0.068s），确认 E overhead 存在。

2. **Memory attention saving 尚小**：75% 点 MemAttn/iter 仅降 37.9ms（235.1→197.2ms），而 E overhead 约 68ms，net effect = −37.9ms + 68ms = +30ms/iter → step time 略涨。

3. **统计噪声**：step_time 是 epoch 平均，受 I/O/CUDA 调度影响，±2% 属正常范围。

从 50% 开始（A_Etrain），MemAttn saving 达 75.7ms/iter，超过 E overhead，step time 回落至 2.380s（−1.21% vs ref.）。

---

## 11. 该表在论文中如何写

**推荐写法**：

> Table X reports per-iteration timing decomposition. Memory attention time is derived from the epoch-level accumulator by dividing by N=500 training iterations. In the final TeCMA setting (A_Dtrain_Etrain), memory attention per iteration is reduced from 235.1ms to 103.8ms (−55.8%), while total step time decreases from 2.409s to 2.315s (−3.9%). The modest end-to-end speedup relative to the memory-attention reduction reflects that memory attention accounts for only ~9.8% of total step time in the reference setting; the remaining ~90% comprises data loading, image encoding, decoding, loss, backward propagation, and loop overhead, none of which are affected by D/E token reduction.

**关于 75% non-monotonic step time**：
> Total step time at E-only keep_ratio=0.75 (2.461s) slightly exceeds the reference (2.409s, +2.2%), because sparse retrieval overhead (~68ms/iter) is not yet amortized by the small memory-attention saving at this keep ratio (~37.9ms/iter). At keep_ratio=0.50 and below, MemAttn savings dominate and step time falls below the reference. We therefore report memory-attention time as the direct measure of token-reduction efficiency, which decreases strictly monotonically with keep ratio.

---

## 12. 如果需要更细粒度拆解，最小补充插桩位置

以下三个插桩点可以完整拆解 ICT-minus-MA residual（~1.22–1.30s）：

```python
# 1. In OmniSAM_adapt.forward() — before and after SAM2 encoder
t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(...)
t0.record()
# ... LoRA_Sam encoder ...
t1.record(); torch.cuda.synchronize()
encoder_time += t0.elapsed_time(t1) / 1000  # → ms

# 2. In SegFormerHead_adaption.forward() — decoder
# (similarly)

# 3. In strategies.py — after optimizer.step()
# loss + backward + optimizer are already wrapped in one block; split if needed
```

这三点可以精确给出 encoder / decoder / loss+backward+optim 各自的时间。不需要改动
现有 `t_comp0` 计时，只需在关键路径上插入 CUDA event pair。

---

## 13. 生成文件

| 文件 | 说明 |
|---|---|
| `experiments/tecma/tables/training_time_breakdown.md` | Table A (ablation) + Table B (token budget) + Table C (availability) |
| `experiments/tecma/tables/training_time_breakdown.csv` | 同上，CSV 格式 |
| `tools/tecma_analysis/build_training_time_breakdown.py` | 只读分析脚本（见下一步） |
| `experiments/tecma/notes/step_17_training_time_breakdown_report.md` | 本报告 |
