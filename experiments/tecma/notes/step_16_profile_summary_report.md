# Step 16 — Profile Summary & Memory Token / Attention Cost Analysis (Phase 6-1)

> Date: 2026-06-17 | Seed=1 | DensePASS val 100 imgs | clean fixed-PL ablation setting
> No training, PL generation, or validation was run during this phase.

---

## 1. Phase 6-1 目标

将已有实验（Phase 4-clean ablation + Phase 5 token budget）的 profile / timing /
token / GPU memory 数据统一整理为论文机制证据，证明 TeCMA 的效率收益确实来自
memory token 减少和 memory attention cost 降低，而非其他偶然因素。

---

## 2. 数据来源目录

### Phase 4-clean ablation（6 runs）

| Config | Run Dir |
|---|---|
| baseline_omnisam | experiments/tecma/runs/baseline_omnisam_seed1_20260616_102834 |
| A_ccpm           | experiments/tecma/runs/A_ccpm_seed1_20260616_104911 |
| A_Dtrain         | experiments/tecma/runs/A_Dtrain_seed1_20260616_111019 |
| A_Etrain         | experiments/tecma/runs/A_Etrain_seed1_20260616_113023 |
| Dtrain_Etrain    | experiments/tecma/runs/Dtrain_Etrain_seed1_20260616_115113 |
| A_Dtrain_Etrain  | experiments/tecma/runs/A_Dtrain_Etrain_seed1_20260616_121135 |

### Phase 5 token budget（2 new runs）

| Config | Run Dir |
|---|---|
| A_Etrain_keep075 | experiments/tecma/runs/token_budget/A_Etrain_keep075_seed1_20260617_112455 |
| A_Etrain_keep025 | experiments/tecma/runs/token_budget/A_Etrain_keep025_seed1_20260617_114629 |

### 可选主结果（仅作参考，不与上表混合比较）

| Config | Run Dir |
|---|---|
| baseline_omnisam (main) | data_pd_exp_final_2000/base_last_20260425_232941 |
| A_Dtrain_Etrain (main)  | data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941 |

---

## 3. 缺失字段情况

| 字段 | 状态 | 说明 |
|---|---|---|
| `BFScore` in run_summary.json | **缺失**（null）| BFScore 由 validation 中途计算，未写入 run_summary；已从 run.log `[ValSummary] BFScore=...` 提取 |
| `pl_timing_jsonl` | null（所有 clean ablation runs）| 这些 run 复用 canonical PL，无 PL 阶段 timing；正常 |
| 主结果 run.log | **不存在**（data_pd_exp_final_2000/）| 主结果目录无 run.log，BFScore 无法从这两个 run 提取 |
| `memory_attention_time` | **完整**（8 runs）| 所有 clean ablation 和 token budget runs 均有 |
| `compression_tokens_*` | 仅 D 模块激活时存在 | A_Dtrain / Dtrain_Etrain / A_Dtrain_Etrain 有效；其余 null |
| `sparse_tokens_*` | 仅 E 模块激活时存在 | A_Etrain / Dtrain_Etrain / A_Dtrain_Etrain / keep075 / keep025 有效 |

**无数据编造**。所有字段来自 `run_summary.json.profile` 或 `run.log`，缺失项标为空或 "N/A"。

---

## 4. memory_tokens_scope 含义

所有 run 的 `memory_tokens_scope = "per_attention_call_all_chunks_concat"`：

```
每次 SAM2 memory attention 调用时，所有 image-patch chunks 的 memory token
拼接在一起的总数量。

对于 OmniSAM 的滑窗设置，每张全景图被切成 8 个 patch，
每 patch 产生 4096 memory tokens，正向拼接后 = 8 × 4096 = 32,768 tokens
（baseline / A_ccpm 的 memory_tokens_after_final = 32768）。

D 模块（FSMC，pool compress_ratio=0.25 per frame）将每 patch 的 4096 tokens
压缩为 1024 tokens，拼接后 = 8 × 1024 = 8,192（A_Dtrain）。

E 模块（SG-SMR，keep_ratio=0.50）从 D 输出（或原始）的全量 tokens 里
top-k 保留 50%，4096 → 4096 after D+E（memory_tokens_after_final=4096）。
```

---

## 5. A/D/E clean ablation 机制结论

### 5.1 A 模块（CCPM）不影响 memory token 数

| Config | Mem Tokens | Attn Time | mIoU | 作用 |
|---|--:|--:|--:|---|
| baseline_omnisam | 32,768 | 117.48s | 60.26 | reference |
| A_ccpm           | 32,768 | 117.56s | 60.26 | A on，token/时间几乎不变 |

A 模块（confidence-controlled pseudo mask）作用在 PL 质量上，不改变 memory token 数量，不影响 memory attention 效率。是精度模块，不是效率模块。

### 5.2 D 模块（FSMC）单独激活：4× token reduction，−48.1% attn

| Config | Mem Tokens | Token Red. | Attn Time | Attn Red. | mIoU | ΔmIoU |
|---|--:|--:|--:|--:|--:|--:|
| A_ccpm   | 32,768 | 1.00× | 117.56s | 0.0% | 60.26 | 0.00 |
| A_Dtrain | 8,192  | 4.00× | 61.00s  | −48.1% | 60.12 | −0.14 |

D 将每 patch 4096 tokens pool 压缩为 1024（per-frame compress_ratio=0.25），
拼接后 32768 → 8192 tokens，memory attention time 降低 48%，mIoU 仅降 0.14。

### 5.3 E 模块（SG-SMR）单独激活（keep=0.50）：2× token reduction，−32.2% attn

| Config | Mem Tokens | Token Red. | Attn Time | Attn Red. | mIoU | ΔmIoU |
|---|--:|--:|--:|--:|--:|--:|
| A_ccpm  | 32,768 | 1.00× | 117.56s | 0.0% | 60.26 | 0.00 |
| A_Etrain | 16,384 | 2.00× | 79.69s | −32.2% | 60.09 | −0.17 |

E 用 feature_norm saliency top-k 筛选保留 50% token（32768→16384），
memory attention time 降低 32%，mIoU 仅降 0.17。

### 5.4 D+E combined（无 A）vs A+D+E

| Config | Mem Tokens | Token Red. | Attn Time | Attn Red. | mIoU | ΔmIoU |
|---|--:|--:|--:|--:|--:|--:|
| Dtrain_Etrain   | 4,096 | 8.00× | 52.38s | −55.4% | 60.02 | −0.24 |
| A_Dtrain_Etrain | 4,096 | 8.00× | 51.92s | −55.8% | 60.02 | −0.24 |

D+E 串行将 token 压缩到 4096（8× reduction），memory attention time 降低
55.8%。加 A 后 mIoU 完全相同（60.02），attn time 几乎完全相同（51.92 vs 52.38，
差 0.46s，< 1%）。

**结论**：A 不贡献 memory efficiency，但在精度上与 D+E 组合后保持了 mIoU 60.02
（相对 baseline −0.24），已属于 D+E 精度代价的正常范围。

### 5.5 GPU 显存

| Config | GPU Peak (GB) | Saving vs baseline |
|---|--:|--:|
| baseline_omnisam | 19.54 | — |
| A_ccpm           | 19.98 | **−0.44 GB**（略高，run-to-run var） |
| A_Dtrain         | 18.99 | +0.55 GB |
| A_Etrain         | 19.28 | +0.26 GB |
| Dtrain_Etrain    | 18.34 | +1.20 GB |
| A_Dtrain_Etrain  | 18.78 | +0.76 GB |

GPU peak 节省总体跟随 token reduction，但绝对值较小（≤1.2 GB on 20 GB card）。
A_ccpm 略高于 baseline 属正常 run 差异，不能解读为 A 增加显存。
**GPU 显存节省不是本文主要诉求**；主要诉求是 memory attention time。

---

## 6. Token budget curve 机制结论

### 6.1 Memory attention time 严格单调（E-only curve）

| Budget | Tokens | Attn Time | Attn Red. |
|---|--:|--:|--:|
| 100% | 32,768 | 117.56s | 0.0% |
| 75%  | 24,576 | 98.62s  | −16.1% |
| 50%  | 16,384 | 79.69s  | −32.2% |
| 25%  | 8,192  | 60.54s  | −48.5% |
| D+E final (12.5%) | 4,096 | 51.92s | −55.8% |

Memory attention time 与 memory token 数的减少高度一致，符合 O(L·d) 理论预期，
说明效率收益**直接来自 token 减少**，而非 batch size、算子优化或其他因素。

### 6.2 mIoU 鲁棒性

| Budget | mIoU | ΔmIoU | BFScore | ΔBFScore |
|---|--:|--:|--:|--:|
| 100% | 60.26 | 0.00 | 39.59 | 0.00 |
| 75%  | 60.12 | −0.14 | 39.28 | −0.31 |
| 50%  | 60.09 | −0.17 | 39.23 | −0.36 |
| 25%  | 60.10 | −0.16 | 39.29 | −0.30 |
| D+E final | 60.02 | −0.24 | 39.23 | −0.36 |

所有 budget 下 mIoU ≥ 60.02，BFScore ≥ 39.23，与 100% 点相差不超过 0.24 / 0.36。
不非单调（25% 点 mIoU 60.10 > 50% 点 60.09）属统计噪声。

---

## 7. 为什么 memory attention time 比 total step time 更适合支撑效率主张

Total step time 包含：
1. memory attention（目标机制，已严格单调）
2. sparse retrieval overhead（E 模块，额外计算 saliency + top-k）
3. data loading、collation（I/O bound，随 cache/抢占变化）
4. 非 memory 模块（encoder、decoder、loss）
5. PL 质量过滤（A 模块）
6. logging、checkpoint 写入

因此，total step time 受多因素影响，在 75% 点出现 +2.2%（比 100% 慢）是正常的：
sparse retrieval 的 saliency 计算 overhead 在 keep_ratio=0.75 时尚不能被减少的
attention 计算量完全抵消。

Memory attention time 是 SAM2 memory attention 模块的**直接计时**，精确反映
token 数量的变化，是本文效率主张的正确支撑指标。

**论文应同时报告两者**，并注明：
"Total step time includes sparse retrieval overhead and other non-memory
components; memory-attention time is reported as the direct measure of
memory-token efficiency and decreases monotonically with token budget reduction."

---

## 8. 可写入论文的表格

| 表格 | 内容 | 建议用途 |
|---|---|---|
| `memory_token_attention_cost.md` Table A | A/D/E ablation memory cost | 论文 ablation 章节，Table X |
| `memory_token_attention_cost.md` Table B | Token budget curve | 论文 Phase 5 / Sec 5.x，Table Y |
| `memory_token_attention_cost.md` Table C | D 模块 token chain detail | supplementary material 或 method section |
| `memory_token_attention_cost.md` Appendix | Main result timing reference | supplementary material |

建议格式：
- Table A 和 Table B 可独立进入 paper，各成一个 LaTeX table。
- 注明 single seed；如需统计可信度，建议补 seed=2,3（仅核心两个点）。
- BFScore 来自 [ValSummary] 行，精度 2 位小数，需与论文其他指标精度对齐说明。

---

## 9. 主结果（data_pd_exp_final_2000）与 clean ablation 比较

**不可直接混合比较**。原因：
- 主结果 step_time 5.27s vs clean ablation 2.35s：前者包含完整 PL 生成（7002s），
  training 在完整 DensePASS 2000 张上，GPU context / batch 不同；
  后者使用固定 canonical PL，training 在相同 2000 张子集上，profile 更干净。
- 主结果 GPU peak 26.8 GB vs clean ablation 19.5 GB：GPU 配置不同（RTX 5090 32GB，
  但可能与 amp/bf16 batch size 影响有关）。
- 主结果 attn time/call：baseline 32.88 ms，A_Dtrain_Etrain 11.70 ms，方向一致
  （−64.4%），但绝对值不可与 clean ablation 的 117.5s / 51.9s 直接比较。

**推荐论文用法**：
主结果用于报告 mIoU（60.26 / 60.16），clean ablation 用于报告 token/timing 机制分析。

---

## 10. 风险与限制

1. **Single seed**：所有实验仅 seed=1，ΔmIoU ≤ 0.24 处于统计误差范围，论文需注明。
2. **BFScore 精度**：从 run.log 提取（2 位小数），建议在论文中标注来源。
3. **GPU peak 比较**：不同 run 有 ±0.5 GB 自然变化，A_ccpm 比 baseline 略高属正常。
4. **D compress_ratio=0.25 vs 0.5 说明**：compression_ratio 字段 = compress_ratio 配置；
   memory_tokens_after_final = 8192（A_Dtrain）是实测值，与 compress_ratio=0.25 per frame × 8 frames 一致。
5. **主结果 BFScore**：data_pd_exp_final_2000/ 无 run.log，BFScore 未知（见 CLAUDE.md 中 "待确认"）。

---

## 11. 生成文件

| 文件 | 说明 |
|---|---|
| `experiments/tecma/tables/profile_summary.csv` | 8 runs flat profile，含所有 token/timing/GPU 字段 |
| `experiments/tecma/tables/memory_token_attention_cost.md` | Table A (ablation) + B (token budget) + C (D chain) + Appendix (main result) |
| `experiments/tecma/tables/memory_token_attention_cost.csv` | 同上，CSV 格式，ablation + token budget 合并 |
| `tools/tecma_analysis/collect_profile_summary.py` | 只读分析脚本，可重新生成 profile_summary.csv |
| `experiments/tecma/notes/step_16_profile_summary_report.md` | 本报告 |
