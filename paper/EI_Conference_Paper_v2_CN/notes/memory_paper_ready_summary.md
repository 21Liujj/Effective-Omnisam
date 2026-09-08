# Memory-efficient Paper-ready Summary for OmniSAM Paper v1.0

## 1. Final Paper-safe Main Result

Full DensePASS protocol 主结果只使用正式 final_2000 口径：验证指标来自 full protocol `metrics.jsonl` 或原始验证日志；BFScore 来自 full protocol 原始验证日志 `[ValSummary] BFScore=...`；训练时间来自 full protocol `metrics.jsonl` / `metrics/train_timing.jsonl`。不要混入 fixed-PL clean ablation 的 BFScore 39.59 / 39.23。

| 方法 | 工程 run 名称 | 论文中建议名称 | mIoU | BFScore | mAcc | PixAcc | FWIoU | mDice | 总训练耗时 | 单 iter 耗时 | 证据来源 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| OmniSAM baseline | `base_last_20260425_232941` | OmniSAM baseline | 60.26 | 39.97 | 71.32 | 88.69 | 80.76 | 72.80 | 2633.36s = 43m53.36s | 5.2655s | mIoU/mAcc/PixAcc/FWIoU/mDice: `data_pd_exp_final_2000/base_last_20260425_232941/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` line 1 and raw log `logs/omnisam_final_2000/base_last_20260425_232941.log:L49`; BFScore: raw log L48; timing: `metrics.jsonl` line 1 and `metrics/train_timing.jsonl` line 1 |
| A_Dtrain_Etrain | `A_Dtrain_Etrain_20260425_232941` | Ours: MTC + SMR | 60.16 | 39.86 | 71.42 | 88.63 | 80.71 | 72.72 | 2043.32s = 34m03.32s | 4.0848s | mIoU/mAcc/PixAcc/FWIoU/mDice: `data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` line 1 and raw log `logs/omnisam_final_2000/A_Dtrain_Etrain_20260425_232941.log:L209`; BFScore: raw log L208; timing: `metrics.jsonl` line 1 and `metrics/train_timing.jsonl` line 1 |

说明：日志中的 mAcc/PixAcc/FWIoU/mDice 为两位小数，`metrics.jsonl` 中保存更高精度；论文 v1.0 主表建议统一保留两位小数。BFScore 未写入 full protocol `metrics.jsonl`，已由原始验证日志直接点验。

## 2. Final Efficiency Calculation

| 指标 | Baseline | A_Dtrain_Etrain | 变化 |
|---|---:|---:|---:|
| mIoU | 60.26 | 60.16 | -0.10 |
| BFScore | 39.97 | 39.86 | -0.11 |
| 总训练耗时（epoch_time / epoch_total_time） | 2633.36s = 43m53.36s | 2043.32s = 34m03.32s | -590.04s = -9m50.04s |
| 单 iter 耗时（step_time / iter_total_time） | 5.2655s | 4.0848s | -1.1807s |
| 总训练耗时下降比例 | 2633.36s | 2043.32s | 22.4% |
| 单 iter 耗时下降比例 | 5.2655s | 4.0848s | 22.4% |

计算公式：

```text
mIoU delta = ours - baseline = 60.16 - 60.26 = -0.10
BFScore delta = ours - baseline = 39.86 - 39.97 = -0.11

total time reduction = (baseline - ours) / baseline
                     = (2633.3628121614456 - 2043.3208468481898) / 2633.3628121614456
                     = 22.4%

step time reduction = (baseline - ours) / baseline
                    = (5.265505558468401 - 4.084782185226679) / 5.265505558468401
                    = 22.4%
```

注意：这里的训练耗时使用 full protocol 主结果表口径，即 `epoch_time/epoch_total_time` 与 `step_time/iter_total_time`。不要与 `epoch_train_time` / `iter_compute_time` 或 clean ablation/profile 的 step time 混写。

## 3. Memory Ablation Results for Paper

以下表格是 fixed-PL clean ablation / profile 口径，不是 full DensePASS protocol 主表口径。BFScore 39.59 / 39.23 只属于 clean ablation，不能写进 full protocol 主结果表。

| 配置 | A/CCPM | MTC/D | SMR/E | Memory tokens | mIoU | BFScore | Memory attention time | Step time | 口径 | 是否可进论文消融表 |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| `baseline_omnisam` | - | - | - | 32768 | 60.26 | 39.59 | 117.48s | 2.3505s | fixed-PL clean ablation / profile | 是，但必须标明 clean ablation 口径 |
| `A_ccpm` | ✓ | - | - | 32768 | 60.26 | 39.59 | 117.56s | 2.4089s | fixed-PL clean ablation / profile；token budget 100% reference | 是 |
| `A_Dtrain` | ✓ | ✓ | - | 8192 | 60.12 | 39.23 | 61.00s | 2.2938s | fixed-PL clean ablation / profile | 是 |
| `A_Etrain` | ✓ | - | ✓ | 16384 | 60.09 | 39.23 | 79.69s | 2.3798s | fixed-PL clean ablation / profile | 是 |
| `Dtrain_Etrain` | - | ✓ | ✓ | 4096 | 60.02 | 39.23 | 52.38s | 2.3237s | fixed-PL clean ablation / profile | 是 |
| `A_Dtrain_Etrain` | ✓ | ✓ | ✓ | 4096 | 60.02 | 39.23 | 51.92s | 2.3153s | fixed-PL clean ablation / profile | 是 |

主要证据来源：`memory_experiment_facts.md` 中对 `experiments/tecma/tables/clean_ablation_results.csv`、`experiments/tecma/tables/memory_token_attention_cost.csv`、`experiments/tecma/tables/paper_ablation_table_draft.md`、clean ablation run logs 的整理。

## 4. Token Budget Trade-off for Paper

以下 token budget 表同样属于 fixed-PL clean ablation / profile 相关口径。100%、75%、50%、25% 是 E-only token budget curve；final 12.5% 是 D+E combined endpoint，不是 E-only 曲线的自然延伸。

| 配置 | Token ratio | Memory tokens | mIoU | BFScore | Step time | Memory attention time | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| 100% / `A_ccpm` | 1.000 | 32768 | 60.26 | 39.59 | 2.4089s | 117.56s | E-only curve reference；A 不改变 memory token 数 |
| 75% / `A_Etrain_keep075` | 0.750 | 24576 | 60.12 | 39.28 | 2.4614s | 98.62s | attention time 下降，但 step time 略高，说明 sparse retrieval overhead 未完全摊销 |
| 50% / `A_Etrain` | 0.500 | 16384 | 60.09 | 39.23 | 2.3798s | 79.69s | E-only keep=0.50，memory attention time 继续下降 |
| 25% / `A_Etrain_keep025` | 0.250 | 8192 | 60.10 | 39.29 | 2.2941s | 60.54s | E-only keep=0.25，attention time 进一步下降，mIoU 仍约 60 |
| final 12.5% / `A_Dtrain_Etrain` | 0.125 | 4096 | 60.02 | 39.23 | 2.3153s | 51.92s | D+E combined endpoint；8x token reduction；不是 E-only 曲线自然延伸 |

论文中可以声称：memory attention time 随 effective memory tokens 减少整体下降。不要声称 total step time 严格单调下降，因为 75% 点 step time 高于 100% reference。

## 5. Recommended LaTeX Tables

### Table A: Main Results

```latex
\begin{table}[t]
\centering
\caption{Full DensePASS protocol results on Cityscapes$\rightarrow$DensePASS. BFScore is extracted from the original validation logs because it is not stored in \texttt{metrics.jsonl}.}
\label{tab:main_results_full_protocol}
\begin{tabular}{lcccccc}
\toprule
Method & mIoU $\uparrow$ & BFScore $\uparrow$ & Train Time $\downarrow$ & Iter Time $\downarrow$ & Memory Tokens & Note \\
\midrule
OmniSAM baseline & 60.26 & 39.97 & 43m53s & 5.27s & 32768 & full protocol \\
Ours (MTC+SMR) & 60.16 & 39.86 & 34m03s & 4.09s & 4096 & full protocol \\
\midrule
Change & -0.10 & -0.11 & -22.4\% & -22.4\% & 8$\times$ reduction & single seed \\
\bottomrule
\end{tabular}
\end{table}
```

### Table B: Efficiency Analysis

```latex
\begin{table}[t]
\centering
\caption{Efficiency analysis under the full DensePASS protocol. Time reduction is computed as $(\mathrm{baseline}-\mathrm{ours})/\mathrm{baseline}$.}
\label{tab:efficiency_analysis_full_protocol}
\begin{tabular}{lccc}
\toprule
Metric & Baseline & Ours (MTC+SMR) & Change \\
\midrule
mIoU & 60.26 & 60.16 & -0.10 \\
BFScore & 39.97 & 39.86 & -0.11 \\
Total train time & 2633.36s (43m53s) & 2043.32s (34m03s) & -22.4\% \\
Per-iteration time & 5.2655s & 4.0848s & -22.4\% \\
Memory tokens & 32768 & 4096 & 8$\times$ reduction \\
\bottomrule
\end{tabular}
\end{table}
```

### Table C: Memory Ablation

```latex
\begin{table}[t]
\centering
\caption{Memory ablation under the fixed-pseudo-label clean ablation/profile setting. These numbers are not mixed with the full-protocol main results.}
\label{tab:memory_ablation_clean}
\begin{tabular}{lccrrrr}
\toprule
Config & MTC & SMR & Memory Tokens & mIoU $\uparrow$ & BFScore $\uparrow$ & Mem. Attn. Time $\downarrow$ \\
\midrule
baseline & -- & -- & 32768 & 60.26 & 39.59 & 117.48s \\
CCPM & -- & -- & 32768 & 60.26 & 39.59 & 117.56s \\
CCPM + MTC & \checkmark & -- & 8192 & 60.12 & 39.23 & 61.00s \\
CCPM + SMR & -- & \checkmark & 16384 & 60.09 & 39.23 & 79.69s \\
MTC + SMR & \checkmark & \checkmark & 4096 & 60.02 & 39.23 & 52.38s \\
CCPM + MTC + SMR & \checkmark & \checkmark & 4096 & 60.02 & 39.23 & 51.92s \\
\bottomrule
\end{tabular}
\end{table}
```

## 6. Paper-ready Claims

1. 在 Cityscapes$\rightarrow$DensePASS full protocol、single seed 设置下，memory-efficient 配置在 mIoU 仅下降 0.10、BFScore 下降 0.11 的情况下，将总训练耗时从 43m53s 降至 34m03s，下降约 22.4%。
2. 在相同 full protocol 设置下，memory-efficient 配置将单 iter 耗时从 5.2655s 降至 4.0848s，说明 memory token 约简能够带来可观的训练效率收益。
3. fixed-PL clean ablation/profile 结果表明，将 effective memory tokens 从 32768 降至 4096 后，memory attention time 从约 117.5s 降至约 51.9s，同时 mIoU 仍保持在约 60 附近，说明 OmniSAM memory token 中存在一定冗余。
4. Token budget 分析显示，随着 E-only keep ratio 从 100% 降至 25%，memory attention time 整体下降；但 total step time 并非严格单调，因此论文中应将 memory attention time 作为 token-efficient 机制的直接证据。
5. 主结果表与消融表采用不同实验口径：主结果表对应 full DensePASS protocol，而 memory ablation / token budget 表对应 fixed-PL clean ablation/profile。论文中必须分开报告，避免混用 BFScore 和 step time。

## 7. Unsafe Claims

- 不能声称多 seed 稳定性；当前 memory 主结果和相关消融均为 single seed 证据。
- 不能声称该方法已在所有数据集上有效；本汇总只覆盖 Cityscapes$\rightarrow$DensePASS 相关结果。
- 不能把 clean ablation/profile 的 BFScore 39.59 / 39.23 写进 full protocol 主结果表；full protocol BFScore 是 39.97 / 39.86。
- 不能声称 step time 随 token budget 严格单调下降；75% token budget 的 step time 高于 100% reference。
- 不能声称无精度损失；full protocol 下 mIoU 下降 0.10、BFScore 下降 0.11，clean ablation 下也存在小幅下降。
- 不能声称显著提升 mIoU；当前结果是基本保持精度并降低训练开销，而不是提升精度。
- 不能把 full protocol 的 5.27s / 4.09s 与 clean ablation/profile 的 2.35s / 2.31s 混为同一 timing 口径。
- 不能说 BFScore 来自 full protocol `metrics.jsonl`；BFScore 来自原始验证日志 `[ValSummary] BFScore=...`。

## 8. Suggested Naming in Paper

| 工程名称 | 论文建议名称 | 是否适合正文直接出现 | 备注 |
|---|---|---|---|
| `A_ccpm` | Confidence-Controlled Pseudo Mask (CCPM) | 可在方法/消融中出现，不建议大量使用工程名 | A 是工程缩写；正文优先使用 CCPM |
| `Dtrain` / D module | Memory Token Compression (MTC) | 是，建议正文使用 MTC | 对应 train-stage memory compression；可在脚注说明工程名 D |
| `Etrain` / E module | Sparse Memory Retrieval (SMR) | 是，建议正文使用 SMR | 对应 train-stage sparse top-k memory retrieval；可在脚注说明工程名 E |
| `A_Dtrain_Etrain` | Ours (MTC + SMR) 或 CCPM + MTC + SMR | 工程名不建议正文大量出现 | 主结果表可写 Ours (MTC+SMR)；若强调包含 A，则在实验设置或脚注说明 |
| `fixed-PL clean ablation` | Fixed-pseudo-label clean ablation | 是 | 用于消融/profile 表，必须与 full protocol 主结果区分 |
| `full DensePASS protocol` | Full Cityscapes$\rightarrow$DensePASS protocol | 是 | 用于主结果表；包含 2000 PL、1 epoch、DensePASS val 100 的正式主实验口径 |
