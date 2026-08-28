# step_01_existing_results_inventory.md
# Phase 0 / Task 0-2 — 已有实验结果盘点

**Run Level**: L0 scan-only  
**日期**: 2026-06-23  
**执行人**: Claude Code  
**修改代码**: 否  
**跑训练**: 否

---

## 1. 已扫描目录

| 目录 | 内容 | 扫描状态 |
|---|---|---|
| `data_pd_exp_final_2000/` | 2026-04 原始完整实验（2000张，seed=1，epoch=1）| 已扫描 |
| `experiments/tecma/runs/` | 2026-06 clean ablation（fixed PL，seed=1，epoch=1）| 已扫描 |
| `experiments/tecma/runs/token_budget/` | 2026-06 E 模块 keep_ratio 扫参 | 已扫描 |
| `experiments/tecma/tables/` | 已整理的消融结果表、per-class IoU、BFScore 审计 | 已扫描 |
| `logs/omnisam_final_2000/` | 原始完整实验训练日志 | 已扫描 |
| `logs/abcd_summary.md` | 旧版多模态 ABCD 实验汇总 | 已扫描 |

**重要说明**：
- `data_pd_exp_final_2000/` 实验（April 2026）= 原始完整训练，包含 PL 生成（2000张，DensePASS，step_time=4-5s/it）
- `experiments/tecma/runs/` 实验（June 2026）= clean ablation，所有 run 均 **reuse** `base_last_20260425_232941/` 的 PL 文件（已确认 `run.log` 中有 `Reuse fixed pseudo-label dir` 标记），无 PL 生成开销（step_time≈2.3s/it）

这是两套不同设定的实验，**不可混用效率数字**。

---

## 2. 完整实验结果表

### 2A. 原始完整实验（data_pd_exp_final_2000）— 用于论文主结果

| Run | Config | Dataset | seed | Complete? | mIoU | mAcc | BFScore | PixAcc | FWIoU | mDice | Train Time (E2E) | Step Time | Peak GPU | Mem Tokens | PL Time | Log Path | Eval Path | Reusable? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_last | baseline_omnisam | CS→DP 2000img | 1 | **Yes** | **60.26** | 71.32 | **39.97** | 88.69 | 80.76 | 72.80 | 2633s(train)+7002s(PL) | 5.27s/it | 26.82GB | 32768 | 7002s | `logs/omnisam_final_2000/base_last_20260425_232941.log` | `data_pd_exp_final_2000/base_last_20260425_232941/checkpoints/…/metrics.jsonl` | **Yes (R0)** |
| A_Dtrain_Etrain | A_Dtrain_Etrain | CS→DP 2000img | 1 | **Yes** | **60.16** | 71.42 | **39.86** | 88.63 | 80.71 | 72.72 | 2043s(train)+7082s(PL) | 4.08s/it | 25.34GB | 4096 | 7082s | `logs/omnisam_final_2000/A_Dtrain_Etrain_20260425_232941.log` | `data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/checkpoints/…/metrics.jsonl` | **Yes (memory-eff baseline)** |
| base_first | baseline_omnisam | CS→DP 2000img | 1 | Yes | 60.19 | 72.55 | 39.38 | 88.56 | 80.66 | 72.82 | N/A | N/A | N/A | 32768 | N/A | `logs/omnisam_final_2000/base_first_20260425_232941.log` | N/A (no metrics.jsonl) | No (warm-up run，仅做热身，非正式结果) |

> **BFScore 来源**：`base_last` 和 `A_Dtrain_Etrain` 均已通过 `step_09_bfscore_evidence_audit.md` 验证（来自训练日志 `[ValSummary] BFScore=...`，非 metrics.jsonl 中的 null 字段）。

### 2B. Clean Ablation（experiments/tecma）— 用于消融表（TeCMA/memory-efficient 论文）

所有 run 均 reuse `data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1` 的 PL 文件。

| Run | Config | seed | Complete? | mIoU | mAcc | BFScore | PixAcc | FWIoU | mDice | Step Time | Iter Compute | Mem Attn (s) | Peak GPU | Mem Tokens | Log Path | Reusable? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_omnisam_20260616 | baseline_omnisam | 1 | **Yes** | **60.26** | 71.71 | **39.59** | 88.63 | 80.73 | 72.84 | 2.35s/it | 1.56s/it | 117.5s | 19.54GB | 32768 | `experiments/tecma/runs/baseline_omnisam_seed1_20260616_102834/run.log` | **Yes (R0，clean设定)** |
| A_ccpm_20260616 | A_ccpm (pseudo_conf_mask) | 1 | **Yes** | **60.26** | 71.72 | **39.59** | 88.64 | 80.73 | 72.85 | 2.41s/it | 1.58s/it | 117.6s | 19.98GB | 32768 | `experiments/tecma/runs/A_ccpm_seed1_20260616_104911/run.log` | **Yes (R1 等价，clean设定)** |
| A_Dtrain_20260616 | A_Dtrain | 1 | **Yes** | 60.12 | 71.56 | 39.23 | 88.58 | 80.65 | 72.71 | 2.29s/it | 1.45s/it | 61.0s | 18.99GB | 8192 (D:4096→1024) | `experiments/tecma/runs/A_Dtrain_seed1_20260616_111019/run.log` | Conditional |
| A_Etrain_20260616 | A_Etrain | 1 | **Yes** | 60.09 | 71.53 | 39.23 | 88.53 | 80.57 | 72.69 | 2.38s/it | 1.54s/it | 79.7s | 19.28GB | 16384 (E:32768→16384) | `experiments/tecma/runs/A_Etrain_seed1_20260616_113023/run.log` | Conditional |
| Dtrain_Etrain_20260616 | Dtrain_Etrain | 1 | **Yes** | 60.02 | 71.52 | 39.23 | 88.57 | 80.65 | 72.59 | 2.32s/it | 1.47s/it | 52.4s | 18.34GB | 4096 (D+E) | `experiments/tecma/runs/Dtrain_Etrain_seed1_20260616_115113/run.log` | Conditional |
| A_Dtrain_Etrain_20260616 | A_Dtrain_Etrain | 1 | **Yes** | 60.02 | 71.53 | 39.23 | 88.57 | 80.66 | 72.60 | 2.32s/it | 1.46s/it | 51.9s | 18.78GB | 4096 (D+E) | `experiments/tecma/runs/A_Dtrain_Etrain_seed1_20260616_121135/run.log` | **Yes (memory-eff，clean设定)** |
| A_Dtrain_Etrain_PLon_20260616 | A_Dtrain_Etrain_PLon | 1 | Yes | 60.00 | 71.22 | 39.41 | 88.57 | 80.63 | 72.56 | 2.36s/it | 1.50s/it | 52.3s | 18.78GB | 4096 | `experiments/tecma/runs/A_Dtrain_Etrain_PLon_seed1_20260616_031742/run.log` | Conditional (含 PL 开关) |

### 2C. Token Budget 扫参（experiments/tecma/runs/token_budget）

| Run | Config | seed | Complete? | mIoU | mAcc | BFScore | Step Time | Mem Tokens | Peak GPU | Reusable? |
|---|---|---|---|---|---|---|---|---|---|---|
| A_Etrain_keep025_20260617 | A_Etrain_keep025 (E keep=0.25) | 1 | Yes | 60.10 | 71.62 | 39.29 | 2.29s/it | 8192 (E: 32768→8192) | 18.93GB | Conditional |
| A_Etrain_keep075_20260617 | A_Etrain_keep075 (E keep=0.75) | 1 | Yes | 60.12 | 71.62 | 39.28 | 2.46s/it | 24576 (E: 32768→24576) | 19.64GB | Conditional |

---

## 3. Per-class IoU 汇总

数据来源：各 `run.log` 中 `per-class IoU (0-100)` 段落。

| class | base_last (final) | baseline (clean) | A_ccpm (clean) | A_Dtrain (clean) | A_Etrain (clean) | Dtrain_Etrain (clean) | A_DtE (clean) |
|---|---:|---:|---:|---:|---:|---:|---:|
| road | 80.38 | 80.23 | 80.25 | 80.26 | 80.07 | 80.25 | 80.25 |
| sidewalk | 47.54 | 47.33 | 47.35 | 47.51 | 47.05 | 47.62 | 47.62 |
| building | 89.69 | 89.67 | 89.67 | 89.49 | 89.41 | 89.53 | 89.53 |
| wall | 43.59 | 43.77 | 43.78 | 43.75 | 43.88 | 44.38 | 44.40 |
| fence | 49.02 | 48.92 | 48.92 | 48.83 | 48.43 | 48.12 | 48.15 |
| pole | 42.90 | 43.64 | 43.64 | 43.37 | 43.55 | 43.36 | 43.36 |
| traffic light | 28.00 | 27.88 | 27.88 | 27.04 | 27.74 | 26.21 | 26.23 |
| traffic sign | 31.40 | 32.52 | 32.52 | 32.53 | 32.11 | 32.24 | 32.23 |
| vegetation | 78.63 | 78.77 | 78.77 | 78.72 | 78.72 | 78.63 | 78.63 |
| terrain | 45.44 | 45.15 | 45.15 | 44.96 | 44.68 | 44.76 | 44.76 |
| sky | 94.57 | 94.61 | 94.61 | 94.62 | 94.59 | 94.62 | 94.62 |
| person | 68.73 | 67.50 | 67.50 | 67.56 | 67.65 | 67.42 | 67.42 |
| car | 83.44 | 83.33 | 83.34 | 82.88 | 83.35 | 83.08 | 83.08 |

> **注意**：`base_last (final)` 和 `baseline (clean)` 两个 baseline run 的 per-class IoU 有微小差异（如 fence: 49.02 vs 48.92），因 PL 设定不完全相同（final=自己生成 PL，clean=reuse固定PL）。

---

## 4. BFScore 来源与可信度

| Run | BFScore | 来源 | 可信度 |
|---|---|---|---|
| base_last (final_2000) | **39.97** | `logs/omnisam_final_2000/base_last.log` `[ValSummary]` | **已核实 (step_09_audit)** |
| A_Dtrain_Etrain (final_2000) | **39.86** | `logs/omnisam_final_2000/A_Dtrain_Etrain.log` `[ValSummary]` | **已核实 (step_09_audit)** |
| baseline_omnisam (clean) | 39.59 | `experiments/tecma/runs/baseline.../run.log` `[ValSummary]` | 已扫描，可信 |
| A_ccpm (clean) | 39.59 | 同上 | 已扫描，可信 |
| A_Dtrain/A_Etrain/Dtrain_Etrain/A_DtE (clean) | 39.23 | 同上 | 已扫描，可信 |

**BFScore 差异说明**：
- final_2000 baseline (39.97) vs clean baseline (39.59) 差 0.38
- 两者 mIoU 相同（60.26），差异来自 PL 设定不同（final=自生成PL，clean=reuse固定PL）以及 optimizer warmup 状态差异
- **论文主结果引用 final_2000 的 39.97**（与 step_09_audit 一致）；消融表引用 clean 设定的 39.59

---

## 5. Reusable? 判断

| Run | Reusable? | 依据 |
|---|---|---|
| base_last (final_2000) | **Yes — R0 论文主结果** | mIoU=60.26, BFScore=39.97, per-class IoU 完整，checkpoint 存在，seed=1，有训练日志可溯源 |
| A_Dtrain_Etrain (final_2000) | **Yes — memory-efficient 主结果** | mIoU=60.16, BFScore=39.86，效率数字经 step_09_audit 验证 |
| baseline_omnisam (clean) | **Yes — R0 消融参照** | 与 final_2000 baseline 用相同 PL，mIoU 完全一致（60.26），BFScore 39.59，clean 设定内可信 |
| A_ccpm (clean) | **Yes — R1 等价参照** | A_ccpm = pseudo_conf_mask 插件 = R1 的等价实现，mIoU=60.26 与 baseline 持平，BFScore=39.59 |
| A_Dtrain (clean) | Conditional | 属于 TeCMA 消融，不属于 PL refiner 线，作为效率参照可复用 |
| A_Etrain (clean) | Conditional | 同上 |
| Dtrain_Etrain (clean) | Conditional | 同上 |
| A_Dtrain_Etrain (clean) | **Yes — R0 (clean) + memory-eff 参照** | mIoU=60.02 with D+E，是 clean 设定下的 memory-efficient baseline |
| A_Dtrain_Etrain_PLon | Conditional | PL 开关 on，mIoU=60.00，可作为 D/E 在 PL 阶段的参照 |
| token_budget runs | Conditional | 效率扫参，不进 R0-R7 核心消融 |
| base_first (final_2000) | **No** | 热身 run，正式结果以 base_last 为准 |

---

## 6. R0-R7 对照现有结果

| R配置 | 是否有已有结果 | 对应现有 Run | mIoU | BFScore | Notes |
|---|---|---|---|---|---|
| **R0 baseline** | **Yes** | base_last (final_2000) | 60.26 | 39.97 | **可直接复用为论文 R0** |
| **R1 conf_mask** | **Yes (等价)** | A_ccpm (clean) | 60.26 | 39.59 | A_ccpm = pseudo_conf_mask = R1 等价实现；但用的 clean 设定 PL，非 R0 相同 full 设定 |
| R2 class_balanced | No | — | — | — | 未实现 |
| R3 plq_basic | No | — | — | — | 未实现 |
| R4 proto_gate | No | — | — | — | 未实现 |
| R5 rbpc | No | — | — | — | 未实现 |
| R6 memory_plq | No | — | — | — | 未实现 |
| **R7 full** | **Partial** | A_Dtrain_Etrain (final_2000) | 60.16 | 39.86 | memory-efficient baseline 完整，但 PseudoLabelRefiner 部分（class balance/PLQ/prototype）尚未实现 |

---

## 7. 命名与设定一致性风险

| 风险项 | 描述 | 严重程度 |
|---|---|---|
| **BFScore 两套数字** | final_2000 baseline=39.97，clean baseline=39.59，同 mIoU 不同 BFScore，混用会导致论文数字矛盾 | **高** — 必须在论文中明确区分两套实验设定 |
| **step_time 两套口径** | final_2000 step_time≈5.3s（含 mem 开销，原始 PL），clean ablation step_time≈2.3s（reuse PL，batch_size=2，无 PL 生成）| **高** — 效率数字不能混用 |
| **R1 设定不完全对等** | A_ccpm 用 clean PL（固定 reuse），而 R2-R7 将用同一批 PL，一致性可接受；但若 R1 重跑需保持相同 PL 设定 | 中 |
| **base_first 混淆风险** | base_first (mIoU=60.19) 和 base_last (mIoU=60.26) 名字相似，但一个是热身一个是正式结果 | 低（已区分） |
| **metrics.jsonl BFScore=null** | final_2000 所有 run 的 metrics.jsonl 中 BFScore 字段均为 null；BFScore 只在训练日志 [ValSummary] 中，需从日志手动提取 | 中 |

---

## 8. 下一步建议

### 哪些结果可以直接复用？

1. **R0 论文主结果**：`base_last_20260425_232941`（mIoU=60.26, BFScore=39.97）**可直接复用**，checkpoint 存在，有完整日志和 per-class IoU 数据。
2. **R1 消融参照**：`A_ccpm_seed1_20260616_104911`（mIoU=60.26, BFScore=39.59）**可作为 R1 等价结果在 clean 设定下复用**，但若需要与 R0 final_2000 放在同一论文表格，需补跑 R1 full 实验（与 R0 same PL setting）。
3. **Memory-efficient 主结果**：`A_Dtrain_Etrain_20260425_232941`（mIoU=60.16, BFScore=39.86）**可直接复用**。
4. **Clean ablation 全套**（baseline/A_ccpm/A_Dtrain/A_Etrain/Dtrain_Etrain/A_DtE）：可作为 TeCMA memory-efficiency 消融表复用，但不可与 final_2000 的效率数字混用。

### 哪些结果必须补跑？

- **R2~R6 全部需要补跑**（尚未实现 PseudoLabelRefiner）
- **R7 full 需要补跑**（R7 = A_Dtrain_Etrain + PseudoLabelRefiner，目前只有 memory-efficient 部分）
- **R1 full（final_2000 设定）可选补跑**：若需要与 base_last (final_2000) 在同一表格直接对比，需补一个完整 PL 生成 + pseudo_conf_mask 的 full run

### 是否可以进入 Phase 1 / Task 1-1 identity refiner？

**可以立即进入**。条件已满足：
- R0 baseline 结果已确认（base_last, mIoU=60.26）
- R1 等价结果已确认（A_ccpm, mIoU=60.26）
- PL pipeline 入口已确认（Task 0-1）
- identity refiner 插入点明确（strategies.py）

Task 1-1 只做骨架（mode=off/identity），不跑完整训练，可安全推进。

### 是否存在日志缺失、评估缺失或命名混乱风险？

1. **BFScore 未写入 metrics.jsonl**：所有实验的 metrics.jsonl 中 BFScore=null，需从训练日志手动提取。已有 audit 脚本和记录，但每次新实验都需要记住这个问题。**建议 Phase 1 开始时在 strategies.py/trainer.py 修复这个 bug**（低优先级，不阻塞）。
2. **两套 baseline 数字**：final_2000（39.97）和 clean（39.59）必须严格区分。PL refiner 消融建议统一用 **final_2000 设定**，与 base_last 直接对比，避免混乱。
3. **base_first 命名风险**：虽然已知 base_first 是热身 run，但命名容易与 base_last 混淆。不建议在论文中引用 base_first。
4. **sanity_runs/ 无 run_summary.json**：sanity_runs 下两个目录为空，无需关注。

---

## 9. 本次任务总结

- **做了什么**：L0 扫描，读取 15 个 run_summary.json、3 个 metrics.jsonl、6 个 run.log 的 per-class IoU 和 BFScore、2 个已整理 table 文件
- **修改了哪些文件**：无
- **生成了哪些输出**：本文件
- **是否跑训练**：否
- **是否改动 pseudo label**：否
- **是否改动 loss**：否
- **当前风险**：BFScore 两套数字需在论文写作时明确区分
- **下一步建议**：进入 Phase 1 / Task 1-1（identity PseudoLabelRefiner 骨架）
