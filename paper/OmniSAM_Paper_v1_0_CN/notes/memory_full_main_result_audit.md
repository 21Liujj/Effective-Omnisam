# Full DensePASS Main Result Audit for Paper v1.0

## 1. Audit Scope

本次只检查两个 full DensePASS protocol 主结果目录，以及 `step_09_main_results_draft_report.md`。没有运行训练、验证或大规模脚本；没有跳出两个主结果目录做全仓库搜索。

| 序号 | 路径 | 是否存在 | 读取目的 | 备注 |
|---|---|---|---|---|
| 1 | `data_pd_exp_final_2000/base_last_20260425_232941/` | 是 | 审计 OmniSAM baseline full protocol 原始指标、timing、PL 使用记录和 checkpoint 结构 | 目录内未发现 `run.log` / `train.log` / `summary.json` / `run_summary.json` |
| 2 | `data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/` | 是 | 审计 A_Dtrain_Etrain full protocol 原始指标、timing、PL 使用记录和 checkpoint 结构 | 目录内未发现 `run.log` / `train.log` / `summary.json` / `run_summary.json` |
| 3 | `experiments/tecma/notes/step_09_main_results_draft_report.md` | 是 | 核对 full main result 来源说明、single seed 限制和相对变化摘要 | 该报告说明 BFScore 已审计，但本次未读取额外 BFScore audit 文件 |
| 4 | `data_pd_exp_final_2000/base_last_20260425_232941/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | 是 | 提取 baseline full protocol 验证指标与 epoch timing 字段 | 1 行 JSONL，含 mIoU/mAcc/PixAcc/FWIoU/mDice，不含 BFScore 字段 |
| 5 | `data_pd_exp_final_2000/base_last_20260425_232941/checkpoints/Cityscapes_13cls/sam2_s/metrics/train_timing.jsonl` | 是 | 提取 baseline full protocol timing、memory token 和 memory attention 字段 | 1 行 JSONL |
| 6 | `data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | 是 | 提取 A_Dtrain_Etrain full protocol 验证指标与 epoch timing 字段 | 1 行 JSONL，含 mIoU/mAcc/PixAcc/FWIoU/mDice，不含 BFScore 字段 |
| 7 | `data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/checkpoints/Cityscapes_13cls/sam2_s/metrics/train_timing.jsonl` | 是 | 提取 A_Dtrain_Etrain full protocol timing、memory token 和 memory attention 字段 | 1 行 JSONL |
| 8 | `*/DensePASS/sam2_s/Cityscapes_13cls/used_path.txt` | 是 | 确认 DensePASS pseudo-label 使用路径记录存在 | 两个目录均存在，大小均为 30940 bytes；本任务未逐行展开 |
| 9 | `*/DensePASS/sam2_s/Cityscapes_13cls/unused_path.txt` | 是 | 确认 unused path 记录存在 | 两个目录均存在，大小均为 0 bytes |

## 2. Directory Structure Summary

### 2.1 Baseline directory

`data_pd_exp_final_2000/base_last_20260425_232941/` 内部关键结构：

| 类型 | 路径 | 审计结果 |
|---|---|---|
| DensePASS PL 目录 | `DensePASS/sam2_s/Cityscapes_13cls/` | 存在 |
| used path | `DensePASS/sam2_s/Cityscapes_13cls/used_path.txt` | 存在，30940 bytes |
| unused path | `DensePASS/sam2_s/Cityscapes_13cls/unused_path.txt` | 存在，0 bytes |
| checkpoint 目录 | `checkpoints/Cityscapes_13cls/sam2_s/` | 存在 |
| metrics | `checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | 存在，1840 bytes |
| timing | `checkpoints/Cityscapes_13cls/sam2_s/metrics/train_timing.jsonl` | 存在，831 bytes |
| logs | `run.log` / `train.log` | 未找到 |
| configs | `config*.yaml` / `memory_config*.yaml` | 未找到 |
| summary | `summary.json` / `run_summary.json` / `profile_summary.json` / `val_summary.json` | 未找到 |
| result tables | `results.csv` / `results.md` | 未找到 |

### 2.2 A_Dtrain_Etrain directory

`data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/` 内部关键结构：

| 类型 | 路径 | 审计结果 |
|---|---|---|
| DensePASS PL 目录 | `DensePASS/sam2_s/Cityscapes_13cls/` | 存在 |
| used path | `DensePASS/sam2_s/Cityscapes_13cls/used_path.txt` | 存在，30940 bytes |
| unused path | `DensePASS/sam2_s/Cityscapes_13cls/unused_path.txt` | 存在，0 bytes |
| checkpoint 目录 | `checkpoints/Cityscapes_13cls/sam2_s/` | 存在 |
| metrics | `checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | 存在，1843 bytes |
| timing | `checkpoints/Cityscapes_13cls/sam2_s/metrics/train_timing.jsonl` | 存在，828 bytes |
| logs | `run.log` / `train.log` | 未找到 |
| configs | `config*.yaml` / `memory_config*.yaml` | 未找到 |
| summary | `summary.json` / `run_summary.json` / `profile_summary.json` / `val_summary.json` | 未找到 |
| result tables | `results.csv` / `results.md` | 未找到 |

## 3. Metric Evidence

| 方法 | mIoU | BFScore | mAcc | PixAcc | FWIoU | mDice | 来源文件 | 来源行/字段 | 是否可作为论文主表数字 |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| OmniSAM baseline | 60.26 | 未确认 | 71.3172056643454 | 88.69062811268108 | 80.76203630977453 | 72.79959709898371 | `data_pd_exp_final_2000/base_last_20260425_232941/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | line 1: `val_mIoU`, `val_mAcc`, `val_PixAcc`, `val_FWIoU`, `val_mDice`; `BFScore` 字段不存在 | mIoU/mAcc/PixAcc/FWIoU/mDice 可作为主表数字；BFScore 不可直接作为主表数字 |
| A_Dtrain_Etrain | 60.16 | 未确认 | 71.424854600759 | 88.62621587484094 | 80.70615628928837 | 72.71661817634441 | `data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | line 1: `val_mIoU`, `val_mAcc`, `val_PixAcc`, `val_FWIoU`, `val_mDice`; `BFScore` 字段不存在 | mIoU/mAcc/PixAcc/FWIoU/mDice 可作为主表数字；BFScore 不可直接作为主表数字 |

重点数字审计：

| 待确认数字 | 审计结论 | 证据说明 |
|---|---|---|
| baseline mIoU = 60.26 | 已确认 | `base_last.../metrics.jsonl` line 1: `val_mIoU: 60.26` |
| baseline BFScore = 39.97 | 未确认 | 两个 full result 目录内未发现 `run.log`，`metrics.jsonl` 不含 `BFScore` 字段；`step_09` 只说明 BFScore 已 `verified_from_train_log`，但本次指定原始目录内无对应 train log |
| A_Dtrain_Etrain mIoU = 60.16 | 已确认 | `A_Dtrain_Etrain.../metrics.jsonl` line 1: `val_mIoU: 60.16` |
| A_Dtrain_Etrain BFScore = 39.86 | 未确认 | 两个 full result 目录内未发现 `run.log`，`metrics.jsonl` 不含 `BFScore` 字段；`step_09` 只给出 BFScore 变化 `-0.11`，未给出绝对值 |

推导值说明：

- `step_09_main_results_draft_report.md` line 61-63 给出 BFScore 变化为 `-0.11`。
- 如果外部证据确认 baseline BFScore 为 39.97，则可推导 A_Dtrain_Etrain BFScore 为 39.86；但这不是本次从两个 full result 原始目录中直接确认的数字。
- 本审计不把推导值写成原始确认值。

## 4. Timing Evidence

| 方法 | 总训练耗时 | 单 iter 耗时 | iter 数 | timing 口径 | 来源文件 | 来源字段/行 | 是否可作为论文效率主表数字 |
|---|---:|---:|---:|---|---|---|---|
| OmniSAM baseline | `epoch_time/epoch_total_time = 2633.3628121614456s = 43m53.36s`; 另有 `epoch_train_time = 2100.0507096275687s = 35m00.05s` | `step_time = iter_total_time = 5.265505558468401s`; `iter_compute_time = 4.200101419255137s` | 500（由 `epoch_time / step_time ≈ 500.1` 与 full protocol 2000 PL / batch size 4 口径一致；`step_09` line 26 说明 2000 PL） | `epoch_time/epoch_total_time` 看起来包含 train loop 之外的 epoch 级耗时；`epoch_train_time` 更接近 train compute 累计；`step_time` 与 `epoch_time / 500` 对齐，不能简单等同纯 train compute | `base_last.../metrics.jsonl`; `base_last.../metrics/train_timing.jsonl` | 两文件 line 1: `step_time`, `epoch_time`, `epoch_train_time`, `epoch_total_time`, `iter_compute_time` | 可作为效率主表数字，但 caption 必须说明使用 `epoch_time/epoch_total_time` 与 `step_time/iter_total_time` 口径；若写纯训练 loop，应改用 `epoch_train_time` / `iter_compute_time` |
| A_Dtrain_Etrain | `epoch_time/epoch_total_time = 2043.3208468481898s = 34m03.32s`; 另有 `epoch_train_time = 1496.7992524802685s = 24m56.80s` | `step_time = iter_total_time = 4.084782185226679s`; `iter_compute_time = 2.993598504960537s` | 500（由 `epoch_time / step_time ≈ 500.23` 与 full protocol 2000 PL / batch size 4 口径一致；`step_09` line 26 说明 2000 PL） | `epoch_time/epoch_total_time` 看起来包含 train loop 之外的 epoch 级耗时；`epoch_train_time` 更接近 train compute 累计；`step_time` 与 `epoch_time / 500` 对齐，不能简单等同纯 train compute | `A_Dtrain_Etrain.../metrics.jsonl`; `A_Dtrain_Etrain.../metrics/train_timing.jsonl` | 两文件 line 1: `step_time`, `epoch_time`, `epoch_train_time`, `epoch_total_time`, `iter_compute_time` | 可作为效率主表数字，但 caption 必须说明使用 `epoch_time/epoch_total_time` 与 `step_time/iter_total_time` 口径；若写纯训练 loop，应改用 `epoch_train_time` / `iter_compute_time` |

重点数字审计：

| 待确认数字 | 审计结论 | 证据说明 |
|---|---|---|
| baseline 总训练耗时 = 43m53s | 已确认为 `epoch_time/epoch_total_time` 口径 | `base_last.../metrics.jsonl` line 1 和 `train_timing.jsonl` line 1: `epoch_time = epoch_total_time = 2633.3628121614456s = 43m53.36s` |
| A_Dtrain_Etrain 总训练耗时 = 34m03s | 已确认为 `epoch_time/epoch_total_time` 口径 | `A_Dtrain_Etrain.../metrics.jsonl` line 1 和 `train_timing.jsonl` line 1: `epoch_time = epoch_total_time = 2043.3208468481898s = 34m03.32s` |
| baseline 单 iter 耗时 = 5.27s | 已确认为 `step_time/iter_total_time` 口径 | `base_last.../metrics.jsonl` line 1 和 `train_timing.jsonl` line 1: `step_time = iter_total_time = 5.265505558468401s` |
| A_Dtrain_Etrain 单 iter 耗时 = 4.09s | 已确认为 `step_time/iter_total_time` 口径 | `A_Dtrain_Etrain.../metrics.jsonl` line 1 和 `train_timing.jsonl` line 1: `step_time = iter_total_time = 4.084782185226679s` |

额外 timing 字段，不建议混写：

| 方法 | `epoch_train_time` | `iter_compute_time` | 说明 |
|---|---:|---:|---|
| baseline | 2100.0507096275687s = 35m00.05s | 4.200101419255137s | 更接近 train compute 累计 / forward-backward-optim 口径 |
| A_Dtrain_Etrain | 1496.7992524802685s = 24m56.80s | 2.993598504960537s | 更接近 train compute 累计 / forward-backward-optim 口径 |


## 5. Efficiency Calculation

以下计算只使用第 3、4 节中已从 full result 原始文件直接确认的字段。

- 总训练耗时下降比例（`epoch_time/epoch_total_time` 口径）：

```text
reduction = (baseline - ours) / baseline
          = (2633.3628121614456 - 2043.3208468481898) / 2633.3628121614456
          = 0.22405 ≈ 22.4%
```

- 单 iter 耗时下降比例（`step_time/iter_total_time` 口径）：

```text
reduction = (baseline - ours) / baseline
          = (5.265505558468401 - 4.084782185226679) / 5.265505558468401
          = 0.22424 ≈ 22.4%
```

- mIoU 变化：

```text
delta = ours - baseline
      = 60.16 - 60.26
      = -0.10
```

- BFScore 变化：待确认。

说明：`step_09_main_results_draft_report.md` line 61-63 报告 BFScore 变化为 `-0.11`，但本次在两个 full result 原始目录中未找到 BFScore 绝对值，因此不在本节基于原始文件计算 BFScore delta。

可补充但需分开口径的 train-compute 计算：

```text
epoch_train_time reduction
= (2100.0507096275687 - 1496.7992524802685) / 2100.0507096275687
= 0.28725 ≈ 28.7%

iter_compute_time reduction
= (4.200101419255137 - 2.993598504960537) / 4.200101419255137
= 0.28725 ≈ 28.7%
```

这组 28.7% 是 `epoch_train_time/iter_compute_time` 口径，不应与主表的 43m53s→34m03s、5.27s→4.09s 口径混写。

## 6. Consistency Check Against 0-2 Facts

| 项目 | full protocol 数字 | clean ablation/profile 数字 | 是否同一口径 | 论文中如何区分 |
|---|---:|---:|---|---|
| baseline BFScore | 39.97 未在本次 full 原始目录直接确认；`step_09` 仅说明 BFScore 已审计并给出 delta | 39.59，来源 `experiments/tecma/runs/baseline_omnisam_seed1_20260616_102834/run.log` / clean ablation tables | 否 | full 主表 BFScore 需要补原始 train log 或 BFScore audit 证据；clean ablation BFScore 只能放消融/profile 表 |
| A_Dtrain_Etrain BFScore | 39.86 未在本次 full 原始目录直接确认；可由 39.97 + `-0.11` 推导但不是直接确认 | 39.23，来源 clean ablation A_Dtrain_Etrain run.log / tables | 否 | full 主表不能使用 39.23；若写 39.86，必须补直接证据或标注为来自 BFScore audit |
| full iter time vs profile step_time | baseline 5.2655s；A_Dtrain_Etrain 4.0848s，来源 full result `metrics.jsonl` / `train_timing.jsonl` | baseline 2.3505s；A_Dtrain_Etrain 2.3153s，来源 clean ablation/profile tables | 否 | full 主表使用 5.27/4.09；clean profile 表使用 2.35/2.31，并明确 fixed-PL clean ablation/profile 口径 |
| full mIoU vs clean ablation mIoU | baseline 60.26；A_Dtrain_Etrain 60.16，来源 full result `metrics.jsonl` | baseline 60.26；A_Dtrain_Etrain 60.02，来源 clean ablation/profile tables | 否 | full 主表写 60.26/60.16；memory 消融表写 60.26/60.02 |
| memory tokens | baseline 32768；A_Dtrain_Etrain 4096，来源 full result timing/metrics 字段 | baseline 32768；A_Dtrain_Etrain 4096，来源 clean profile tables | 数值一致，但实验口径不同 | 可作为机制一致性证据；主结果和消融表仍需分开 |
| memory attention time | baseline 263.0156s；A_Dtrain_Etrain 93.6298s，来源 full result `train_timing.jsonl` | baseline/profile reference 117.48/117.56s；A_Dtrain_Etrain 51.92s，来源 clean profile tables | 否 | full protocol 和 clean profile 的 attention time 不混用；如都写，需要标注数据规模和 run 口径 |

## 7. Paper-safe Main Table Recommendation

### 7.1 如果 full protocol 数字被确认

当前 full protocol 中，mIoU 和 timing 已由原始目录直接确认；BFScore 绝对值未在本次原始目录内直接确认。可安全进入主结果表的字段如下：

| 方法 | mIoU | BFScore | 总训练耗时 | 单 iter 耗时 | 相对变化 |
|---|---:|---:|---:|---:|---|
| OmniSAM baseline | 60.26 | TODO：需补直接证据；候选值 39.97 未在本次原始目录确认 | 2633.36s = 43m53.36s | 5.2655s ≈ 5.27s | reference |
| A_Dtrain_Etrain / Memory-efficient configuration | 60.16 | TODO：需补直接证据；候选值 39.86 未在本次原始目录确认 | 2043.32s = 34m03.32s | 4.0848s ≈ 4.09s | mIoU -0.10；总耗时 -22.4%；单 iter -22.4%；BFScore delta 待直接证据确认 |

注意：若后续补到 `step_09_bfscore_evidence_audit.md` 或原始 train log，并且其中直接记录 baseline BFScore=39.97、A_Dtrain_Etrain BFScore=39.86，则可把 BFScore 字段补进主表。否则，BFScore 只能在 notes 中写为“待确认/历史审计记录未纳入本次原始目录”。

### 7.2 如果部分数字无法确认

保守替代表述：

- 可以填：
  - baseline mIoU = 60.26；A_Dtrain_Etrain mIoU = 60.16；来源均为 full result `metrics.jsonl` line 1。
  - baseline 总耗时 = 43m53.36s；A_Dtrain_Etrain 总耗时 = 34m03.32s；来源均为 full result `metrics.jsonl` / `train_timing.jsonl` line 1 的 `epoch_time/epoch_total_time`。
  - baseline 单 iter = 5.2655s；A_Dtrain_Etrain 单 iter = 4.0848s；来源均为 `step_time/iter_total_time`。
  - mIoU delta = -0.10；总耗时下降约 22.4%；单 iter 下降约 22.4%。
- 必须写 TODO：
  - baseline BFScore = 39.97 的直接原始证据；
  - A_Dtrain_Etrain BFScore = 39.86 的直接原始证据；
  - BFScore delta 的原始数字级计算。
- 只能放 notes，不能进正文主结论：
  - “39.97→39.86”在本次审计中只能作为待补证据候选值；
  - “BFScore -0.11”来自 `step_09` 摘要，可作为线索，但不能替代原始字段证据；
  - clean ablation/profile 的 BFScore 39.59/39.23、step_time 2.35/2.31 不能进入 full main result 主表。

## 8. Unsafe Claims

- 不能写 baseline BFScore=39.97、A_Dtrain_Etrain BFScore=39.86 已由 full result 原始目录直接确认。本次两个主结果目录中没有 run.log，metrics.jsonl 也没有 BFScore 字段。
- 不能把 step_09 中的 BFScore delta -0.11 当作两个 BFScore 绝对值的原始证据。
- 不能把 clean ablation/profile 的 BFScore 39.59/39.23 混入 full DensePASS protocol 主结果表。
- 不能把 clean ablation/profile 的 step_time 2.3505/2.3153 写成 full protocol 的单 iter 耗时；full protocol 原始文件确认的是 5.2655/4.0848。
- 不能在不标注口径的情况下写训练耗时 43m53s→34m03s。原始字段表明这是 epoch_time/epoch_total_time 口径；另有 epoch_train_time 35m00s→24m57s 和 iter_compute_time 4.20s→2.99s。
- 不能声称 full protocol 的 BFScore 已在本次审计中完整闭环；仍需补充 step_09_bfscore_evidence_audit.md 或原始 train log。
- 不能声称多 seed 稳定性。step_09_main_results_draft_report.md line 34-38 明确当前主结果是 seed=1、epoch=1、无方差估计。
