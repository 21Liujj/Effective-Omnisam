# BFScore Raw Log Check for Full DensePASS Main Results

## 1. Checked Files

| 序号 | 日志路径 | 是否存在 | 是否读取 | 备注 |
|---|---|---|---|---|
| 1 | `logs/omnisam_final_2000/base_last_20260425_232941.log` | 是 | 是 | 仅提取包含 `BFScore`、`[ValSummary]`、`mIoU`、`Epoch`、`Train`、`Val` 的最终验证相关少量行；确认 final validation 摘要位于日志末尾附近 |
| 2 | `logs/omnisam_final_2000/A_Dtrain_Etrain_20260425_232941.log` | 是 | 是 | 仅提取包含 `BFScore`、`[ValSummary]`、`mIoU`、`Epoch`、`Train`、`Val` 的最终验证相关少量行；确认 final validation 摘要位于日志末尾附近 |

## 2. Extracted BFScore Lines

| 方法 | 日志路径 | 行号 | 原始日志行 | 提取的 BFScore | 是否可作为论文主表证据 |
|---|---|---:|---|---:|---|
| OmniSAM baseline | `logs/omnisam_final_2000/base_last_20260425_232941.log` | 48 | `[ValSummary] BFScore=39.97` | 39.97 | 是，原始 full DensePASS protocol 训练/验证日志直接确认 |
| A_Dtrain_Etrain | `logs/omnisam_final_2000/A_Dtrain_Etrain_20260425_232941.log` | 208 | `[ValSummary] BFScore=39.86` | 39.86 | 是，原始 full DensePASS protocol 训练/验证日志直接确认 |

重点确认：

- baseline 存在 `[ValSummary] BFScore=39.97`，日志路径和行号为 `logs/omnisam_final_2000/base_last_20260425_232941.log:L48`。
- A_Dtrain_Etrain 存在 `[ValSummary] BFScore=39.86`，日志路径和行号为 `logs/omnisam_final_2000/A_Dtrain_Etrain_20260425_232941.log:L208`。

## 3. Extracted Related Metric Lines

| 方法 | 指标 | 数值 | 日志路径 | 行号 | 原始日志行 |
|---|---|---:|---|---:|---|
| OmniSAM baseline | FWDice | 88.42 | `logs/omnisam_final_2000/base_last_20260425_232941.log` | 47 | `[ValSummary] FWDice=88.42` |
| OmniSAM baseline | BFScore | 39.97 | `logs/omnisam_final_2000/base_last_20260425_232941.log` | 48 | `[ValSummary] BFScore=39.97` |
| OmniSAM baseline | mIoU | 60.26 | `logs/omnisam_final_2000/base_last_20260425_232941.log` | 49 | `Epoch 1: val mIoU=60.26, mAcc=71.32, PixAcc=88.69, FWIoU=80.76, mDice=72.80, best=60.26; update_best=True (all criteria satisfied)` |
| OmniSAM baseline | mAcc | 71.32 | `logs/omnisam_final_2000/base_last_20260425_232941.log` | 49 | `Epoch 1: val mIoU=60.26, mAcc=71.32, PixAcc=88.69, FWIoU=80.76, mDice=72.80, best=60.26; update_best=True (all criteria satisfied)` |
| OmniSAM baseline | PixAcc | 88.69 | `logs/omnisam_final_2000/base_last_20260425_232941.log` | 49 | `Epoch 1: val mIoU=60.26, mAcc=71.32, PixAcc=88.69, FWIoU=80.76, mDice=72.80, best=60.26; update_best=True (all criteria satisfied)` |
| OmniSAM baseline | FWIoU | 80.76 | `logs/omnisam_final_2000/base_last_20260425_232941.log` | 49 | `Epoch 1: val mIoU=60.26, mAcc=71.32, PixAcc=88.69, FWIoU=80.76, mDice=72.80, best=60.26; update_best=True (all criteria satisfied)` |
| OmniSAM baseline | mDice | 72.80 | `logs/omnisam_final_2000/base_last_20260425_232941.log` | 49 | `Epoch 1: val mIoU=60.26, mAcc=71.32, PixAcc=88.69, FWIoU=80.76, mDice=72.80, best=60.26; update_best=True (all criteria satisfied)` |
| A_Dtrain_Etrain | FWDice | 88.38 | `logs/omnisam_final_2000/A_Dtrain_Etrain_20260425_232941.log` | 207 | `[ValSummary] FWDice=88.38` |
| A_Dtrain_Etrain | BFScore | 39.86 | `logs/omnisam_final_2000/A_Dtrain_Etrain_20260425_232941.log` | 208 | `[ValSummary] BFScore=39.86` |
| A_Dtrain_Etrain | mIoU | 60.16 | `logs/omnisam_final_2000/A_Dtrain_Etrain_20260425_232941.log` | 209 | `Epoch 1: val mIoU=60.16, mAcc=71.42, PixAcc=88.63, FWIoU=80.71, mDice=72.72, best=60.16; update_best=True (all criteria satisfied)` |
| A_Dtrain_Etrain | mAcc | 71.42 | `logs/omnisam_final_2000/A_Dtrain_Etrain_20260425_232941.log` | 209 | `Epoch 1: val mIoU=60.16, mAcc=71.42, PixAcc=88.63, FWIoU=80.71, mDice=72.72, best=60.16; update_best=True (all criteria satisfied)` |
| A_Dtrain_Etrain | PixAcc | 88.63 | `logs/omnisam_final_2000/A_Dtrain_Etrain_20260425_232941.log` | 209 | `Epoch 1: val mIoU=60.16, mAcc=71.42, PixAcc=88.63, FWIoU=80.71, mDice=72.72, best=60.16; update_best=True (all criteria satisfied)` |
| A_Dtrain_Etrain | FWIoU | 80.71 | `logs/omnisam_final_2000/A_Dtrain_Etrain_20260425_232941.log` | 209 | `Epoch 1: val mIoU=60.16, mAcc=71.42, PixAcc=88.63, FWIoU=80.71, mDice=72.72, best=60.16; update_best=True (all criteria satisfied)` |
| A_Dtrain_Etrain | mDice | 72.72 | `logs/omnisam_final_2000/A_Dtrain_Etrain_20260425_232941.log` | 209 | `Epoch 1: val mIoU=60.16, mAcc=71.42, PixAcc=88.63, FWIoU=80.71, mDice=72.72, best=60.16; update_best=True (all criteria satisfied)` |

说明：以上数值均为日志中直接出现的值；本任务没有重新计算 mIoU 或 BFScore。

## 4. Consistency with Previous Audits

- 与 `memory_bfscore_evidence_audit.md` 一致：
  - baseline BFScore = 39.97；
  - A_Dtrain_Etrain BFScore = 39.86；
  - 对应原始日志路径分别为 `logs/omnisam_final_2000/base_last_20260425_232941.log` 和 `logs/omnisam_final_2000/A_Dtrain_Etrain_20260425_232941.log`。
- 与 `memory_full_main_result_audit.md` 一致：
  - `memory_full_main_result_audit.md` 已确认 full result 目录内 `metrics.jsonl` 没有 BFScore 字段；本次进一步确认 BFScore 存在于 full DensePASS protocol 原始验证日志的 `[ValSummary]` 行中。
- 现在可以把 39.97 / 39.86 写入论文 v1.0 主结果表。
- 来源建议标注为：
  - `logs/omnisam_final_2000/base_last_20260425_232941.log:L48`
  - `logs/omnisam_final_2000/A_Dtrain_Etrain_20260425_232941.log:L208`
- 如果表格脚注需要说明口径，建议写：BFScore is extracted from `[ValSummary] BFScore=...` in the full DensePASS protocol validation logs, because `metrics.jsonl` does not store BFScore.

## 5. Paper Recommendation

论文 v1.0 主结果表可以填入 BFScore 39.97 和 39.86，来源为 full DensePASS protocol 原始验证日志。
