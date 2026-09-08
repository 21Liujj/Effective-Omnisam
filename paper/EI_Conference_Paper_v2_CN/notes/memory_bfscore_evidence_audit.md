# BFScore Evidence Audit for Full DensePASS Main Results

## 1. Audit Scope

| 序号 | 路径 | 是否存在 | 是否读取 | 读取目的 |
|---|---|---|---|---|
| 1 | `experiments/tecma/notes/step_09_bfscore_evidence_audit.md` | 是 | 是 | 专门补查 full main result BFScore 证据；读取 BFScore 数值、原始日志路径和行号引用 |
| 2 | `experiments/tecma/notes/step_09_main_results_draft_report.md` | 是 | 是 | 核对 full DensePASS protocol 主结果来源说明和 BFScore delta 摘要 |
| 3 | `experiments/tecma/notes/step_19_per_class_iou_report.md` | 是 | 是 | 检查是否含 BFScore 或 boundary 相关主结果证据 |
| 4 | `experiments/tecma/notes/step_19_per_class_iou_diagnostic_report.md` | 是 | 是 | 检查 diagnostic 口径中的 BFScore / boundary 文本，避免误用 |
| 5 | `experiments/tecma/notes/step_20_21_confirmed_prediction_visualization_report.md` | 是 | 是 | 检查 visualization/diagnostic 是否含 BFScore 或 boundary 线索 |
| 6 | `experiments/tecma/tables/paper_ablation_table_draft.md` | 是 | 是 | 检查 fixed-PL clean ablation BFScore，确认与 full protocol 区分 |
| 7 | `experiments/tecma/tables/clean_ablation_results.csv` | 是 | 是 | 检查 clean ablation BFScore 数字和 run 口径 |
| 8 | `experiments/tecma/tables/per_class_iou.csv` | 是 | 是 | 检查是否含 boundary 类别说明；不含 BFScore 主指标 |
| 9 | `experiments/tecma/tables/per_class_iou_diagnostic.csv` | 是 | 是 | 检查 diagnostic 表中是否含 BFScore；未发现 BFScore 字段 |
| 10 | `experiments/tecma/figures/failure_cases_diagnostic/failure_case_candidates.csv` | 是 | 是 | 检查 boundary/failure case 诊断口径 |
| 11 | `experiments/tecma/figures/qualitative_diagnostic/diagnostic_sample_manifest.csv` | 是 | 是 | 检查 qualitative diagnostic 中 boundary 相关样本说明 |
| 12 | `data_pd_exp_final_2000/base_last_20260425_232941/` | 是 | 是（仅文件名轻量查找） | 在目录内查找文件名含 `bfscore` / `boundary` / `bfs` / `summary` / `metric` / `result` 的文本/表格候选 |
| 13 | `data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/` | 是 | 是（仅文件名轻量查找） | 同上 |

补充说明：两个 full result 目录内本次只找到 `checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` 命中文件名关键词；未在这两个目录内发现 `bfscore`、`boundary`、`bfs`、`summary`、`result` 相关额外文本/表格文件。`metrics.jsonl` 已在 0-2b 中确认不含 BFScore 字段。

## 2. BFScore Evidence Candidates

| 序号 | 来源文件 | 方法/配置 | BFScore 数值 | 是否 full protocol | 证据类型 | 是否可进论文主表 |
|---|---|---|---:|---|---|---|
| 1 | `experiments/tecma/notes/step_09_bfscore_evidence_audit.md` lines 19-22 | baseline_omnisam | 39.97 | 是，报告称对应正式 final_2000 / full DensePASS protocol | 报告证据；引用原始训练日志 `logs/omnisam_final_2000/base_last_20260425_232941.log` L48 `[ValSummary] BFScore=39.97` | 可以作为论文主表候选，但应在 notes 中注明来源是 BFScore 证据审计报告引用训练日志；若要求最严格原始证据，需再读取/归档原始 log |
| 2 | `experiments/tecma/notes/step_09_bfscore_evidence_audit.md` lines 19-22 | A_Dtrain_Etrain | 39.86 | 是，报告称对应正式 final_2000 / full DensePASS protocol | 报告证据；引用原始训练日志 `logs/omnisam_final_2000/A_Dtrain_Etrain_20260425_232941.log` L208 `[ValSummary] BFScore=39.86` | 可以作为论文主表候选，但应注明同上 |
| 3 | `experiments/tecma/notes/step_09_bfscore_evidence_audit.md` lines 28-41 | baseline_omnisam / A_Dtrain_Etrain | 39.97 / 39.86 | 是，报告摘录与 full mIoU 相邻的 ValSummary 片段 | 报告内日志片段摘录；不是本次直接读取原始 `.log` | 可以作为主表候选证据；建议后续把原始 log 路径也纳入 paper notes 索引 |
| 4 | `experiments/tecma/notes/step_09_main_results_draft_report.md` lines 20-26, 59-65 | A_Dtrain_Etrain vs baseline | delta=-0.11 | 是，full DensePASS protocol 摘要 | 报告摘要；未给 BFScore 绝对值 | 可写 delta 线索；不能单独替代绝对值来源 |
| 5 | `experiments/tecma/tables/paper_ablation_table_draft.md` lines 13-20 | OmniSAM baseline / TeCMA full 等 | 39.59 / 39.23 等 | 否，fixed-PL clean ablation | clean ablation 表；说明 BFScore 来自 run.log `[ValSummary]` | 不可进 full protocol 主表；可进消融表 |
| 6 | `experiments/tecma/tables/clean_ablation_results.csv` lines 1-7 | baseline_omnisam / A_Dtrain_Etrain 等 | 39.59 / 39.23 等 | 否，fixed-PL clean ablation/profile | clean ablation CSV | 不可进 full protocol 主表；可进消融/profile 表 |
| 7 | `experiments/tecma/notes/step_19_per_class_iou_diagnostic_report.md` lines 9-12 | A_ccpm / A_Dtrain / A_Etrain_keep025 / A_Dtrain_Etrain | 39.59 / 39.23 / 39.29 / 39.23 | 否，diagnostic / clean ablation 相关 run | diagnostic 报告中的 run 摘要 | 不可进 full protocol 主表 |
| 8 | `experiments/tecma/notes/step_19_per_class_iou_report.md` | A_ccpm / A_Dtrain_Etrain / baseline_omnisam | 未列 BFScore | 否，per-class IoU consolidation | diagnostic/per-class IoU 报告 | 不可作为 BFScore 主表证据 |
| 9 | `experiments/tecma/notes/step_20_21_confirmed_prediction_visualization_report.md` line 43 | diagnostic sample | 无 BFScore；含 `boundary_erosion` | 否，visualization/diagnostic | 定性可视化报告 | 不可作为 BFScore 主表证据 |
| 10 | `experiments/tecma/figures/failure_cases_diagnostic/failure_case_candidates.csv` lines 3-7 | failure case candidates | 无 BFScore；含 boundary/边界候选 | 否，diagnostic | failure case 候选表 | 不可作为 BFScore 主表证据 |
| 11 | `experiments/tecma/figures/qualitative_diagnostic/diagnostic_sample_manifest.csv` lines 2-13 | qualitative samples | 无 BFScore；含 boundary/thin structure 样本说明 | 否，diagnostic | qualitative manifest | 不可作为 BFScore 主表证据 |
| 12 | full result directories 内 `metrics.jsonl` | baseline_omnisam / A_Dtrain_Etrain | 未发现 BFScore | 是，full protocol 原始目录 | 原始 metrics 文件，但无 BFScore 字段 | 不能提供 BFScore 主表数字 |

## 3. Target Numbers Check

| 目标数字 | 是否找到直接证据 | 来源文件 | 来源字段/行 | 是否可写入论文 |
|---|---|---|---|---|
| baseline BFScore = 39.97 | 是（报告证据引用原始训练日志；本次未直接读取原始 log） | `experiments/tecma/notes/step_09_bfscore_evidence_audit.md` | lines 19-22；lines 28-33；引用 `logs/omnisam_final_2000/base_last_20260425_232941.log` L48 `[ValSummary] BFScore=39.97` | 可以写入论文主表，但建议脚注/notes 标明 BFScore 来自训练日志审计；若导师要求“原始日志本体”，需后续补读/归档 log |
| A_Dtrain_Etrain BFScore = 39.86 | 是（报告证据引用原始训练日志；本次未直接读取原始 log） | `experiments/tecma/notes/step_09_bfscore_evidence_audit.md` | lines 19-22；lines 36-41；引用 `logs/omnisam_final_2000/A_Dtrain_Etrain_20260425_232941.log` L208 `[ValSummary] BFScore=39.86` | 可以写入论文主表，条件同上 |
| BFScore delta = -0.11 | 是 | `experiments/tecma/notes/step_09_main_results_draft_report.md`; `step_09_bfscore_evidence_audit.md` | `step_09_main...` lines 61-63；绝对值 39.97/39.86 可计算差值 -0.11，但本报告不把 delta 反推为绝对值 | 可以写入论文；建议同时引用绝对值审计报告 |

注意：本次确实在 `step_09_bfscore_evidence_audit.md` 中找到了 39.97/39.86，但该文件是历史证据审计报告，证据类型是“报告引用原始训练日志行”，不是本次直接打开 `logs/omnisam_final_2000/*.log` 得到的原始日志证据。

## 4. Conflict and Scope Analysis

| 数字 | 方法/配置 | 实验口径 | 来源 | 解释 |
|---:|---|---|---|---|
| 39.97 | baseline_omnisam | full DensePASS protocol / final_2000 | `step_09_bfscore_evidence_audit.md` lines 19-22, 28-33 | 存在于专门 BFScore 审计报告中，报告引用正式训练日志 `logs/omnisam_final_2000/base_last_20260425_232941.log` L48 |
| 39.86 | A_Dtrain_Etrain | full DensePASS protocol / final_2000 | `step_09_bfscore_evidence_audit.md` lines 19-22, 36-41 | 存在于专门 BFScore 审计报告中，报告引用正式训练日志 `logs/omnisam_final_2000/A_Dtrain_Etrain_20260425_232941.log` L208 |
| -0.11 | A_Dtrain_Etrain vs baseline | full DensePASS protocol / final_2000 | `step_09_main_results_draft_report.md` lines 61-63 | full main result 摘要中的 BFScore 相对变化；不能单独反推绝对值，但与 39.97/39.86 一致 |
| 39.59 | baseline_omnisam 或 A_ccpm | fixed-PL clean ablation/profile | `paper_ablation_table_draft.md` lines 13-16；`clean_ablation_results.csv` lines 1-3 | 不是 full protocol；用于 clean ablation/profile，不可混入主结果表 |
| 39.23 | A_Dtrain / A_Etrain / Dtrain_Etrain / A_Dtrain_Etrain | fixed-PL clean ablation/profile | `paper_ablation_table_draft.md` lines 17-20；`clean_ablation_results.csv` lines 4-7 | 不是 full protocol；用于消融/profile，不可写成 A_Dtrain_Etrain full main BFScore |
| 39.29 | A_Etrain_keep025 | diagnostic / token budget run | `step_19_per_class_iou_diagnostic_report.md` lines 9-12 | diagnostic/token-budget 口径，不能用于 full main table |

结论：39.97 / 39.86 存在，但当前可见证据是 `step_09_bfscore_evidence_audit.md` 对原始训练日志的审计摘录；39.59 / 39.23 明确是 fixed-PL clean ablation/profile 口径。两组数字不能混用，因为 full protocol 使用正式 final_2000 目录和完整 PL/训练流程，而 clean ablation/profile 使用 `experiments/tecma/runs/...` 下的 fixed pseudo-label 消融与 profile 口径。

## 5. Recommendation for Paper v1.0

### 如果 39.97 和 39.86 找到直接证据

本次在 `step_09_bfscore_evidence_audit.md` 中找到 39.97 和 39.86，并且该报告明确标注 `verified_from_train_log`，引用了原始训练日志路径和行号。因此论文 v1.0 主结果表可暂填：

| 方法 | BFScore | BFScore 证据 |
|---|---:|---|
| OmniSAM baseline | 39.97 | `step_09_bfscore_evidence_audit.md` lines 19-22, 28-33；引用 `logs/omnisam_final_2000/base_last_20260425_232941.log` L48 |
| A_Dtrain_Etrain | 39.86 | `step_09_bfscore_evidence_audit.md` lines 19-22, 36-41；引用 `logs/omnisam_final_2000/A_Dtrain_Etrain_20260425_232941.log` L208 |

建议在 `notes` 或表格脚注中说明：BFScore 未写入 full result `metrics.jsonl`，而是由 `[ValSummary] BFScore=...` 训练日志行审计得到；本次未重新评测、未重新计算 BFScore。

### 如果没有找到直接证据

严格来说，如果导师/投稿要求“本次审计必须直接打开原始 `.log` 文件而不是引用历史审计报告”，则仍应保守处理：

- 主结果表 BFScore 字段可暂写 `TODO` 或加脚注“来自历史 BFScore evidence audit”；
- 正文只写 “BFScore 相对变化约 -0.11，仍需补充原始日志归档证据”；
- 或暂时不在主结果表中列 BFScore，只在后续实验计划中补充原始 log 归档。

## 6. Unsafe Claims

- 不能说 BFScore 是从 full result 目录内的 `metrics.jsonl` 读取的；该字段在 full result `metrics.jsonl` 中不存在。
- 不能把 clean ablation 的 39.59 / 39.23 写成 full DensePASS protocol 的 BFScore。
- 不能只用 `delta=-0.11` 反推 39.97 / 39.86 并伪装成原始证据。
- 不能把 diagnostic / visualization 中的 boundary、thin structure、failure case 文本当作 BFScore 指标证据。
- 不能声称本任务重新计算或重新验证了 BFScore；本任务只读取已有文本/表格/日志审计材料。
- 不能忽略证据口径：39.97/39.86 是 full main result 日志审计口径，39.59/39.23 是 clean ablation/profile 口径。
