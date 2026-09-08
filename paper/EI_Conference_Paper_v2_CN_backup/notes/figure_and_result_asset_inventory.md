# OmniSAM 中文论文 v1.1：实验结果资产盘点（子任务 1-10b-0）

> 扫描基准：`/root/autodl-tmp/OmniSAM-main/`  
> 本次仅做已存在资产的文件级盘点与小型 CSV/文本元数据检查；未运行训练、验证，未生成新图或 zip，未修改论文正文、LaTeX 表格或参考文献。

## 1. 结论摘要

- **可直接用于论文的统计图已有 3 张**：mIoU 速度—精度图、BFScore 速度—质量图、token budget—attention time 图。
- **可直接作为论文表格数据源的核心 CSV 已齐全**：干净消融、token budget、训练时间拆分、per-class IoU；另有 PseudoLabelRefiner full-run 主指标与逐类 IoU。
- **定性素材数量充足，但“baseline vs ours”成品对比链路仍不完整**：confirmed prediction 目录包含 4 个方法变体的 12 张预测（共 48 张），未发现名为 baseline 的预测目录；diagnostic 目录有 input、GT、overlay/panel，但存在 `missing_predictions.txt`，不能直接当作完整 baseline-vs-ours 成品图。
- 因此：**足以支撑 v1.1 的两张效率/权衡主图和一张机制图；定性 Figure 4 有可筛选素材，但进入论文前仍需在 1-10b-1 中核对并组版，且需要补齐/定位 baseline prediction。**

## 2. 指定候选资产逐项确认

| 路径 | 类型 | 存在 | 大小 | 可能的论文用途 | v1.1 建议 |
|---|---|---:|---:|---|---|
| `experiments/tecma/figures/speed_accuracy_tradeoff_miou.png` | PNG | 是 | 83.9 KiB | 速度—精度（mIoU）权衡主图 | 是，直接纳入 |
| `experiments/tecma/figures/speed_accuracy_tradeoff_bfscore.png` | PNG | 是 | 88.0 KiB | 速度—边界质量（BFScore）权衡图 | 是，但正文引用前核对 BFScore 证据口径 |
| `experiments/tecma/figures/token_budget_attention_time.png` | PNG | 是 | 78.7 KiB | memory token budget 与 attention 时间关系 | 是，适合作为效率/机制图 |
| `experiments/tecma/tables/clean_ablation_results.csv` | CSV | 是 | 2.0 KiB | 干净消融主表的数据源 | 是，作为表格权威数据源 |
| `experiments/tecma/tables/token_budget_tradeoff.csv` | CSV | 是 | 2.3 KiB | token budget 精度—效率权衡表数据源 | 是 |
| `experiments/tecma/tables/training_time_breakdown.csv` | CSV | 是 | 1.6 KiB | Train/PL/Val/E2E 时间拆分数据源 | 是；报告时保持阶段分离 |
| `experiments/tecma/tables/per_class_iou.csv` | CSV | 是 | 1.3 KiB | DensePASS 13 类 per-class IoU 与类别增减分析 | 是，可制类别柱状图/附表 |
| `experiments/pl_refiner/tables/pl_fullrun_main_metrics.csv` | CSV | 是 | 0.6 KiB | PseudoLabelRefiner full-run 主指标 | 有条件；先确认其是否进入 v1.1 主线 |
| `experiments/pl_refiner/tables/pl_fullrun_per_class_iou.csv` | CSV | 是 | 0.6 KiB | PseudoLabelRefiner per-class IoU | 有条件；适合作为补充分析 |

## 3. 可视化资产目录盘点

| 目录 | 文件级情况 | input | GT | baseline pred | ours/方法预测 | overlay/color mask | 判断 |
|---|---|---:|---:|---:|---:|---:|---|
| `experiments/tecma/eval_predictions_confirmed/` | 50 个文件（PNG 48） | 未从文件名确认 | 未从文件名确认 | 未发现 | 有 | 未从文件名确认 | 确认导出的 raw argmax 预测；需与 input/GT 配对并确认 baseline 来源后组版。 |
| `experiments/tecma/figures/qualitative_diagnostic/` | 73 个文件（PNG 60） | 有 | 有 | 未发现 | 有 | 有 | 诊断候选素材；检测到 missing_predictions，不能直接视为完整方法对比图。 |
| `experiments/tecma/figures/failure_cases_diagnostic/` | 8 个文件（PNG 6） | 未从文件名确认 | 有 | 未发现 | 未从文件名确认 | 未从文件名确认 | 已有素材，需人工看图确认内容与版式。 |
| `experiments/tecma/figures/qualitative_confirmed/` | 132 个文件（PNG 84） | 有 | 有 | 未发现 | 有 | 有 | 已有 panel/input/GT，可作为优先筛选的定性素材。 |
| `experiments/tecma/figures/failure_cases_confirmed/` | 6 个文件（PNG 6） | 未从文件名确认 | 有 | 未发现 | 未从文件名确认 | 未从文件名确认 | 已有素材，需人工看图确认内容与版式。 |

### 3.1 confirmed prediction 细节

- `eval_predictions_confirmed/` 下识别到的方法目录：`A_Dtrain`, `A_Dtrain_Etrain`, `A_Etrain_keep025`, `A_ccpm`。
- 每个方法目录对应同一组 12 个 DensePASS 验证样本，文件名为 `*_pred_argmax.png`；总计 48 张预测。
- **未发现 `baseline/` 目录，也未发现文件名明确标注 baseline 的预测。**当前目录本身不能独立形成 Input / GT / Baseline / Ours 四列成品图。
- `A_Dtrain_Etrain` 可作为 ours 候选，但必须以 v1.1 的方法定义和结果表口径为准。

### 3.2 diagnostic / failure-case 细节

- `qualitative_diagnostic/` 检出 12 个样本目录：`sample_101`, `sample_111`, `sample_122`, `sample_155`, `sample_210`, `sample_24`, `sample_263`, `sample_289`, `sample_305`, `sample_319`, `sample_363`, `sample_68`。
- 从文件名可确认常见资产包括：`input_panorama.png`、`ground_truth_color.png`、`ground_truth_focus_overlay.png`、`seam_candidate_overlay.png`、`diagnostic_candidate_panel.png`。
- 多个样本含 `missing_predictions.txt`，说明这些目录主要用于候选诊断，而非已完成的 baseline-vs-ours 定性成图。
- `failure_cases_diagnostic/` / `failure_cases_confirmed/` 已存在，应在下一步优先人工查看 panel，选择真正能解释 seam、边界、小目标或类别混淆的案例，避免仅凭文件名写结论。

## 4. 表格与可画统计图的数据源

以下 CSV 均为小型、可直接被论文表格或绘图脚本消费的数据源：

| 数据源 | 数据行数（不含表头） | 字段/表头 | 用途 |
|---|---:|---|---|
| `experiments/tecma/tables/clean_ablation_results.csv` | 6 | `config / mIoU / BFScore / mAcc / PixAcc / FWIoU / mDice / step_time / iter_compute_time / memory_attention_time / gpu_mem_peak / memory_tokens / compression_before / compression_after / compression_ratio / sparse_before / sparse_after / sparse_keep_ratio / delta_mIoU / delta_BFScore / attn_time_reduction_pct / token_reduction_x / peak_mem_saving_GB / step_time_change_pct / run_dir` | 主结果/消融表 |
| `experiments/tecma/tables/token_budget_tradeoff.csv` | 5 | `budget_label / config / curve_type / module_setting / memory_tokens / token_ratio / token_reduction_x / mIoU / delta_mIoU / BFScore / delta_BFScore / mAcc / PixAcc / FWIoU / mDice / step_time / step_time_change_pct / memory_attention_time / attn_time_reduction_pct / gpu_mem_peak / peak_mem_saving_GB / peak_mem_saving_pct / run_dir / pl_reuse_evidence` | 精度—效率权衡图表 |
| `experiments/tecma/tables/training_time_breakdown.csv` | 10 | `table / config / modules / tokens / mIoU / BFScore / step_time / data_time / iter_compute_time / ma_total / ma_per_iter_ms / ma_share_pct / non_mem_per_iter_s / step_delta_pct / ma_red_pct / gpu_peak / n_iters` | 效率拆分表 |
| `experiments/tecma/tables/per_class_iou.csv` | 13 | `class_id / class_name / baseline_omnisam_IoU / A_ccpm_IoU / A_Dtrain_Etrain_IoU / delta_IoU / abs_delta_IoU / delta_vs_baseline / rank_drop / note` | 逐类柱状图/类别增减分析 |
| `experiments/pl_refiner/tables/pl_fullrun_main_metrics.csv` | 3 | `Run / Method / Memory-efficient / PL Refiner / mIoU / d_mIoU / mAcc / d_mAcc / PixAcc / FWIoU / mDice / BFScore / d_BFScore / step_time_s / epoch_time_s / peak_mem_GB / train_steps / val_steps / done` | 主结果/消融表 |
| `experiments/pl_refiner/tables/pl_fullrun_per_class_iou.csv` | 13 | `Class / R0 / R2 / R2R4 / d_R2_minus_R0 / d_R2R4_minus_R2 / d_R2R4_minus_R0` | 逐类柱状图/类别增减分析 |

### 4.1 其他已存在的表格/说明资产

- `experiments/tecma/tables/` 还包含对应 Markdown 版本，以及 `main_results_draft.md`、`main_results_registry.md`、`paper_ablation_table_draft.md`、`memory_token_attention_cost.csv/.md`、`profile_summary.csv`。这些适合用于核对数据来源与复现实验口径。
- `paper/OmniSAM_Paper_v1_0_CN/tables/` 已有 5 个 `.tex` 表：`main_results.tex`、`efficiency_analysis.tex`、`memory_ablation.tex`、`pseudo_label_ablation.tex`、`token_budget_tradeoff.tex`。本任务仅确认其存在，**未修改**。
- `paper/OmniSAM_Paper_v1_0_CN/figures/` 当前为空，因此候选统计图尚未复制进论文 figures 目录。

## 5. 额外数据集资产

- `logs/wildpass_results/`：共 106 个文件；其中 `vis_check/` 有 10 张 `overlay_*.png`，并有 metrics/timing JSON、summary Markdown、CSV。可作为 WildPASS 补充实验/数据检查资产，但 overlay 文件名未表明 baseline/ours，不能直接作为主对比图。
- `logs/Standford2d3d_results/`：共 21 个文件；其中 `vis_check/` 有 10 张 `overlay_*.png`。同时存在 `MISSING_CKPT.md`、`MISSING_PATHS.md` 与 `NEED_CONFIRM_STANFORD360_MAPPING.md`，说明该组证据链尚未闭合，**不建议直接纳入 v1.1 主结果**；仅可暂列补充诊断候选。

## 6. v1.1 图编号建议

### Figure 2（主效率—精度图）

- 首选：`experiments/tecma/figures/speed_accuracy_tradeoff_miou.png`。
- 理由：直接表达主结论“训练/内存注意力加速，同时 mIoU 基本持平”，最适合作为主结果图。

### Figure 3（机制/效率解释图）

- 首选：`experiments/tecma/figures/token_budget_attention_time.png`。
- 可选并列子图：`experiments/tecma/figures/speed_accuracy_tradeoff_bfscore.png`，但在引用具体 BFScore 前需继续遵守已有证据审计结论。

### Figure 4（定性结果）

- 优先素材池：`experiments/tecma/figures/qualitative_confirmed/`，其次为 `experiments/tecma/figures/failure_cases_confirmed/`。
- raw prediction 来源：`experiments/tecma/eval_predictions_confirmed/A_Dtrain_Etrain/`（ours 候选）。
- input/GT/诊断 overlay 来源：`experiments/tecma/figures/qualitative_diagnostic/sample_*/`。
- **当前不建议直接选用某一张作为最终 Figure 4**：缺少明确 baseline prediction 资产，且 diagnostic 样本中存在 missing prediction 标记。下一步应人工看图并确认四列配对后再复制/组版。

## 7. 当前缺口

要形成“可直接发论文”的定性对比图，目前至少缺少或尚未定位：

1. 与 12 个 confirmed 样本严格同 ID 的 **baseline prediction**；
2. 明确且统一的 Input / GT / Baseline / Ours 四列成品 panel；
3. baseline 与 ours 使用同一 palette、同一尺寸、同一 crop/ERP 展示方式的确认记录；
4. failure case 的文字归因与指标证据（不能仅凭视觉猜测）；
5. 若采用 BFScore 图，正文中的 BFScore 数值和来源需与已有审计记录一致。

## 8. 下一步建议

建议进入 **1-10b-1：筛选并拷贝论文使用的图表资产**：

1. 先人工查看 3 张统计图的分辨率、字体和配色，确定 Figure 2/3；
2. 从 `qualitative_confirmed/` 和 `failure_cases_confirmed/` 各选 2–4 个样本；
3. 定位或补齐这些样本的 baseline prediction，再核对 `A_Dtrain_Etrain` 是否为 v1.1 的 ours；
4. 只在确认后将选定资产复制到论文 `figures/`，并保持原始实验资产不变；
5. 用 CSV 重新核对 LaTeX 表中的数值，但不要在资产筛选阶段改写实验结论。

## 9. 执行边界确认

- 未运行训练；未运行验证；未生成图片；未复制图片；未生成 zip。
- 未修改 `sections/*.tex`、`tables/*.tex`、`refs.bib`。
- 本次唯一写入文件：`paper/OmniSAM_Paper_v1_0_CN/notes/figure_and_result_asset_inventory.md`。
