# OmniSAM 中文论文 v1.1 Figure 2--4 插入审计（1-10b-2b）

## 1. 本任务使用的图片

| Figure | 文件路径 | 章节与实验口径 |
|---|---|---|
| Figure 2 | `paper/OmniSAM_Paper_v1_0_CN/figures/fig_speed_accuracy_tradeoff_miou.png` | `sections/05_ablation.tex`；fixed-PL clean ablation/profile |
| Figure 3 | `paper/OmniSAM_Paper_v1_0_CN/figures/fig_token_budget_attention_time.png` | `sections/05_ablation.tex`；fixed-PL clean ablation/profile |
| Figure 4 | `paper/OmniSAM_Paper_v1_0_CN/figures/fig_qualitative_full_protocol.png` | `sections/04_experiments.tex`；Cityscapes → DensePASS full protocol |

Figure 2 为 859×672 RGBA PNG，Figure 3 为 929×671 RGBA PNG；二者均含 Matplotlib 150 DPI 元数据且可正常解码。它们保持原图和原始数字不变，没有重新绘制。两图 caption 和正文均明确限定为 fixed-PL clean ablation/profile，不作为 full protocol 主结果。

## 2. Figure 4 的确定性构建

- 构建脚本：`paper/OmniSAM_Paper_v1_0_CN/tools/build_qualitative_comparison.py`。
- 输出：`paper/OmniSAM_Paper_v1_0_CN/figures/fig_qualitative_full_protocol.png`。
- 样本：24、68、101、111。
- 布局：4 行 × 4 列；列依次为 Input、Ground Truth、OmniSAM Baseline、Ours (MTC+SMR)。
- 输入映射严格从 `notes/baseline_qualitative_export_audit.md` 核验；每一条路径必须出现在该审计中，否则脚本停止。
- 所有单元格保持完整 2048×400 ERP，不 crop、不 flip、不修改 palette、不修改预测类别；只添加白色间隔、英文列标题和样本标记。
- 使用 Pillow 确定性拼接，不使用生成式 AI，不进行模型推理或指标计算。

## 3. Full-protocol checkpoint 身份

### OmniSAM Baseline

`data_pd_exp_final_2000/base_last_20260425_232941/checkpoints/Cityscapes_13cls/sam2_s/best_iou60.26.pth`

该 checkpoint 对应 Cityscapes → DensePASS、sam2_s、13 类、full protocol baseline；正式 mIoU 为 60.26，BFScore 为 39.97。

### Ours (MTC+SMR)

`data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/checkpoints/Cityscapes_13cls/sam2_s/best_iou60.16.pth`

该 checkpoint 对应同一 Cityscapes → DensePASS、sam2_s、13 类、full protocol，正式 mIoU 为 60.16，BFScore 为 39.86。

Figure 4 完全使用 `figures/qualitative_raw/baseline/` 与 `figures/qualitative_raw/ours/` 中由上述两个正式 checkpoint 导出的预测。没有使用旧 `experiments/tecma/eval_predictions_confirmed/A_Dtrain_Etrain/`；后者来自 clean-ablation `best_iou60.02.pth`，不属于本图协议。

## 4. 修改的 LaTeX 文件

1. `paper/OmniSAM_Paper_v1_0_CN/sections/04_experiments.tex`
   - 在“效率分析”之后新增“定性结果”小节。
   - 插入 Figure 4 和保守的定性说明。
2. `paper/OmniSAM_Paper_v1_0_CN/sections/05_ablation.tex`
   - 在 Memory-efficient 模块消融分析后插入 Figure 2。
   - 在 Token budget 分析后插入 Figure 3。

`main.tex` 已存在 `\usepackage{graphicx}`，因此没有修改。没有修改其他 section。

## 5. 新增 labels

- `fig:speed_accuracy_tradeoff`
- `fig:token_budget_attention`
- `fig:qualitative_full_protocol`

三个 label 在论文 `.tex` 范围内均唯一，正文 `\ref{}` 与 label 一致。

## 6. 内容边界确认

- 未修改 `tables/*.tex` 或任何实验数字。
- 未修改 `refs.bib`。
- 未修改 `main.tex`。
- 未修改已有 notes 审计文件。
- 未修改模型、训练、配置代码或任何原始预测。
- 未运行训练、验证、指标重算或模型推理。
- 未生成 zoom-in、虚假箭头/框、Figure 1 pipeline 或 Overleaf zip。
- Figure 3 保留并明确：100\%--25\% 为 E-only curve，final 12.5\% 为 D+E combined endpoint；没有声称 total step time 严格单调下降。
- Figure 4 的正文没有声称 Ours 全面优于 Baseline，也没有把定性样例写成统计显著结论。
