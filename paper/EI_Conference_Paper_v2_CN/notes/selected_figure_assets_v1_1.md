# OmniSAM 中文论文 v1.1：选定图表资产清单（子任务 1-10b-1）

> 工作目录：`/root/autodl-tmp/OmniSAM-main/`  
> 本任务基于 `notes/figure_and_result_asset_inventory.md` 筛选资产。仅复制两张已确定使用的统计图；未运行训练、验证或指标计算，未生成新实验图或拼接图，未修改任何 LaTeX、表格、代码或参考文献。

## A. 已正式选中的统计图

| Figure | 原路径 | 目标路径 | 复制校验 | 建议 caption 用途 |
|---|---|---|---|---|
| Figure 2 | `experiments/tecma/figures/speed_accuracy_tradeoff_miou.png` | `paper/OmniSAM_Paper_v1_0_CN/figures/fig_speed_accuracy_tradeoff_miou.png` | 85887 bytes；SHA-256 与源文件一致 | 速度—精度权衡：比较不同配置的训练效率与 DensePASS mIoU。 |
| Figure 3 | `experiments/tecma/figures/token_budget_attention_time.png` | `paper/OmniSAM_Paper_v1_0_CN/figures/fig_token_budget_attention_time.png` | 80580 bytes；SHA-256 与源文件一致 | Token budget 与 memory attention 时间的关系：展示 memory token 压缩带来的注意力计算收益。 |

### 使用说明

- Figure 2 正式选用 `fig_speed_accuracy_tradeoff_miou.png`。
- Figure 3 正式选用 `fig_token_budget_attention_time.png`。
- 本步骤只准备图片文件；未插入 `figure` 环境，也未修改正文 caption。

## B. 未选中的统计图

| 资产 | 状态 | 原因 |
|---|---|---|
| `experiments/tecma/figures/speed_accuracy_tradeoff_bfscore.png` | 可选未启用；未复制 | BFScore 具体数值及证据口径仍需与既有 BFScore evidence audit 保持一致。v1.1 当前先采用 mIoU trade-off 和 token-budget attention-time 两张证据链更明确的图。 |

## C. Figure 4 定性图候选素材盘点

### Baseline 核对原则与结果

- 只把路径或说明中有明确证据标为 `baseline` 的预测认定为 OmniSAM baseline。
- `A_ccpm` 是一个已确认导出的实验方法目录，但现有文件名和目录结构不足以证明它等同于 OmniSAM baseline；因此下表将其列为“替代方法参考”，**不能标注为 baseline**。
- 本次扫描未找到与候选样本配套、明确标名或有证据说明为 OmniSAM baseline 的 prediction。

| 样本 ID | Input | GT | Ours（A_Dtrain_Etrain） | Baseline | A_ccpm 参考 | 正式四列图是否齐全 |
|---:|---|---|---|---|---|---|
| 24 | `experiments/tecma/figures/qualitative_confirmed/sample_24/input_panorama.png` | `experiments/tecma/figures/qualitative_confirmed/sample_24/ground_truth_color.png` | `experiments/tecma/eval_predictions_confirmed/A_Dtrain_Etrain/24_pred_argmax.png` | **缺失** | `experiments/tecma/eval_predictions_confirmed/A_ccpm/24_pred_argmax.png`（不可标为 baseline） | 否：缺真正 baseline prediction |
| 68 | `experiments/tecma/figures/qualitative_confirmed/sample_68/input_panorama.png` | `experiments/tecma/figures/qualitative_confirmed/sample_68/ground_truth_color.png` | `experiments/tecma/eval_predictions_confirmed/A_Dtrain_Etrain/68_pred_argmax.png` | **缺失** | `experiments/tecma/eval_predictions_confirmed/A_ccpm/68_pred_argmax.png`（不可标为 baseline） | 否：缺真正 baseline prediction |
| 101 | `experiments/tecma/figures/qualitative_confirmed/sample_101/input_panorama.png` | `experiments/tecma/figures/qualitative_confirmed/sample_101/ground_truth_color.png` | `experiments/tecma/eval_predictions_confirmed/A_Dtrain_Etrain/101_pred_argmax.png` | **缺失** | `experiments/tecma/eval_predictions_confirmed/A_ccpm/101_pred_argmax.png`（不可标为 baseline） | 否：缺真正 baseline prediction |
| 111 | `experiments/tecma/figures/qualitative_confirmed/sample_111/input_panorama.png` | `experiments/tecma/figures/qualitative_confirmed/sample_111/ground_truth_color.png` | `experiments/tecma/eval_predictions_confirmed/A_Dtrain_Etrain/111_pred_argmax.png` | **缺失** | `experiments/tecma/eval_predictions_confirmed/A_ccpm/111_pred_argmax.png`（不可标为 baseline） | 否：缺真正 baseline prediction |

### 候选素材判断

- 上述 4 个样本均已定位到 Input、GT 和 `A_Dtrain_Etrain` prediction；同时可定位到同 ID 的 `A_ccpm` prediction。
- 由于缺少真正的 OmniSAM baseline prediction，目前只能形成 Input / GT / A_ccpm / Ours 的技术检查组合，不能把该组合直接写成 Input / GT / Baseline / Ours。
- 本任务未复制这些定性原始素材，也未生成合成 Figure 4。

## D. 结论

| 图编号 | 当前状态 | 缺口/说明 |
|---|---|---|
| Figure 2 | **已具备** | 正式图片已复制到论文 `figures/`。 |
| Figure 3 | **已具备** | 正式图片已复制到论文 `figures/`。 |
| Figure 4 | **暂不具备正式四列图** | 已有 4 个样本的 Input、GT、Ours，并有 A_ccpm 参考预测；缺少有明确身份依据的 OmniSAM baseline prediction，且尚未拼图。 |

### Figure 4 仍需补齐

1. 找到或导出与候选样本 ID 完全一致的 OmniSAM baseline prediction；
2. 核对 baseline 与 ours 的 palette、分辨率和 ERP 展示范围一致；
3. 在后续独立任务中选择 3–4 行并生成统一版式的四列图；
4. 在有方法身份依据前，不得把 `A_ccpm` 改称 OmniSAM baseline。

## 执行边界确认

- 本任务只复制了两张 PNG 到 `paper/OmniSAM_Paper_v1_0_CN/figures/`。
- 本任务唯一新增说明文件为 `paper/OmniSAM_Paper_v1_0_CN/notes/selected_figure_assets_v1_1.md`。
- 未修改 `main.tex`、`sections/*.tex`、`tables/*.tex`、`refs.bib`。
- 未运行训练、验证或指标计算；未生成实验图、拼接图或 zip。
