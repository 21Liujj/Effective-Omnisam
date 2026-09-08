# OmniSAM full-protocol Baseline 定性预测导出审计（1-10b-2a）

> 本审计只针对 DensePASS val 样本 24、68、101、111 的轻量推理导出。未运行训练、未进行反向传播、未执行完整验证、未重新计算或修改论文指标。

## 1. GPU 与仓库预检查

- GPU：NVIDIA GeForce RTX 5090，预检查时 0 MiB / 32607 MiB、0% utilization，可用于轻量推理。
- 工作目录：`/root/autodl-tmp/OmniSAM-main/`。
- `git status --short` 显示仓库原先已有大量修改/未跟踪文件；本任务未清理、覆盖或改动这些既有工作。

## 2. 正式 checkpoint 与身份依据

### OmniSAM Baseline

- 完整路径：`/root/autodl-tmp/OmniSAM-main/data_pd_exp_final_2000/base_last_20260425_232941/checkpoints/Cityscapes_13cls/sam2_s/best_iou60.26.pth`。
- checkpoint 文件名：`best_iou60.26.pth`。
- 正式 run：`base_last_20260425_232941`。
- 协议：Cityscapes → DensePASS，`sam2_s`，Cityscapes/DensePASS 13 类，full protocol，seed=1，2000 张 PL，1 epoch。
- 配置：`configs/ablations/omnisam_base.yaml`；启动脚本对正式 `base_last` 明确覆盖为 A/CCPM、D/MTC、E/SMR 全部关闭。
- 启动入口：`run_adaptation.py`，正式启动脚本为 `scripts/run_plugin_final_2000_base_vs_ADEtrain.sh`。
- 指标身份证据：`.../base_last.../metrics.jsonl` 记录 `val_mIoU=60.26`；`logs/omnisam_final_2000/base_last_20260425_232941.log` 记录 `[ValSummary] BFScore=39.97` 与 mIoU=60.26。

### Full-protocol Ours

- 完整路径：`/root/autodl-tmp/OmniSAM-main/data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/checkpoints/Cityscapes_13cls/sam2_s/best_iou60.16.pth`。
- checkpoint 文件名：`best_iou60.16.pth`。
- 正式 run：`A_Dtrain_Etrain_20260425_232941`，即论文 Ours（A + D_train + E_train / MTC + SMR）。
- 同一启动脚本、同一 seed、同一 2000 张 PL、同一 DensePASS 13 类验证协议；D/E 在 PL 关闭、训练启用。
- 指标身份证据：对应 `metrics.jsonl` 记录 `val_mIoU=60.16`；原始 full log 记录 BFScore=39.86。

### 为什么没有复用旧 prediction

- 全仓库未找到来自 `base_last_20260425_232941/best_iou60.26.pth` 的现成 baseline 定性 prediction，因此本次进行新推理。
- `experiments/tecma/eval_predictions_confirmed/A_ccpm/` 来自 clean-ablation `A_ccpm.../best_iou60.26.pth`，不能冒充 full-protocol OmniSAM baseline。
- 更重要的是，旧 `eval_predictions_confirmed/A_Dtrain_Etrain/` 来自 clean-ablation `best_iou60.02.pth`，不是 full-protocol ours 60.16。经用户确认，本次同时导出 full baseline 和 full ours，以保证 Figure 4 两个方法列协议完全一致。

## 3. 使用的导出入口与处理协议

- 工具：`paper/OmniSAM_Paper_v1_0_CN/tools/export_full_protocol_qualitative.py`。
- 该工具复用项目已有 `DensePASS13`、`build_sam2`、`OmniSAM_adapt`、`smart_load_checkpoint`、`merge_width_scatter_np`；未重新实现模型核心逻辑。
- 与既有 confirmed exporter 一致：`crop_size=(4096,1024)`、`sliding_window=(384,1024)`、`passes=2`、双 pass memory inference、原尺寸 merge 后用 nearest-neighbor 输出到 `2048×400`。
- 仅处理 4 个 subset 条目；临时 subset list 位于系统 `/tmp` 并在推理结束后删除，未修改原始 data list。
- 推理使用 `model.eval()`、`torch.no_grad()`，无反向传播。

## 4. 样本 ID 与真实文件映射

| 样本 ID | subset 条目 | DensePASS input | DensePASS GT | 原始 input 尺寸 | 原始 GT 尺寸 |
|---:|---|---|---|---:|---:|
| 24 | `/24_.png` | `data/DensePASS/leftImg8bit/val/24_.png` | `data/DensePASS/gtFine/val/24_labelTrainIds.png` | 2048×400 | 2048×400 |
| 68 | `/68_.png` | `data/DensePASS/leftImg8bit/val/68_.png` | `data/DensePASS/gtFine/val/68_labelTrainIds.png` | 2048×400 | 2048×400 |
| 101 | `/101_.png` | `data/DensePASS/leftImg8bit/val/101_.png` | `data/DensePASS/gtFine/val/101_labelTrainIds.png` | 2048×400 | 2048×400 |
| 111 | `/111_.png` | `data/DensePASS/leftImg8bit/val/111_.png` | `data/DensePASS/gtFine/val/111_labelTrainIds.png` | 2048×400 | 2048×400 |

编号不是未经验证的数据集索引假设：`DensePASS13` list 条目 `/24_.png` 被 loader 返回为样本名 `24_.png`；input/GT 的真实文件命名已逐项存在性核对。

## 5. Input / GT / Baseline / Ours 素材清单

| ID | Input | GT | Full Baseline argmax / color | Full Ours argmax / color | 预测尺寸 | 类别范围 |
|---:|---|---|---|---|---:|---|
| 24 | `data/DensePASS/leftImg8bit/val/24_.png` | `data/DensePASS/gtFine/val/24_labelTrainIds.png` | `paper/OmniSAM_Paper_v1_0_CN/figures/qualitative_raw/baseline/sample_024/pred_argmax.png`<br>`paper/OmniSAM_Paper_v1_0_CN/figures/qualitative_raw/baseline/sample_024/pred_color.png` | `paper/OmniSAM_Paper_v1_0_CN/figures/qualitative_raw/ours/sample_024/pred_argmax.png`<br>`paper/OmniSAM_Paper_v1_0_CN/figures/qualitative_raw/ours/sample_024/pred_color.png` | 2048×400（两者一致） | baseline 0–12；ours 0–12 |
| 68 | `data/DensePASS/leftImg8bit/val/68_.png` | `data/DensePASS/gtFine/val/68_labelTrainIds.png` | `paper/OmniSAM_Paper_v1_0_CN/figures/qualitative_raw/baseline/sample_068/pred_argmax.png`<br>`paper/OmniSAM_Paper_v1_0_CN/figures/qualitative_raw/baseline/sample_068/pred_color.png` | `paper/OmniSAM_Paper_v1_0_CN/figures/qualitative_raw/ours/sample_068/pred_argmax.png`<br>`paper/OmniSAM_Paper_v1_0_CN/figures/qualitative_raw/ours/sample_068/pred_color.png` | 2048×400（两者一致） | baseline 0–12；ours 0–12 |
| 101 | `data/DensePASS/leftImg8bit/val/101_.png` | `data/DensePASS/gtFine/val/101_labelTrainIds.png` | `paper/OmniSAM_Paper_v1_0_CN/figures/qualitative_raw/baseline/sample_101/pred_argmax.png`<br>`paper/OmniSAM_Paper_v1_0_CN/figures/qualitative_raw/baseline/sample_101/pred_color.png` | `paper/OmniSAM_Paper_v1_0_CN/figures/qualitative_raw/ours/sample_101/pred_argmax.png`<br>`paper/OmniSAM_Paper_v1_0_CN/figures/qualitative_raw/ours/sample_101/pred_color.png` | 2048×400（两者一致） | baseline 0–12；ours 0–12 |
| 111 | `data/DensePASS/leftImg8bit/val/111_.png` | `data/DensePASS/gtFine/val/111_labelTrainIds.png` | `paper/OmniSAM_Paper_v1_0_CN/figures/qualitative_raw/baseline/sample_111/pred_argmax.png`<br>`paper/OmniSAM_Paper_v1_0_CN/figures/qualitative_raw/baseline/sample_111/pred_color.png` | `paper/OmniSAM_Paper_v1_0_CN/figures/qualitative_raw/ours/sample_111/pred_argmax.png`<br>`paper/OmniSAM_Paper_v1_0_CN/figures/qualitative_raw/ours/sample_111/pred_color.png` | 2048×400（两者一致） | baseline 0–12；ours 0–12 |

## 6. 一致性核验

| 检查项 | 结论 |
|---|---|
| 同一 DensePASS 样本 | 通过。四个方法输出按 loader 返回的真实样本名 24/68/101/111 保存，并与 input、GT 同 ID 配对。 |
| Baseline 与 Ours 分辨率 | 通过。8 张 argmax 均为 2048×400；对应 color 图同尺寸。 |
| 13 类范围 | 通过。逐图读取 argmax，所有像素值均在 0–12 内；没有改写预测类别。 |
| Color palette | 通过。Baseline 与 full Ours 均由同一正式 Cityscapes 13 类 RGB palette 渲染；逐像素验证 `pred_color == palette[pred_argmax]`。 |
| ERP 展示范围 | 通过。两者使用同一 DensePASS ERP input、同一双 pass 全宽滑窗和同一 merge；最终均展示完整 2048×400 ERP。 |
| resize | 一致。两者同样从 merge 结果以 nearest-neighbor resize 到 2048×400；input/GT 在后续组版时也应沿用既有 confirmed panel 的 2048×400 处理。 |
| crop / 左右翻转 | 未发现方法间差异。两者使用同一 dataset 和 exporter；没有额外 crop 或 flip。双 pass 是项目正式正/反滑窗推理的一部分，而不是对其中一个方法单独翻转。 |
| 覆盖已有结果 | 未覆盖。输出写入新的 `qualitative_raw/baseline` 与 `qualitative_raw/ours`；旧 clean-ablation prediction 保持不变。 |

## 7. Palette 定义

统一 palette 的类别顺序为：road、sidewalk、building、wall、fence、pole、traffic light、traffic sign、vegetation、terrain、sky、person、car；RGB 值与既有 `generate_confirmed_panels.py` 的 `PALETTE_13` 完全一致。

## 8. Figure 4 完备性结论

**已经具备正式 Figure 4 的原始组版条件。** 四个样本均具备：

1. DensePASS input；
2. 对应 DensePASS GT；
3. 正式 full-protocol OmniSAM baseline（mIoU 60.26 checkpoint）argmax 与 color prediction；
4. 正式 full-protocol Ours（mIoU 60.16 checkpoint）argmax 与 color prediction；
5. 相同样本 ID、相同 2048×400 分辨率、相同 13 类 palette 和相同双 pass ERP 推理协议。

后续 1-10b-2b 可以只做版式拼接和 LaTeX 插入，不应再使用旧 clean-ablation `A_Dtrain_Etrain` 预测，也不得用 `A_ccpm` 替代 baseline。

## 9. 执行边界

- 未运行训练、完整验证或指标重算；没有修改任何 checkpoint。
- 未修改模型核心代码、训练代码、配置文件、data list。
- 未修改 `main.tex`、`sections/*.tex`、`tables/*.tex`、`refs.bib`。
- 未生成四列拼图或 zip。
- 本任务新增内容仅为：受限导出工具、`figures/qualitative_raw/{baseline,ours}/` 下的原始预测，以及本审计文件。
