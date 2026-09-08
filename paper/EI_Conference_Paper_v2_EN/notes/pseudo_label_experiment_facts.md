# Pseudo-label Refinement Experiment Facts for Paper v1.0

## 1. Inspected Sources

| 序号 | 文件路径 | 是否存在 | 读取目的 | 是否用于论文事实 |
|---|---|---|---|---|
| 1 | `experiments/pl_refiner/tables/pl_fullrun_main_metrics.csv` | 是 | 读取 R0/R2/R2R4 主指标、训练步数、验证步数、耗时、显存与 done 字段 | 是，作为主结果表事实来源 |
| 2 | `experiments/pl_refiner/tables/pl_fullrun_diag_summary.csv` | 是 | 读取 refiner/prototype 诊断记录、有效比例、ignore 比例、loss 与 prototype 相似度统计 | 是，作为诊断/分析事实来源 |
| 3 | `experiments/pl_refiner/tables/pl_fullrun_per_class_iou.csv` | 是 | 读取 R0/R2/R2R4 per-class IoU 与差值 | 是，作为 per-class 初步分析来源 |
| 4 | `experiments/pl_refiner/notes/step_02_identity_refiner_report.md` | 是 | 确认 `PseudoLabelRefiner` 插入点、`off`/`identity` 模式语义、是否改动 pseudo label/loss | 是，作为模块与工程语义来源 |
| 5 | `experiments/pl_refiner/notes/step_03_config_dryrun_report.md` | 是 | 确认 R0-R7 配置意图、R1/R6/R7 状态、dry-run 约束和未实现标记 | 是，作为配置/状态来源；注意该报告早于后续 R2/R2R4 fullrun |
| 6 | `experiments/pl_refiner/fullruns/R0_identity_20260625_135318/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | 是 | 读取 R0_identity 的验证指标、reuse_pl_dir、step/epoch/memory 记录 | 是，作为 R0 数字来源 |
| 7 | `experiments/pl_refiner/fullruns/R2_class_balanced_20260625_135318/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | 是 | 读取 R2_class_balanced 的验证指标、reuse_pl_dir、step/epoch/memory 记录 | 是，作为 R2 数字来源 |
| 8 | `experiments/pl_refiner/fullruns/R2R4_proto_gate_20260625_154537/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | 是 | 读取 R2R4_proto_gate 的验证指标、reuse_pl_dir、step/epoch/memory 记录 | 是，作为 R2R4 数字来源 |
| 9 | `experiments/pl_refiner/configs/R6_memory_plq.yaml` | 是 | 确认 R6_memory_plq 配置、是否实现、是否有 full training 许可 | 是，作为 TODO/limitation 来源 |
| 10 | `adaptation/pseudo_label/prototype_gate.py` | 是 | 确认 prototype gate 的工程名称和大致功能 | 是，仅用于模块功能描述，不用于新增实验数字 |
| 11 | `adaptation/pseudo_label/prototype_bank.py` | 是 | 确认 prototype bank 当前是否可用 | 是，仅用于模块状态描述 |
| 12 | `adaptation/plugins/pseudo_conf_mask.py` | 是 | 确认 pseudo_conf_mask 的工程名称和大致功能 | 是，仅用于模块功能描述 |

## 2. Confirmed Experimental Setting

- 基础模型：待确认。候选结果路径均位于 `checkpoints/Cityscapes_13cls/sam2_s/`，可确认使用 `sam2_s` 实验目录；是否完整对应 OmniSAM/SAM2.1 small 需从非本任务文件确认。
- backbone：`sam2_s`（来源：3 个 `metrics.jsonl` 文件路径；`pl_fullrun_main_metrics.csv` 也记录相同 run 对应结果）。
- 源域：`Cityscapes_13cls`（来源：3 个 `metrics.jsonl` 文件路径）。
- 目标域：`DensePASS`（来源：3 个 `metrics.jsonl` 中 `reuse_pl_dir = data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1`）。
- 类别协议：13 类（来源：`Cityscapes_13cls` 路径；`R6_memory_plq.yaml` 中 `num_classes: 13`）。
- 训练数据规模：候选 summary 只确认 `train_steps=500/500`（来源：`pl_fullrun_main_metrics.csv`）；是否对应 DensePASS 2000 张 full protocol 需待确认。
- 验证集规模：候选 summary 只确认 `val_steps=100/100`（来源：`pl_fullrun_main_metrics.csv`）；是否即 DensePASS val 100 张需待确认。
- seed：待确认。候选文件未记录 seed。
- 评价指标：mIoU、mAcc、PixAcc、FWIoU、mDice、BFScore（来源：`pl_fullrun_main_metrics.csv` 表头；3 个 `metrics.jsonl` 含 mIoU/mAcc/PixAcc/FWIoU/mDice）。
- 是否与 memory-efficient 线使用同一实验协议：部分可确认、整体待确认。`pl_fullrun_main_metrics.csv` 的 `Memory-efficient` 列为 `yes`，方法名为 `Efficient + ...`；但 3 个 `metrics.jsonl` 中 `memory_tokens_before=4096.0`、`memory_tokens_after=4096.0`、`memory_token_ratio=1.0`，未显示 4096→1024 压缩。因此只能保守写为：伪标签线在 summary 中标注为 memory-efficient 相关设置，但是否等同 v1.0 memory-efficient 主线完整配置仍待确认。
- 是否 fixed-PL / reused-PL / full training，请尽量确认：
  - 可确认是 reused-PL / fixed-PL 训练：3 个 `metrics.jsonl` 均记录相同 `reuse_pl_dir = data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1`，且 PL generation 相关字段为 `NA`。
  - 可确认完成了 `train_steps=500/500` 与 `val_steps=100/100`（来源：`pl_fullrun_main_metrics.csv`）。
  - 是否可称为 full protocol：待确认。`pl_fullrun_main_metrics.csv` 文件名含 `fullrun`，但 `done` 列为 `False`，因此论文中不应直接称“最终完整实验已完成”。

## 3. Pseudo-label Modules Inventory

| 工程名称 | 论文建议名称 | 作用 | 来源文件 | 是否已参与实验 |
|---|---|---|---|---|
| `R0_identity` / `mode=identity` | Identity PL Control / 原始伪标签对照 | 不改变 `frame_pl_t` 与 `valid_mask`；用于验证 refiner 插入点不会改变 pseudo label 或 loss 路径 | `step_02_identity_refiner_report.md`；R0 `metrics.jsonl`；`pl_fullrun_main_metrics.csv` | 是。R0 结果存在：mIoU 60.02（来源：R0 `metrics.jsonl` / `pl_fullrun_main_metrics.csv`） |
| `R2_class_balanced` | Class-balanced Pseudo-label Refinement (CBPR) | 按类别平衡策略筛选/保留伪标签，目标是改善稀有类有效监督比例；具体实现细节未在本任务指定代码文件中展开 | `pl_fullrun_main_metrics.csv`；`pl_fullrun_diag_summary.csv`；`step_03_config_dryrun_report.md` | 是。R2 结果存在：mIoU 60.07（来源：R2 `metrics.jsonl` / `pl_fullrun_main_metrics.csv`）。注意 `step_03` 早期报告曾标记 R2 为 NOT_IMPLEMENTED_YET，后续 fullrun 已有结果 |
| `R2R4_proto_gate` | CBPR + Prototype-guided Reliability Gate (PRG) | 在 R2 class-balanced 基础上加入 current-batch prototype similarity gate，用特征-类别原型相似度进一步过滤低可靠 pseudo label | `prototype_gate.py`；`pl_fullrun_main_metrics.csv`；`pl_fullrun_diag_summary.csv` | 是。R2R4 结果存在：mIoU 60.05（来源：R2R4 `metrics.jsonl` / `pl_fullrun_main_metrics.csv`） |
| `pseudo_conf_mask` | Class-adaptive Confidence Thresholding (CCT) 或 Confidence Masking | 根据置信度阈值保留 pseudo label 像素，并计算 masked unsupervised CE；代码支持阈值、最小有效像素/比例和 fallback | `adaptation/plugins/pseudo_conf_mask.py`；`step_03_config_dryrun_report.md` | 已有独立配置/既有实验入口。`step_03` 记录 `R1_conf_mask` 为论文消融配置，并提到对应 A_ccpm clean ablation；但本次 `pl_fullrun_main_metrics.csv` 未列 R1，因此本文件不将 R1 数字并入主 PL 结果 |
| `prototype_gate` | Prototype-guided Reliability Gate (PRG) | 计算当前 batch 类原型、每位置 cosine similarity、class-wise percentile keep/filter mask，并上采样到 label 分辨率后作用于 valid mask | `adaptation/pseudo_label/prototype_gate.py` | 是，作为 R2R4 组合的一部分参与实验（来源：`pl_fullrun_main_metrics.csv` 与 `pl_fullrun_diag_summary.csv` 的 R2R4 行） |
| `prototype_bank` | Prototype Bank（暂不建议作为论文正式模块名） | 当前文件仅为 placeholder，`PROTOTYPE_BANK_AVAILABLE = False`；未确认有可用持久化 prototype bank 实现 | `adaptation/pseudo_label/prototype_bank.py` | 否。不能声称 prototype bank 已作为独立模块完成实验 |
| `R6_memory_plq` | Pseudo-label Quality Refinement (PQR) / Memory-aware PLQ（待确认） | 配置意图是使用 cross-frame memory consistency 增强 PLQ score；`memory_consistency.enable: true`，`consistency_weight: 0.3`，`memory_queue_size: 9` | `experiments/pl_refiner/configs/R6_memory_plq.yaml` | 否。配置明确写明 `status: NOT_IMPLEMENTED_YET`，`full_train_allowed: conditional`，无本任务候选结果 |

## 4. Main Pseudo-label Results

| 实验名称 | 论文建议名称 | mIoU | mAcc | PixAcc | FWIoU | mDice | 关键配置 | 来源文件 | 是否可写入论文 |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| R0_identity | Identity PL Control | 60.02 | 71.71 | 88.58 | 80.66 | 72.59 | `PL Refiner=identity`；`reuse_pl_dir=data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1`；`train_steps=500/500`；`val_steps=100/100` | `pl_fullrun_main_metrics.csv`；R0 `metrics.jsonl` | 可作为对照写入，但需标注 reused/fixed PL 与单次 run；`done=False` 需谨慎 |
| R2_class_balanced | Class-balanced Pseudo-label Refinement (CBPR) | 60.07 | 72.28 | 88.55 | 80.67 | 72.66 | `PL Refiner=class-balanced`；同一 `reuse_pl_dir`；`train_steps=500/500`；`val_steps=100/100` | `pl_fullrun_main_metrics.csv`；R2 `metrics.jsonl` | 可写为初步消融结果；相对 R0 的 mIoU +0.05、mAcc +0.57、mDice +0.07 均来自 `pl_fullrun_main_metrics.csv`，但幅度很小，不宜声称显著提升 |
| R2R4_proto_gate | CBPR + Prototype-guided Reliability Gate (PRG) | 60.05 | 72.24 | 88.54 | 80.64 | 72.64 | `PL Refiner=class-balanced + proto-gate`；同一 `reuse_pl_dir`；`train_steps=500/500`；`val_steps=100/100` | `pl_fullrun_main_metrics.csv`；R2R4 `metrics.jsonl` | 可写为初步组合消融；相对 R2 mIoU -0.02（来源：`pl_fullrun_main_metrics.csv` 的 R2R4 `d_mIoU`），相对 R0 约 +0.03（由同表 R2R4 60.05 与 R0 60.02 直接比较），不宜声称带来最终性能增益 |

补充：`pl_fullrun_main_metrics.csv` 还给出 BFScore：R0 39.35、R2 39.54、R2R4 39.57；但用户指定主表不要求 BFScore 列，若论文使用该数字，必须明确引用该 CSV，且需确认 BFScore 口径。

## 5. Diagnostic / Pseudo-label Quality Evidence

| 实验名称 | 诊断指标 | 数值 | 含义 | 来源文件 | 是否可写入论文分析 |
|---|---|---:|---|---|---|
| R0_identity | `refiner_records` | 4500 | 记录到的 refiner 诊断条数；R0 为 identity control | `pl_fullrun_diag_summary.csv` | 可放 notes/appendix；正文不必强调 |
| R0_identity | `proto_records` | 0 | 未记录 prototype 诊断，符合 identity control | `pl_fullrun_diag_summary.csv` | 可用于说明 R0 未使用 prototype gate |
| R0_identity | `mean_pseudo_vr` | 0.9908 | summary 中的 pseudo valid ratio 均值；具体口径需结合 logging 确认 | `pl_fullrun_diag_summary.csv` | 谨慎使用，建议只放 notes |
| R0_identity | `mean_loss_unsup` | 0.0194 | unsupervised loss 均值统计 | `pl_fullrun_diag_summary.csv` | 可作为内部诊断，不建议作为正文性能结论 |
| R2_class_balanced | `refiner_records` | 4500 | class-balanced refiner 诊断记录数 | `pl_fullrun_diag_summary.csv` | 可写入 appendix/analysis |
| R2_class_balanced | `proto_records` | 4500 | 记录了 prototype 相关诊断；说明 R2 run 中已有 prototype statistics logging | `pl_fullrun_diag_summary.csv` | 可写入 notes；若正文使用需解释 proto_records 与实际 gate 的关系 |
| R2_class_balanced | `mean_total_vr` | 0.4884 | class-balanced 后总体有效/保留比例均值 | `pl_fullrun_diag_summary.csv` | 可作为“筛选更严格”的诊断证据 |
| R2_class_balanced | `mean_common_vr` | 0.4852 | common 类有效/保留比例均值 | `pl_fullrun_diag_summary.csv` | 可用于类别均衡分析，但不宜夸大 |
| R2_class_balanced | `mean_rare_vr` | 0.6807 | rare 类有效/保留比例均值，高于 common 类 | `pl_fullrun_diag_summary.csv` | 可作为“稀有类保留更充分”的初步证据 |
| R2_class_balanced | `mean_ext_rare_vr` | 0.6632 | extreme rare 类有效/保留比例均值 | `pl_fullrun_diag_summary.csv` | 可作为 appendix 诊断 |
| R2_class_balanced | `mean_ignore_r` | 0.5116 | 被 ignore/过滤比例均值 | `pl_fullrun_diag_summary.csv` | 可写为过滤强度诊断 |
| R2_class_balanced | `mean_loss_unsup` | 0.0068 | unsupervised loss 均值低于 R0 的 0.0194 | `pl_fullrun_diag_summary.csv` | 只可作为训练诊断，不能直接解释为性能提升 |
| R2_class_balanced | `proto_sim_mean` / `proto_sim_std` | 0.8793 / 0.0709 | prototype similarity 均值/标准差 | `pl_fullrun_diag_summary.csv` | 可用于说明 pseudo label 与当前 batch prototype 的一致性统计 |
| R2_class_balanced | `proto_valid_ratio` / `proto_nan` | 0.4893 / 0.0000 | prototype 有效比例与 NaN 计数 | `pl_fullrun_diag_summary.csv` | 可作为稳定性诊断 |
| R2R4_proto_gate | `refiner_records` | 4500 | R2R4 refiner 诊断记录数 | `pl_fullrun_diag_summary.csv` | 可写入 appendix/analysis |
| R2R4_proto_gate | `proto_records` | 4500 | R2R4 prototype 诊断记录数 | `pl_fullrun_diag_summary.csv` | 可写入 appendix/analysis |
| R2R4_proto_gate | `mean_total_vr` | 0.4478 | proto-gate 后总体有效/保留比例低于 R2 的 0.4884，说明过滤更强 | `pl_fullrun_diag_summary.csv` | 可写为诊断证据，但不能声称提升 mIoU |
| R2R4_proto_gate | `mean_common_vr` | 0.4853 | common 类有效/保留比例均值 | `pl_fullrun_diag_summary.csv` | 可用于类别组分析 |
| R2R4_proto_gate | `mean_rare_vr` | 0.6806 | rare 类有效/保留比例均值 | `pl_fullrun_diag_summary.csv` | 可用于类别组分析 |
| R2R4_proto_gate | `mean_ext_rare_vr` | 0.6641 | extreme rare 类有效/保留比例均值 | `pl_fullrun_diag_summary.csv` | 可用于 appendix |
| R2R4_proto_gate | `mean_ignore_r` | 0.5522 | 过滤/ignore 比例高于 R2 的 0.5116 | `pl_fullrun_diag_summary.csv` | 可写为 prototype gate 增强过滤的诊断证据 |
| R2R4_proto_gate | `mean_loss_unsup` | 0.0054 | unsupervised loss 均值低于 R2 的 0.0068 | `pl_fullrun_diag_summary.csv` | 只可作为训练诊断，不能直接声称最终精度提高 |
| R2R4_proto_gate | `proto_sim_mean` / `proto_sim_std` | 0.8789 / 0.0708 | prototype similarity 均值/标准差，与 R2 接近 | `pl_fullrun_diag_summary.csv` | 可用于说明 gate 运行稳定 |
| R2R4_proto_gate | `proto_valid_ratio` / `proto_nan` | 0.4893 / 0.0000 | prototype 有效比例与 NaN 计数 | `pl_fullrun_diag_summary.csv` | 可作为稳定性诊断 |

## 6. Per-class IoU Evidence

| 实验名称 | 相比 R0 提升明显的类别 | 相比 R0 下降明显的类别 | 是否建议写入论文分析 |
|---|---|---|---|
| R2_class_balanced | `wall` +0.70，`pole` +0.34，`traffic light` +0.29，`fence` +0.25（来源：`pl_fullrun_per_class_iou.csv`） | `person` -0.48，`traffic sign` -0.21，`sidewalk` -0.19（来源：`pl_fullrun_per_class_iou.csv`） | 可作为初步 per-class 观察，但幅度小且单 seed/单 run，建议放 appendix 或讨论，不建议作为主结论 |
| R2R4_proto_gate | `wall` +0.71，`traffic light` +0.32，`pole` +0.30，`fence` +0.27（来源：`pl_fullrun_per_class_iou.csv`） | `person` -0.42，`sidewalk` -0.41，`traffic sign` -0.26（来源：`pl_fullrun_per_class_iou.csv`） | 可写为“少数结构/小物体类有轻微改善、行人与人行道下降”的观察；需避免过度解释 |
| R2R4_proto_gate vs R2_class_balanced | R2R4 相比 R2 仅 `person` +0.06、`traffic light` +0.03、`fence` +0.02、`car` +0.01、`wall` +0.01（来源：`pl_fullrun_per_class_iou.csv`） | `sidewalk` -0.22、`terrain` -0.06、`traffic sign` -0.05、`road` -0.04、`pole` -0.04（来源：`pl_fullrun_per_class_iou.csv`） | 不建议在正文强调 R4 带来显著 per-class 改善；更适合作为 limitation/TODO |

说明：本节只总结最值得注意的类别，不列全 13 类。per-class 数据来源明确，但统计口径仍建议在论文最终表格前复核。

## 7. Comparison with Memory-efficient Main Results

| 项目 | Memory-efficient 线 | Pseudo-label 线 | 论文中如何处理 |
|---|---|---|---|
| 主贡献定位 | 已有明确主线：MTC/SMR 等 memory-efficient 方向以训练速度/显存/attention 加速为核心，并已形成论文可用结果 | 当前 PL 线 R2/R2R4 mIoU 分别为 60.07/60.05，R0 为 60.02（来源：`pl_fullrun_main_metrics.csv`），提升极小 | 伪标签优化不宜替代 memory-efficient 作为主贡献；可作为辅助模块或附加消融 |
| 精度收益 | memory-efficient 主线已有独立汇总，不在本文件混用数字 | PL 线当前只能显示 mIoU 基本持平，R2 相对 R0 +0.05，R2R4 相对 R2 -0.02（来源：`pl_fullrun_main_metrics.csv`） | 论文中应保守写“保持精度/初步改善监督可靠性”，不要写“显著提升 mIoU” |
| 诊断收益 | memory-efficient 线重点是 memory tokens/attention time/step time | PL 线诊断显示 R2/R2R4 对有效比例、ignore 比例、rare/common 保留比例和 prototype similarity 有可记录统计（来源：`pl_fullrun_diag_summary.csv`） | 可在 Method/Discussion 中说明其目标是提高目标域监督信号可靠性；正文数字建议少量使用，更多放 appendix/notes |
| 与 MTC+SMR 组合 | memory-efficient 主线已有最终配置，但本文件不复述其数字 | `pl_fullrun_main_metrics.csv` 标注 `Memory-efficient=yes`，但 `metrics.jsonl` 记录 `memory_tokens_before=4096.0`、`memory_tokens_after=4096.0`、`memory_token_ratio=1.0`，组合关系未完全清楚 | 不能声称已完成与最终 MTC+SMR 的完整组合实验；需写“组合实验待确认/待补充” |
| R6_memory_plq | 不适用 | `R6_memory_plq.yaml` 存在，但明确 `status: NOT_IMPLEMENTED_YET`、`full_train_allowed: conditional`，无结果 | 只能作为 TODO/Future Work，不应进入主表 |
| 论文 v1.0 风险 | 主线结果更稳 | PL 线缺少多 seed、R6/full 组合、最终 full protocol 确认 | PL 线建议作为辅助实验和 limitation，而不是 Abstract 的定量卖点 |

## 8. Paper-ready Claims

1. 在 reused/fixed PL 设置下，`R2_class_balanced` 相比 `R0_identity` 取得非常小的 mIoU 变化：60.07 vs 60.02（来源：`pl_fullrun_main_metrics.csv` 与对应 `metrics.jsonl`）。这只能表述为“初步结果显示精度基本持平并略有改善”，不能表述为显著提升。
2. `R2R4_proto_gate` 在 R2 基础上进一步过滤 pseudo label，使 `mean_total_vr` 从 0.4884 降到 0.4478、`mean_ignore_r` 从 0.5116 升到 0.5522（来源：`pl_fullrun_diag_summary.csv`），说明 prototype gate 的主要可见效果是提高过滤强度，而非当前已证明提升最终 mIoU。
3. class-balanced 策略对 rare/extreme-rare 类保留比例较高：R2 的 `mean_rare_vr=0.6807`、`mean_ext_rare_vr=0.6632`，高于 `mean_common_vr=0.4852`（来源：`pl_fullrun_diag_summary.csv`）。可保守写作“用于改善目标域伪标签监督的类别覆盖/可靠性”。
4. per-class 初步结果显示 `wall`、`pole`、`traffic light`、`fence` 等类别相对 R0 有小幅改善，但 `person`、`sidewalk`、`traffic sign` 有下降（来源：`pl_fullrun_per_class_iou.csv`）。这适合写成分析/讨论，不适合作为强结论。
5. 当前结果更适合作为 memory-efficient 主线的补充模块：PL refinement 有诊断收益和小幅/持平精度表现，但 R6_memory_plq 与完整组合仍未完成（来源：`R6_memory_plq.yaml` 与 `pl_fullrun_main_metrics.csv`）。

## 9. Results Not Safe to Claim Yet

- 不能声称 pseudo-label refinement 显著提升 mIoU。当前 R2 相对 R0 仅 +0.05 mIoU，R2R4 相对 R2 为 -0.02 mIoU（来源：`pl_fullrun_main_metrics.csv`），均属于非常小的变化。
- 不能声称 prototype gate 已带来最终精度收益。R2R4 的 mIoU 60.05 低于 R2 的 60.07（来源：`pl_fullrun_main_metrics.csv` 与 R2/R2R4 `metrics.jsonl`）。
- 不能声称 R6_memory_plq 已完成。`R6_memory_plq.yaml` 明确写明 `status: NOT_IMPLEMENTED_YET`、`full_train_allowed: conditional`。
- 不能声称已完成和 memory-efficient 最终主配置的完整组合实验。summary 虽标记 `Memory-efficient=yes`，但 metrics 中 memory token 记录为 4096.0→4096.0，组合关系需进一步确认。
- 不能声称多 seed 稳定性。候选文件未记录 seed，也未提供多 seed 结果。
- 不能声称 full protocol 最终主表结果已安全定稿。`pl_fullrun_main_metrics.csv` 的 `done` 列为 `False`，且 fixed/reused PL 与 full protocol 关系仍需确认。
- 不能把 pseudo-label 线数字与 memory-efficient 主线数字混用。PL 线数字必须来自本文件列出的 PL summary/metrics；memory 主线数字应来自 memory-efficient 专门汇总。
- 不能声称 `prototype_bank` 已实现或参与实验。`prototype_bank.py` 仅为 placeholder，`PROTOTYPE_BANK_AVAILABLE=False`。

## 10. Recommended Use in Paper v1.0

| 论文位置 | 是否建议使用伪标签线内容 | 使用方式 |
|---|---|---|
| Abstract | 暂不建议写定量结果 | 不写 pseudo-label refinement 的 mIoU 提升；最多在非常概括的贡献句中提到“可靠伪标签过滤/质量控制”作为辅助思想，但若篇幅紧张建议不写 |
| Introduction | 可少量使用 | 可作为 UDA 中目标域伪标签噪声问题的动机；不要写已经显著解决该问题 |
| Method | 建议使用 | 可正式描述 `pseudo_conf_mask`、CBPR、PRG 的设计思想与插入点；需明确这是 refiner/filtering 模块，不改变目标域标签使用协议 |
| Experiments | 谨慎使用 | 可放一张小型 ablation 或 appendix table，列 R0/R2/R2R4 的 mIoU/mAcc/PixAcc/FWIoU/mDice；需标注 reused/fixed PL、单 run、`done=False` 风险 |
| Ablation | 建议作为辅助消融 | R0/R2/R2R4 可以作为“pseudo-label refinement 初步消融”；结论写“基本保持精度，R2 略有改善，R4 需继续调参/验证” |
| Discussion / Future Work | 强烈建议使用 | 明确写 R6_memory_plq、full combination、多 seed、full protocol 仍是 TODO；讨论诊断收益和 mIoU 未显著提升之间的差距 |

明确建议：

- 可以正式写：R0/R2/R2R4 的初步消融数字；R2/R2R4 的诊断指标；prototype gate 当前更体现过滤强度而不是已证明性能提升。
- 只能占位 TODO：R6_memory_plq、R7_full、与最终 MTC+SMR 的完整组合、多 seed 稳定性、full protocol 最终主表。
- 暂时不要写入正文：任何“pseudo-label refinement 显著提升 mIoU”的表述；任何“prototype bank 已完成”的表述；任何未由 `pl_fullrun_main_metrics.csv`、`pl_fullrun_diag_summary.csv`、`pl_fullrun_per_class_iou.csv` 或对应 `metrics.jsonl` 支撑的数字。
