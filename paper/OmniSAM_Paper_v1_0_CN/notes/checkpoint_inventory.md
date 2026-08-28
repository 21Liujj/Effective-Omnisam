# Checkpoint Inventory and Reproducible Scope for OmniSAM Paper v1.0

> 本盘点只使用轻量文件清单、大小、索引、少量 metrics/config/list 文本和入口文件 grep 结果；未读取任何 `.pth/.pt` 权重内容，未运行训练/验证，未下载、删除、移动或覆盖 checkpoint。

## 1. Inspected Sources

| 序号 | 路径 | 是否存在 | 检查内容 | 是否用于论文范围判断 |
|---|---|---|---|---|
| 1 | `ckpt_zoo/` | 是 | 目录约 72K；含 `README.txt`、`CKPT_INDEX.txt`、`index.json`，`files/` 为归档 symlink | 是 |
| 2 | `ckpt_zoo/CKPT_INDEX.txt` | 是 | 24,689 bytes；记录 74 条 ckpt/text prototype 索引 | 是 |
| 3 | `ckpt_zoo/index.json` | 是 | 30,681 bytes；机器可读 ckpt 索引 | 是 |
| 4 | `data_pd_exp_final_2000/base_last_20260425_232941/checkpoints/` | 是 | 约 760M；含 final baseline ckpt 与 `metrics.jsonl` | 是 |
| 5 | `data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/checkpoints/` | 是 | 约 760M；含 A_Dtrain_Etrain ckpt 与 `metrics.jsonl` | 是 |
| 6 | `experiments/pl_refiner/fullruns/*/checkpoints/` | 是 | `fullruns` 约 2.1G；含 R0/R2/R2R4 checkpoint 与 metrics | 是 |
| 7 | `experiments/tecma/runs/*/checkpoints/` | 是 | `runs` 约 5.3G；含多组 Cityscapes_13cls/sam2_s ablation ckpt | 是 |
| 8 | `experiments/tecma/runs/token_budget/*/checkpoints/` | 是 | `token_budget` 约 713M；含 keep-ratio ablation ckpt | 是 |
| 9 | `logs/wildpass_results/val_ckpts/` | 是 | 约 8.4M；含 WildPASS eval logs/metrics/config/checkpoint path 文本，无大 ckpt | 是 |
| 10 | `logs/Standford2d3d_results/` | 是 | 约 4.4M；含 Stanford reports、空 summary、`MISSING_CKPT.md`、`MISSING_PATHS.md` | 是 |
| 11 | `configs/extra_exp/wildpass/` | 是 | 约 12K；含 WildPASS13 eval/train configs | 是 |
| 12 | `configs/extra_exp/standford2d3d/` | 是 | 约 12K；含 Stanford360_SPAN8 eval/train configs | 是 |
| 13 | `dataset/*_list/` | 是 | 存在 cityscapes/densepass/s2d3d_pan/s2d3d_pin/structured3d/synpass/wildpass list | 是 |
| 14 | `README.md` | 未找到 | 根目录未发现 `README.md` | 是，说明无法用 README 证明官方 ckpt 范围 |
| 15 | `validation.py` | 是 | 37,604 bytes；支持 DensePASS、WildPASS、Stanford2D3D 验证路径 | 是 |
| 16 | `train.py` | 是 | 4,297 bytes；训练入口 choices 含 Cityscapes、Stanford2D3D、SynPASS | 是 |
| 17 | `run_adaptation.py` | 是 | 49,177 bytes；UDA 入口 choices 含 Cityscapes、Stanford2D3D、SynPASS，默认 ckpt 为 `exp2/Cityscapes/best_sam2_s_10.pth` | 是 |

## 2. Local Checkpoint Inventory

| 序号 | checkpoint 路径 | 文件/目录大小 | 推测来源 | 推测 backbone | 推测数据设置 | 推测训练阶段 | 推测类别协议 | 判断依据 |
|---|---|---:|---|---|---|---|---|---|
| 1 | `exp2/Cityscapes/best_sam2_s_10.pth` -> `/root/autodl-tmp/omnisam_s.pth` | 178M | unknown / base checkpoint | sam2_s | Cityscapes source/base；用于 DensePASS UDA 默认 ckpt | source-only 或 base，待确认 | 13cls 推测 | symlink 存在；`run_adaptation.py` 默认 `--src_ckpt`/`--tgt_ckpt`；`ckpt_zoo` [041] |
| 2 | `data_pd_exp_final_2000/base_last_20260425_232941/checkpoints/Cityscapes_13cls/sam2_s/` | 760M | experiment-generated / user-trained | sam2_s | Cityscapes -> DensePASS | adapted / final baseline | Cityscapes_13cls | 含 `best_iou60.26.pth`、`last_sam2_s_13cls.pth`、`resume_sam2_s.pth`、`metrics.jsonl`；metrics val_mIoU 60.26 |
| 3 | `data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/checkpoints/Cityscapes_13cls/sam2_s/` | 760M | experiment-generated / user-trained | sam2_s | Cityscapes -> DensePASS | adapted / final A_Dtrain_Etrain | Cityscapes_13cls | 含 `best_iou60.16.pth`、`last_sam2_s_13cls.pth`、`resume_sam2_s.pth`、`metrics.jsonl`；metrics val_mIoU 60.16 |
| 4 | `experiments/pl_refiner/fullruns/R0_identity_20260625_135318/checkpoints/Cityscapes_13cls/sam2_s/` | 约 714M（文件合计） | experiment-generated | sam2_s | reused/fixed PL: DensePASS PL from `base_last.../epoch1` | ablation / pseudo-label control | Cityscapes_13cls | 含 `best_iou60.02.pth`、last、resume、metrics |
| 5 | `experiments/pl_refiner/fullruns/R2_class_balanced_20260625_135318/checkpoints/Cityscapes_13cls/sam2_s/` | 约 714M（文件合计） | experiment-generated | sam2_s | reused/fixed PL | ablation / pseudo-label refinement | Cityscapes_13cls | 含 `best_iou60.07.pth`、last、resume、metrics |
| 6 | `experiments/pl_refiner/fullruns/R2R4_proto_gate_20260625_154537/checkpoints/Cityscapes_13cls/sam2_s/` | 约 714M（文件合计） | experiment-generated | sam2_s | reused/fixed PL | ablation / pseudo-label refinement | Cityscapes_13cls | 含 `best_iou60.05.pth`、last、resume、metrics |
| 7 | `experiments/tecma/runs/*/checkpoints/Cityscapes_13cls/sam2_s/` | runs 总计约 5.3G | experiment-generated | sam2_s | Cityscapes -> DensePASS | ablation | Cityscapes_13cls | 发现 baseline、A_ccpm、A_Dtrain、A_Etrain、Dtrain_Etrain、A_Dtrain_Etrain、PLon、sanity32 等目录；每组含 `best_iou*.pth`、last、metrics |
| 8 | `experiments/tecma/runs/token_budget/*/checkpoints/Cityscapes_13cls/sam2_s/` | token_budget 总计约 713M | experiment-generated | sam2_s | Cityscapes -> DensePASS | ablation / token budget | Cityscapes_13cls | 含 `A_Etrain_keep025.../best_iou60.10.pth` 与 `A_Etrain_keep075.../best_iou60.12.pth` |
| 9 | `ckpt_zoo/files/*` | 74 条 symlink/index entries；目录本身 72K | mixed archive / symlink | mostly sam2_s | mostly Cityscapes_13cls/sam2_s | mixed | mostly 13cls | `README.txt` 说明多为 symlink；索引含 `best_sam2_s_10`、`best_iou60.26` 等 |
| 10 | `logs/wildpass_results/val_ckpts/*` | 8.4M logs，无大 `.pth` | evaluation logs | sam2_s | WildPASS13 eval using existing Cityscapes/DensePASS ckpt | evaluation-only | 13cls | `metrics.json` 中 ckpt 指向 `exp2/Cityscapes/best_sam2_s_10.pth` 或 `base_last.../best_iou60.26.pth` |
| 11 | `logs/Standford2d3d_results/val_ckpts/20260511_170002/MISSING_CKPT.md` | 110 bytes | missing marker | unknown | Stanford360 / indoor | not available | 8cls requested | 文件说明 val 需要 `--paper_ckpt` 与 `--final_ckpt`，均为 indoor 8 类 OmniSAM checkpoint |

## 3. Official / Base Checkpoint Evidence

| checkpoint | 是否本地存在 | README/索引是否提到 | 对应 backbone | 对应数据设置 | 是否可直接用于验证 | 证据 |
|---|---|---|---|---|---|---|
| `exp2/Cityscapes/best_sam2_s_10.pth` / `/root/autodl-tmp/omnisam_s.pth` | 是 | `ckpt_zoo/CKPT_INDEX.txt` [041] 提到；根 `README.md` 未找到 | sam2_s | Cityscapes source/base，类别协议待确认 | 可作为已有脚本默认 ckpt 使用；是否“official”无法确认 | symlink 存在，实际文件 178M；`run_adaptation.py` 默认 ckpt 指向该路径 |
| `data_pd_exp_final_2000/base_last.../best_iou60.26.pth` | 是 | `ckpt_zoo/CKPT_INDEX.txt` [042] 提到；非 README 官方说明 | sam2_s | Cityscapes -> DensePASS / Cityscapes_13cls | 可直接用于 DensePASS/WildPASS validation 脚本；无需重新训练 | 本地文件约 186M；metrics val_mIoU 60.26；WildPASS eval config 引用该 ckpt |
| `data_pd_exp_final_2000/A_Dtrain_Etrain.../best_iou60.16.pth` | 是 | 本地目录存在；未用 README 证明官方 | sam2_s | Cityscapes -> DensePASS / Cityscapes_13cls | 可用于本地验证；本任务未运行 | 本地文件约 186M；metrics val_mIoU 60.16 |
| `pd_auto_iter/checkpoints/.../best_sam2_s_iou60.14.pth` | 索引提到，实际路径未作为主依据逐项确认 | `ckpt_zoo/CKPT_INDEX.txt` [066] 提到 | sam2_s | Cityscapes_13cls/sam2_s | 可能可用，需单独确认现存性 | 仅索引证据；本盘点不把它作为论文主结果依据 |
| OmniSAM_s / sam2_s 官方 DensePASS 相关 ckpt | 无法确认“官方”身份；可确认本地存在 sam2_s DensePASS adapted ckpt | 索引提到多个 Cityscapes_13cls/sam2_s ckpt；根 README 缺失 | sam2_s | Cityscapes -> DensePASS | 本地可验证，但官方来源无法确认 | `ckpt_zoo` 是本地归档，不是官方发布说明 |
| SynPASS 相关官方 ckpt | 未确认 | `dataset/synpass_list` 存在；未在 checkpoint 清单中发现明确 SynPASS ckpt | unknown | SynPASS | 无法确认 | 仅有 data list 与 train/adaptation choices；未见 ckpt |
| WildPASS 相关官方 ckpt | 未确认 | 有 WildPASS eval config/log；未见 WildPASS-trained ckpt | sam2_s for evaluated ckpt | WildPASS13 eval using Cityscapes/DensePASS ckpt | 可复查已有 logs；不能证明有 WildPASS official ckpt | eval logs 的 ckpt 指向 Cityscapes/DensePASS ckpt |
| Stanford2D3D / indoor official ckpt | 未找到 | `MISSING_CKPT.md` 明确缺 ckpt | unknown | Stanford360_SPAN8 / indoor 8cls | 否 | MISSING_CKPT 指出需要 indoor 8 类 OmniSAM checkpoint |

重点回答：当前能确认存在 **sam2_s / Cityscapes_13cls / DensePASS adapted** 的本地 checkpoint；不能确认它们是“官方 DensePASS ckpt”。当前不能确认存在 SynPASS / WildPASS / Stanford2D3D / indoor 相关 official checkpoint。根目录 `README.md` 未找到，因此不应根据 README 断言官方发布范围。

## 4. Reproducible Evaluation Scope

| 设置 | 数据集 | backbone | 类别协议 | checkpoint 来源 | 是否需要重新训练 | 当前证据 | 论文 v1.0 是否可作为正式实验 |
|---|---|---|---|---|---|---|---|
| Cityscapes -> DensePASS / baseline final | DensePASS val | sam2_s | 13cls | `data_pd_exp_final_2000/base_last.../best_iou60.26.pth` | 否 | ckpt 与 metrics.jsonl 存在；DensePASS val list 100 行；validation.py 支持 DensePASS | 是，建议作为 v1.0 主实验范围 |
| Cityscapes -> DensePASS / A_Dtrain_Etrain | DensePASS val | sam2_s | 13cls | `data_pd_exp_final_2000/A_Dtrain_Etrain.../best_iou60.16.pth` | 否 | ckpt 与 metrics.jsonl 存在；DensePASS val list 100 行；memory-efficient 指标存在 | 是，建议作为 v1.0 主实验/消融范围 |
| PL refinement R0/R2/R2R4 | DensePASS val | sam2_s | 13cls | `experiments/pl_refiner/fullruns/*/best_iou*.pth` | 否 | ckpt 与 metrics.jsonl 存在；reused/fixed PL setting 已在 0-3a 记录 | 可作为探索性消融，不建议主贡献 |
| TECMA / memory-efficient ablations | DensePASS val | sam2_s | 13cls | `experiments/tecma/runs/*/best_iou*.pth` | 否 | 多个 ablation ckpt 与 metrics.jsonl 存在 | 可作为 ablation 或内部证据；正式采用需选择口径一致的 run |
| WildPASS 相关设置 | WildPASS13 val | sam2_s | 13cls | eval logs 使用 `exp2/Cityscapes/best_sam2_s_10.pth` 与 `base_last.../best_iou60.26.pth` | 否，如只复查已有 ckpt eval；本任务未运行 | `logs/wildpass_results/val_ckpts/*/metrics.json`、WildPASS13 config、500 行 val list 存在 | 建议作为补充/附录或待确认，不作为主表强结论 |
| Stanford2D3D / SPAN8 | Stanford360_SPAN8 | unknown/sam2_s config | eval config 为 8cls；train config 有 class_set 13（需澄清） | 缺 indoor 8 类 ckpt | 是/或需获得 ckpt | `MISSING_CKPT.md` 明确缺 `--paper_ckpt` 与 `--final_ckpt`；SPAN8 lists train 780 / val 260 存在 | 不建议作为 v1.0 正式定量实验；可作 future work |
| SynPASS | SynPASS | unknown | unknown / list 存在 | 未发现对应 ckpt | 需要训练或确认 ckpt | `dataset/synpass_list` 存在；`train.py`/`run_adaptation.py` choices 包含 SynPASS；未见 checkpoint | 不建议作为 v1.0 正式实验 |
| S2D3D pin/pan legacy protocol | S2D3D pin/pan | unknown | unknown | 未发现对应 ckpt | 需要训练或确认 ckpt | `dataset/s2d3d_pin_list`、`dataset/s2d3d_pan_list` 存在；无 ckpt 证据 | 不建议作为 v1.0 正式实验 |

## 5. Dataset and Class Protocol Inventory

| 数据集/协议 | 路径 | 类别数 | train/val/test list 是否存在 | 是否有 label.txt | 是否有对应 checkpoint | 论文 v1.0 用途 |
|---|---|---:|---|---|---|---|
| Cityscapes_13cls | `dataset/cityscapes_list/` | 13（由实验路径/协议推断） | `train.txt` 2975 行；`val.txt` 500 行 | 是，`label.txt` 500 行 | 是，大量 `Cityscapes_13cls/sam2_s` ckpt | 主实验源域与 checkpoint 路径协议 |
| DensePASS | `dataset/densepass_list/` | 13/19 均由代码支持；v1.0 主线用 13 | `train.txt` 2000 行；`val.txt` 100 行 | 是，`label.txt` 100 行 | 是，final baseline/A_Dtrain_Etrain/PL/TECMA ckpt | v1.0 主目标域/验证集 |
| WildPASS13 | `dataset/wildpass_list/`；`data_extra/processed/WildPASS13/lists/`；`configs/extra_exp/wildpass/` | 13 | val list 500 行；未确认 processed train list | `dataset/wildpass_list` 下未见 label.txt | 有 eval logs 使用 Cityscapes/DensePASS ckpt；未见 WildPASS-trained ckpt | 补充验证/附录，需谨慎 |
| Stanford360_SPAN8 | `data_extra/processed/Stanford360_SPAN8/lists/`；`configs/extra_exp/standford2d3d/standford360_span8_eval.yaml` | 8（eval config `num_classes: 8`） | `train.txt` 780 行；`val.txt` 260 行 | 未在本次枚举中确认 | 未找到；`MISSING_CKPT.md` 明确缺 indoor 8 类 ckpt | Future work / protocol preparation |
| S2D3D pan | `dataset/s2d3d_pan_list/` | 待确认 | `train.txt` 1040；`val.txt` 373；`trainval.txt` 1413 | 是，`label.txt` 373 行 | 未找到对应 ckpt | 不进入 v1.0 主实验 |
| S2D3D pin | `dataset/s2d3d_pin_list/` | 待确认 | `train.txt` 52903；`val.txt` 17593；`trainval.txt` 70496 | 是，`label.txt` 17593 行；另有 `name2label.json` | 未找到对应 ckpt | 不进入 v1.0 主实验 |
| SynPASS | `dataset/synpass_list/` | 待确认 | `train.txt` 5700；`val.txt` 2180；`my_val.txt` 2180；`test.txt` 1200 | 是，`label.txt` 2180 行 | 未找到对应 ckpt | Future work / potential source domain |
| Structured3D | `dataset/structured3d_list/` | 待确认 | `train.txt` 18362；`val.txt` 1775；`test.txt` 1696 | 是，`label.txt` 1775 行 | 未找到对应 ckpt | 暂不进入 v1.0 |

## 6. Missing or Unconfirmed Checkpoints

| 目标实验设置 | 是否有数据 | 是否有配置 | 是否有 checkpoint | 缺失内容 | 对论文影响 |
|---|---|---|---|---|---|
| SynPASS 扩展 | 是，`dataset/synpass_list` | 入口 choices 包含 SynPASS；未见专门实验 config | 未确认 | 缺 SynPASS source/adapted ckpt、metrics、protocol 说明 | 不能声称 SynPASS 泛化或跨源域实验完成 |
| WildPASS 完整定量验证 | 是，WildPASS13 val 500 行与 eval config/log 存在 | 是，`configs/extra_exp/wildpass/*` | 有 DensePASS/Cityscapes ckpt 的 WildPASS eval logs；未见 WildPASS-trained ckpt | 缺清晰 final paper protocol、完整可复核主表口径；WildPASS metrics 口径需确认 | 可作补充或待确认，不宜主表强结论 |
| Stanford2D3D / indoor protocol | 是，SPAN8 lists 存在 | 是，eval/train configs 存在 | 未找到；MISSING_CKPT 明确缺 | 缺 indoor 8 类 `paper_ckpt` 和 `final_ckpt`，缺完整 eval metrics | 只能写 future work / limitation |
| 多 backbone | 未见 sam2_b/sam2_l checkpoint 证据 | 入口有 backbone 参数但 checkpoint 清单主要为 sam2_s | 未确认 | 缺 sam2_b/sam2_l ckpt 和 metrics | v1.0 不应声称多 backbone 完成 |
| source-only vs adapted 对照 | base ckpt `best_sam2_s_10.pth` 存在；adapted ckpt 存在 | 有入口默认 ckpt | 部分有 | source-only 在 DensePASS 的正式 metrics 需确认；WildPASS paper_ckpt logs 存在但不等于 DensePASS 主表 | 若论文需要 source-only 主对照，应单独确认已有 metrics 或补验证 |
| R6_memory_plq | 不适用 | 有 `experiments/pl_refiner/configs/R6_memory_plq.yaml`（0-3a 已确认） | 无完成结果 | 缺实现与 ckpt | 只能 TODO/Future Work |
| official OmniSAM 多数据集 ckpt | 不清楚 | 根 README 缺失；ckpt_zoo 是本地归档 | 无法确认 | 缺 official index 或 README 证据 | 不应写“official 只/已开源某某 ckpt” |

## 7. Implication for Paper v1.0

- 当前 v1.0 主实验应以 **Cityscapes -> DensePASS / sam2_s / 13cls** 为主，因为该设置下本地有完整 checkpoint、metrics、DensePASS train/val list 与多组 ablation 结果。
- WildPASS / Stanford2D3D / SynPASS 等设置虽然存在数据 list、配置或部分评估日志，但缺少可确认的完整 checkpoint 与统一 protocol，应作为补充实验、后续实验或 limitation。
- 不应声称已经完成多数据集泛化；当前证据更适合支持“主实验在 DensePASS 上完成，其他数据集协议正在补充”。
- 不应声称 OmniSAM 只开源了 DensePASS ckpt，除非后续找到 README/official index 明确支持；本次根目录 `README.md` 未找到，`ckpt_zoo` 只是本地 checkpoint 归档。
- 更安全的论文表述是：**“当前 v1.0 实验主要基于当前本地可确认 checkpoint 与可复现实验设置展开，重点评估 Cityscapes -> DensePASS 的 sam2_s / 13 类协议。”**

## 8. Paper-ready Limitation Paragraph Draft

当前实验主要围绕 Cityscapes -> DensePASS 的 sam2_s / 13 类协议展开，这是因为该设置下本地 checkpoint、伪标签、验证指标和消融记录最完整，可复现实验链路最清晰。对于 WildPASS、SynPASS 以及 Stanford360/SPAN8 等更广泛数据集，目前仓库中虽包含部分数据协议、配置或评估日志，但缺少统一确认的完整 checkpoint 与多 seed 结果。因此，本文 v1.0 暂不将多数据集泛化作为强结论，后续将补充更多公开 checkpoint、室内/室外场景协议和跨数据集验证，以进一步评估方法的泛化能力。

## 9. Recommended Next Experiments

| 优先级 | 实验 | 是否需要新训练 | 预计成本 | 对论文价值 | 是否建议 1.0 前完成 |
|---|---|---|---|---|---|
| 必须补 | 确认 DensePASS source-only vs adapted 主对照 metrics 是否已有可引用文件；若已有仅整理，不重跑 | 不一定 | 低 | 保证主表 baseline 口径完整 | 建议完成 |
| 必须补 | 固化 Cityscapes -> DensePASS / sam2_s / 13cls 的 checkpoint-to-metrics 对应表 | 否 | 低 | 提高可复现性，避免 ckpt 混用 | 建议完成 |
| 建议补 | WildPASS13 使用已存在 `base_last`/`A_Dtrain_Etrain` ckpt 的统一评估口径复核 | 不需要训练；可能需要验证（本任务未运行） | 中 | 可作为补充泛化实验 | 视时间决定；若不跑则只写已有日志为初步结果 |
| 建议补 | Stanford360_SPAN8 indoor checkpoint 获取或训练前的 protocol cleanup | 可能需要 | 中-高 | 支持室内全景泛化 | 不建议 v1.0 前强行完成，除非 ckpt 已可用 |
| 时间充足再补 | SynPASS source/adaptation 实验 | 需要 | 高 | 扩展跨源域/合成数据实验 | 不建议 v1.0 前完成 |
| 时间充足再补 | 多 backbone（sam2_b/sam2_l）对比 | 需要 | 高 | 增强模型规模分析 | 不建议 v1.0 前完成 |
| 暂不建议 | R6_memory_plq + MTC/SMR full combination | 需要实现与训练 | 高 | 对 pseudo-label 线有价值，但不影响 memory-efficient 主结论 | v1.0 可作为 Future Work |
| 暂不建议 | WildPASS/Stanford/SynPASS 全部补齐成主表 | 需要大量训练/验证 | 很高 | 泛化结论强，但成本与风险大 | 不建议 v1.0 前完成 |
