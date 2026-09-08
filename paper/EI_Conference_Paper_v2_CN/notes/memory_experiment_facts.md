# Memory-efficient Experiment Facts for Paper v1.0

## 1. Inspected Sources

| 序号 | 文件路径 | 是否存在 | 读取目的 | 是否用于论文事实 |
|---|---|---|---|---|
| 1 | `experiments/tecma/tables/clean_ablation_results.csv` | 是 | 读取 clean ablation 中 baseline、A/D/E、D+E、A+D+E 的 mIoU、BFScore、memory tokens、attention time、GPU peak 等字段 | 是，主要用于 memory 消融和机制表；注意这是 fixed-PL clean ablation，不等同于 full final_2000 主结果 |
| 2 | `experiments/tecma/tables/token_budget_tradeoff.csv` | 是 | 读取 E-only token budget curve 与 final D+E endpoint 的 token、mIoU、BFScore、step time、attention time | 是，用于 token budget trade-off 表 |
| 3 | `experiments/tecma/tables/training_time_breakdown.csv` | 是 | 读取 per-iter step time、iter compute time、memory attention per-iter 换算字段、n_iters | 是，用于效率分解；发现 token_budget 部分 CSV 行存在列错位风险，优先采用报告和 token_budget_tradeoff.csv |
| 4 | `experiments/tecma/tables/memory_token_attention_cost.csv` | 是 | 读取 ablation + token budget 合并后的 memory token、attention cost、compression/sparse token 字段 | 是，主要用于 token 配置、attention cost 和消融事实 |
| 5 | `experiments/tecma/tables/paper_ablation_table_draft.md` | 是 | 读取已经整理过的论文消融表草稿与注意事项 | 是，用于交叉核对 clean ablation；其中结论文字仅作为参考，不直接当作新增实验事实 |
| 6 | `experiments/tecma/notes/step_09_main_results_draft_report.md` | 是 | 读取 full DensePASS protocol 主结果来源说明、single seed 限制、主结果变化量 | 是，但该文件只给出主结果变化量/Speedup，不给出全部绝对值；绝对数需后续从主结果表或原始 metrics 再确认 |
| 7 | `experiments/tecma/notes/step_15_token_budget_results_report.md` | 是 | 读取 token budget curve 的报告化结果、PL 复用说明、风险限制 | 是，用于 token budget trade-off 和安全表述 |
| 8 | `experiments/tecma/notes/step_16_profile_summary_report.md` | 是 | 读取 memory_tokens_scope、D/E token 链路、attention time 与显存说明 | 是，用于解释 token 统计口径和效率来源 |
| 9 | `experiments/tecma/notes/step_17_training_time_breakdown_report.md` | 是 | 读取 step time、memory attention time 口径、N_iters=500、per-iter 换算和非单调解释 | 是，用于效率计算和论文限制说明 |
| 10 | `experiments/tecma/runs/baseline_omnisam_seed1_20260616_102834/run.log` | 是 | 核对 baseline fixed-PL run 的插件状态、PL 复用、训练迭代数、ValSummary BFScore、val mIoU | 是，用于 clean ablation baseline 的日志证据 |
| 11 | `experiments/tecma/runs/A_Dtrain_Etrain_seed1_20260616_121135/run.log` | 是 | 核对 A+D+E fixed-PL run 的插件状态、keep_ratio、PL 复用、ValSummary BFScore、val mIoU | 是，用于 clean ablation full TeCMA endpoint 的日志证据 |
| 12 | `experiments/tecma/runs/token_budget/A_Etrain_keep025_seed1_20260617_114629/run.log` | 是 | 核对 keep_ratio=0.25 run 的插件状态、PL 复用、ValSummary BFScore、val mIoU | 是，用于 token budget 25% 点日志证据 |
| 13 | `experiments/tecma/runs/token_budget/A_Etrain_keep075_seed1_20260617_112455/run.log` | 是 | 核对 keep_ratio=0.75 run 的插件状态、PL 复用、ValSummary BFScore、val mIoU | 是，用于 token budget 75% 点日志证据 |
| 14 | `configs/ablations/omnisam_base_plus4_efficienttam.yaml` | 是 | 读取 D / efficient_tam_memory 配置开关、pool、compress_ratio、apply_in_pl/apply_in_train | 是，用于模块配置说明；注意该配置是独立 D 配置，不一定等同于最终 train-only 配置 |
| 15 | `configs/ablations/omnisam_base_plus5_efficientsam2_sparse.yaml` | 是 | 读取 E / efficient_sam2_sparse 配置开关、keep_ratio、apply_in_pl/apply_in_train | 是，用于模块配置说明；注意该配置是独立 E 配置，不一定等同于最终 train-only 配置 |
| 16 | `model/modules/memory_compressor.py` | 是 | 读取 SpatialMemoryCompressor 的实现形态、pool2d、token shape 与统计字段 | 是，用于确认方法实现位置和机制，不用于实验数字 |
| 17 | `model/modules/sparse_memory_retriever.py` | 是 | 读取 SparseMemoryRetriever 的 top-k、feature_norm score、keep_ratio、统计字段 | 是，用于确认方法实现位置和机制，不用于实验数字 |

## 2. Confirmed Experimental Setting

- 基础模型：OmniSAM / OmniSAM_adapt（从 run 路径、报告标题与项目上下文可确认；本次候选文件未重新读取完整 build 配置，细节仍建议后续从 config 快照核对）
- backbone：待确认。本次指定文件中未直接给出 `sam2.1_hiera_small` 或 `sam2_s` 的完整 backbone 字段；`step_09_main_results_draft_report.md` 提到 canonical PL 路径含 `sam2_s`。
- 源域：Cityscapes。来源：`paper_ablation_table_draft.md` 第 3 行写明 Cityscapes→DensePASS；`step_09_main_results_draft_report.md` 第 59 行写明 Cityscapes→DensePASS。
- 目标域：DensePASS。来源同上。
- 类别协议：Cityscapes_13cls / 13 类。来源：多个 run.log 的 PL 复用路径 `.../DensePASS/sam2_s/Cityscapes_13cls/epoch1`，如 `baseline_omnisam.../run.log` 第 32 行；但本次文件未读取类别列表本身。
- 训练/验证数据规模，如果能确认：
  - clean ablation / token budget：训练迭代数 `N_iters=500`，来自 `step_17_training_time_breakdown_report.md` 第 70-72 行，说明 `2000 PL images / batch_size=4`；对应 run.log 也显示 `Epoch 1 [Train]: 0/500`。
  - 验证集：DensePASS val 100 imgs，来自 `step_15_token_budget_results_report.md` 第 3 行与 `step_16_profile_summary_report.md` 第 3 行；run.log 显示 `Epoch 1 [Val] 0/100`。
  - full main result：`step_09_main_results_draft_report.md` 第 26 行写明完整 DensePASS protocol：2000 张 PL、跑满 1 epoch、100 张验证集。
- seed：seed=1。来源：`step_09_main_results_draft_report.md` 第 36 行；`step_15_token_budget_results_report.md` 第 3 行；run 目录名也含 `seed1`。
- 主要评价指标：mIoU、BFScore、mAcc、PixAcc、FWIoU、mDice。来源：`clean_ablation_results.csv`、`token_budget_tradeoff.csv`、run.log 的 validation summary。
- 主要效率指标：memory_tokens_after_final / memory_tokens、token_reduction_x、step_time、iter_compute_time、memory_attention_time、memory attention per iter、gpu_mem_peak。来源：`memory_token_attention_cost.csv`、`training_time_breakdown.csv`、`step_16_profile_summary_report.md`、`step_17_training_time_breakdown_report.md`。
- 是否基于公开 OmniSAM checkpoint 或当前已有 checkpoint：待确认。本次候选文件确认了复用 canonical pseudo-label 目录，但未直接确认训练初始化 checkpoint 路径或是否为公开 OmniSAM checkpoint。

## 3. Confirmed Main Results

### 3.1 Full DensePASS protocol 主结果（当前只能部分确认）

`step_09_main_results_draft_report.md` 第 22-27 行说明 full main result 来自：

- `data_pd_exp_final_2000/base_last_20260425_232941/`
- `data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/`

该报告第 59-63 行给出 A_Dtrain_Etrain vs baseline 的变化量：mIoU −0.10、BFScore −0.11、Step 1.29×、Mem-Attn 2.81×、Mem-Tokens 8.0×、Peak-Mem −1.48 GB。  
但是，本次允许深入读取的文件中没有 `main_results_draft.md` 或上述 `data_pd_exp_final_2000` 原始 metrics，因此以下用户指定的 full main result 绝对数只能标为“待确认”：baseline BFScore 39.97、efficient BFScore 39.86、efficient mIoU 60.16、训练耗时 43m53s→34m03s、单 iter 5.27s→4.09s。

| 方法/配置 | mIoU | BFScore | 训练耗时 | 单 iter 耗时 | 关键配置 | 结果来源 | 是否可写入论文 |
|---|---:|---:|---:|---:|---|---|---|
| OmniSAM baseline（full DensePASS protocol） | 60.26（绝对值在本次候选文件中与 clean ablation 一致；full 主结果来源需后续核对原始 metrics） | 待确认（用户给定 39.97，但本次候选文件未直接确认；`step_09` 仅确认相对 efficient 为 +0.11） | 待确认（用户给定 43m53s，但本次候选文件未直接确认） | 待确认（用户给定 5.27s，但本次候选文件未直接确认） | baseline；full DensePASS protocol；seed=1；epoch=1 | `step_09_main_results_draft_report.md` 第 22-27、59-63 行 | 暂不建议写绝对 BFScore/耗时；可写“主结果来源存在，需原始 metrics 复核” |
| Memory-efficient configuration / Ours / MTC+SMR（full DensePASS protocol） | 待确认（用户给定 60.16；`step_09` 只确认相对 baseline −0.10） | 待确认（用户给定 39.86；`step_09` 只确认相对 baseline −0.11） | 待确认（用户给定 34m03s，但本次候选文件未直接确认） | 待确认（用户给定 4.09s，但本次候选文件未直接确认） | A_Dtrain_Etrain；memory tokens 8.0× reduction；seed=1；epoch=1 | `step_09_main_results_draft_report.md` 第 22-27、59-63 行 | 暂不建议写绝对值；可写“相对变化：mIoU −0.10、BFScore −0.11、Step 1.29×、Mem-Tokens 8.0×（需原始表复核）” |

### 3.2 Fixed-PL clean ablation / profile 结果（已确认，可用于机制与消融表）

| 方法/配置 | mIoU | BFScore | 训练耗时 | 单 iter 耗时 | 关键配置 | 结果来源 | 是否可写入论文 |
|---|---:|---:|---:|---:|---|---|---|
| OmniSAM baseline（fixed-PL clean ablation） | 60.26 | 39.59 | 待确认（本组文件给 step time，不给完整训练总耗时） | step_time=2.3505 s/iter；iter_compute_time=1.5573 s/iter | A=off, D=off, E=off；memory tokens=32768 | `clean_ablation_results.csv` 第 2 行；`memory_token_attention_cost.csv` ablation baseline 行；`baseline_omnisam.../run.log` 第 677-678 行 BFScore=39.59、第 679 行 mIoU=60.26 | 是，但必须标注为 fixed-PL clean ablation / profile，不应混作 full final_2000 主结果 |
| A_ccpm reference（token budget 100%） | 60.26 | 39.59 | 待确认 | step_time=2.4089 s/iter；iter_compute_time=1.5815 s/iter | A=on, D=off, E=off；memory tokens=32768 | `token_budget_tradeoff.csv` 100% 行；`training_time_breakdown.csv` ablation A_ccpm 行；`step_15_token_budget_results_report.md` 第 43、53 行 | 是，用作 token budget curve 的 100% reference；不应当作 pure baseline 主结果 |
| Memory-efficient configuration / TeCMA full / A_Dtrain_Etrain（fixed-PL clean ablation） | 60.02 | 39.23 | 待确认 | step_time=2.3153 s/iter；iter_compute_time=1.4643 s/iter | A=on, D=on train, E=on train；memory tokens=4096；D 4096→1024 per-frame；E 8192→4096 in D+E chain；keep_ratio=0.5 | `clean_ablation_results.csv` 第 7 行；`memory_token_attention_cost.csv` ablation A_Dtrain_Etrain 行；`A_Dtrain_Etrain.../run.log` 第 677-679 行 | 是，但只用于 clean ablation / token-efficiency 机制分析；不应混作 full final_2000 主结果 |

### 3.3 已发现的数字冲突或口径差异

| 指标 | 候选文件中确认的数字 | 用户提示中希望确认的数字 | 当前处理 |
|---|---:|---:|---|
| baseline BFScore | 39.59（clean ablation/profile：`clean_ablation_results.csv`、run.log） | 39.97 | 冲突/口径不同。39.97 未在本次 17 个文件中直接确认，标为待确认 |
| efficient BFScore | 39.23（clean ablation/profile：A_Dtrain_Etrain） | 39.86 | 冲突/口径不同。`step_09` 只确认 full main result BFScore 相对下降 −0.11，未给绝对值 |
| efficient mIoU | 60.02（clean ablation/profile：A_Dtrain_Etrain） | 60.16 | 冲突/口径不同。`step_09` 只确认 full main result相对 baseline −0.10，若 baseline 60.26 则可推得 60.16，但本报告不把推导值当作已确认原始数字 |
| 单 iter 耗时 | 2.3505/2.4089/2.3153 s/iter（clean ablation/profile） | 5.27→4.09 s/iter | 口径不同。5.27→4.09 未在本次候选文件中直接出现，标为待确认 |
| 训练总耗时 | 本次候选文件只给 `step_time`、`iter_compute_time`、`memory_attention_time` 等 profile 口径 | 43m53s→34m03s | 未确认。需要后续读取 full main result 的 train_timing.jsonl 或主结果表 |
| BFScore 来源 | clean ablation BFScore 来自 run.log `[ValSummary]` | — | `paper_ablation_table_draft.md` 第 11 行与 `step_15` 第 23 行均说明 BFScore 来自 run.log，因为 run_summary.json 中 BFScore 为空 |

## 4. Token Budget and Module Configuration

| 模块 | 配置项 | baseline | efficient | 来源文件 | 是否可写入论文 |
|---|---|---:|---:|---|---|
| Baseline memory tokens | 每次 SAM2 memory attention 调用时所有 chunks 拼接后的 token 数 | 32768 | — | `step_16_profile_summary_report.md` 第 60-70 行；`memory_token_attention_cost.csv` baseline 行 | 是 |
| D / FSMC / SpatialMemoryCompressor | per-frame memory compression tokens | 无压缩 / 4096 tokens per patch | 4096 → 1024（per-frame/patch 口径） | `clean_ablation_results.csv` A_Dtrain 与 A_Dtrain_Etrain 行；`memory_token_attention_cost.csv` 的 `compression_tokens_before=4096`、`compression_tokens_after=1024`；`step_16` 第 72-73 行 | 是，需注明 per-frame/patch 口径 |
| D / FSMC / SpatialMemoryCompressor | final tokens after D-only | 32768 | 8192（A_Dtrain） | `clean_ablation_results.csv` A_Dtrain 行；`step_16` 第 72-73、92-100 行 | 是 |
| E / SG-SMR / SparseMemoryRetriever | E-only sparse retrieval tokens at keep=0.50 | 32768 | 16384 | `clean_ablation_results.csv` A_Etrain 行；`memory_token_attention_cost.csv` A_Etrain 行；`step_16` 第 195-198 行 | 是 |
| D+E combined | sparse retrieval tokens after D output | 32768 baseline final tokens | 8192 → 4096（D+E chain） | `clean_ablation_results.csv` Dtrain_Etrain/A_Dtrain_Etrain 行；`memory_token_attention_cost.csv` Dtrain_Etrain/A_Dtrain_Etrain 行 | 是 |
| D+E combined | final effective memory tokens | 32768 | 4096 | `memory_token_attention_cost.csv` A_Dtrain_Etrain 行；`step_15` 第 75-80 行；`step_16` 第 116-120 行 | 是 |
| D / config | `efficient_tam_memory.enabled` | false in baseline | true in D config | `configs/ablations/omnisam_base_plus4_efficienttam.yaml` 第 10-16 行 | 是，但该 YAML 是 D 独立配置，非最终 A_Dtrain_Etrain 快照 |
| D / config | `mode`, `compress_ratio`, `pool_stride` | — | mode=`pool`, compress_ratio=0.5, pool_stride=2 | `configs/ablations/omnisam_base_plus4_efficienttam.yaml` 第 10-14 行 | 可写入方法配置；需说明实测 token ratio 是空间池化后的 4096→1024，即 0.25 token ratio |
| E / config | `efficient_sam2_sparse.enabled` | false in baseline | true in E config | `configs/ablations/omnisam_base_plus5_efficientsam2_sparse.yaml` 第 18-21 行 | 是，但该 YAML 是 E 独立配置，非最终 A_Dtrain_Etrain 快照 |
| E / config | `keep_ratio` | — | 0.5 | `configs/ablations/omnisam_base_plus5_efficientsam2_sparse.yaml` 第 18-22 行；A_Dtrain_Etrain run.log 第 4 行 | 是 |
| Token budget / keep075 | E-only keep_ratio | — | 0.75；memory tokens=24576 | `token_budget_tradeoff.csv` 75% 行；`A_Etrain_keep075.../run.log` 第 4 行；`step_15` 第 44、54 行 | 是 |
| Token budget / keep025 | E-only keep_ratio | — | 0.25；memory tokens=8192 | `token_budget_tradeoff.csv` 25% 行；`A_Etrain_keep025.../run.log` 第 4 行；`step_15` 第 46、56 行 | 是 |
| Sparse retrieval scoring | saliency score | — | feature_norm / top-k | `model/modules/sparse_memory_retriever.py` 第 52-56、150-202、257-278 行；`step_17` 第 134-137 行 | 是，用于方法描述，不作为实验数字 |
| Memory compression implementation | pooling | — | `F.avg_pool2d` with kernel/stride | `model/modules/memory_compressor.py` 第 24-37、129-153 行 | 是，用于方法描述，不作为实验数字 |

注意：`configs/ablations/omnisam_base_plus4_efficienttam.yaml` 和 `configs/ablations/omnisam_base_plus5_efficientsam2_sparse.yaml` 中 `apply_in_pl: true`、`apply_in_train: true`，但 clean ablation/token-budget run.log 中 A_Dtrain_Etrain 与 token budget run 显示 `apply_in_train=True, apply_in_pl=False`（例如 `A_Dtrain_Etrain.../run.log` 第 4 行）。因此论文中描述最终实验配置时，应以后续最终 config 快照或 run.log 为准，不能仅引用这两个独立 ablation YAML。

## 5. Ablation and Trade-off Results

### 5.1 Clean ablation / fixed pseudo labels

| 实验名 | 配置 | mIoU | BFScore | 训练耗时/iter time | 结论 | 来源 |
|---|---|---:|---:|---:|---|---|
| baseline_omnisam | A=off, D=off, E=off；tokens=32768 | 60.26 | 39.59 | step_time=2.3505 s/iter；memory_attention_time=117.48 s/epoch | fixed-PL clean ablation baseline | `clean_ablation_results.csv` baseline 行；`baseline_omnisam.../run.log` 第 677-679 行 |
| A_ccpm | A=on, D=off, E=off；tokens=32768 | 60.26 | 39.59 | step_time=2.4089 s/iter；memory_attention_time=117.56 s/epoch | A 不改变 memory token 数；作为 token budget 100% reference | `clean_ablation_results.csv` A_ccpm 行；`step_16` 第 83-90 行 |
| A_Dtrain | A=on, D=on train, E=off；D 4096→1024 per-frame；tokens=8192 | 60.12 | 39.23 | step_time=2.2938 s/iter；memory_attention_time=61.00 s/epoch | D-only 在 fixed-PL clean ablation 中可形成 4× token reduction 与约 48% attention time reduction | `clean_ablation_results.csv` A_Dtrain 行；`paper_ablation_table_draft.md` 第 17 行；`step_16` 第 92-100 行 |
| A_Etrain | A=on, D=off, E=on keep=0.50；tokens=16384 | 60.09 | 39.23 | step_time=2.3798 s/iter；memory_attention_time=79.69 s/epoch | E-only keep=0.50 形成 2× token reduction 与约 32% attention time reduction | `clean_ablation_results.csv` A_Etrain 行；`paper_ablation_table_draft.md` 第 18 行；`step_16` 第 195-198 行 |
| Dtrain_Etrain | A=off, D=on train, E=on keep=0.50；tokens=4096 | 60.02 | 39.23 | step_time=2.3237 s/iter；memory_attention_time=52.38 s/epoch | D+E 组合可达 8× token reduction；A 不贡献 memory efficiency | `clean_ablation_results.csv` Dtrain_Etrain 行；`paper_ablation_table_draft.md` 第 19 行；`step_16` 第 116-124 行 |
| A_Dtrain_Etrain | A=on, D=on train, E=on keep=0.50；tokens=4096 | 60.02 | 39.23 | step_time=2.3153 s/iter；memory_attention_time=51.92 s/epoch | TeCMA full / A+D+E clean endpoint：8× token reduction，attention time 约 −55.8%，mIoU −0.24，BFScore −0.36（相对 clean baseline） | `clean_ablation_results.csv` A_Dtrain_Etrain 行；`paper_ablation_table_draft.md` 第 20-24 行；`A_Dtrain_Etrain.../run.log` 第 677-679 行 |

### 5.2 Token budget trade-off / E-only curve + D+E endpoint

| 实验名 | 配置 | mIoU | BFScore | 训练耗时/iter time | 结论 | 来源 |
|---|---|---:|---:|---:|---|---|
| 100% / A_ccpm | A=on, D=off, E=off；tokens=32768 | 60.26 | 39.59 | step_time=2.4089 s/iter；attention=117.56 s/epoch | 作为 token budget reference | `token_budget_tradeoff.csv` 100% 行；`step_15` 第 43、53 行 |
| 75% / A_Etrain_keep075 | A=on, E=on keep=0.75；tokens=24576 | 60.12 | 39.28 | step_time=2.4614 s/iter；attention=98.62 s/epoch | attention time 下降 16.11%，但 total step time 略高，说明稀疏检索 overhead 尚未被完全抵消 | `token_budget_tradeoff.csv` 75% 行；`A_Etrain_keep075.../run.log` 第 4、678-679 行；`step_15` 第 44、54 行；`step_17` 第 128-139 行 |
| 50% / A_Etrain | A=on, E=on keep=0.50；tokens=16384 | 60.09 | 39.23 | step_time=2.3798 s/iter；attention=79.69 s/epoch | E-only 50% 形成 2× token reduction，attention time 下降约 32% | `token_budget_tradeoff.csv` 50% 行；`step_15` 第 45、55 行 |
| 25% / A_Etrain_keep025 | A=on, E=on keep=0.25；tokens=8192 | 60.10 | 39.29 | step_time=2.2941 s/iter；attention=60.54 s/epoch | E-only 25% 形成 4× token reduction，attention time 下降约 48.5%，mIoU 仍约 60.10 | `token_budget_tradeoff.csv` 25% 行；`A_Etrain_keep025.../run.log` 第 4、678-679 行；`step_15` 第 46、56 行 |
| final 12.5% / A_Dtrain_Etrain | A=on, D=on compress, E=on keep=0.50；tokens=4096 | 60.02 | 39.23 | step_time=2.3153 s/iter；attention=51.92 s/epoch | D+E endpoint，不是 E-only curve 延伸点；达到 8× token reduction 与约 55.8% attention time reduction | `token_budget_tradeoff.csv` final 行；`step_15` 第 47、57、70-88 行 |

TODO：

- compression-only / D-only 在 clean ablation 中已有 `A_Dtrain`，但若论文需要“无 A 的 D-only”或 full final_2000 主协议下的 D-only，当前本报告未确认。
- sparse-only / E-only 在 clean ablation 中已有 `A_Etrain`，但若论文需要“无 A 的 E-only”或 full final_2000 主协议下的 E-only，当前本报告未确认。
- full DensePASS protocol 的主表绝对 BFScore、总训练耗时、5.27s→4.09s iter time 需要后续读取主结果表或原始 metrics/timing，不能从本次文件直接写入。

## 6. Efficiency Calculation

### 6.1 Clean ablation / profile 口径：A_ccpm reference → A_Dtrain_Etrain

原始数字：

- reference `A_ccpm`：step_time = 2.408930161572993 s/iter；memory_attention_time = 117.56346551328897 s/epoch；mIoU = 60.26；BFScore = 39.59。来源：`clean_ablation_results.csv` 第 3 行、`token_budget_tradeoff.csv` 100% 行。
- efficient `A_Dtrain_Etrain`：step_time = 2.3153213208429517 s/iter；memory_attention_time = 51.91887534968555 s/epoch；mIoU = 60.02；BFScore = 39.23。来源：`clean_ablation_results.csv` 第 7 行、`memory_token_attention_cost.csv` A_Dtrain_Etrain 行。

计算：

- 单 iter step time 下降比例：  
  `reduction = (baseline - ours) / baseline = (2.408930161572993 - 2.3153213208429517) / 2.408930161572993 = 0.03886 ≈ 3.89%`。  
  来源交叉确认：`step_15_token_budget_results_report.md` 第 57 行给出 −3.89%；`step_17_training_time_breakdown_report.md` 第 115-120 行给出 −0.094s/iter（−3.89%）。
- memory attention time 下降比例：  
  `reduction = (117.56346551328897 - 51.91887534968555) / 117.56346551328897 = 0.5584 ≈ 55.84%`。  
  来源交叉确认：`step_15` 第 57、82-84 行；`step_17` 第 102-108 行。
- mIoU 变化：  
  `delta = ours - baseline = 60.02 - 60.26 = -0.24`。来源：`clean_ablation_results.csv` 第 7 行、`paper_ablation_table_draft.md` 第 20-24 行。
- BFScore 变化：  
  `delta = ours - baseline = 39.23 - 39.59 = -0.36`。来源：`clean_ablation_results.csv` 第 7 行、`paper_ablation_table_draft.md` 第 20-24 行。

### 6.2 Clean ablation / profile 口径：OmniSAM baseline → A_Dtrain_Etrain

原始数字：

- baseline_omnisam：step_time = 2.350520139120519 s/iter；memory_attention_time = 117.48247451335192 s/epoch；mIoU = 60.26；BFScore = 39.59。来源：`clean_ablation_results.csv` baseline 行。
- A_Dtrain_Etrain：step_time = 2.3153213208429517 s/iter；memory_attention_time = 51.91887534968555 s/epoch；mIoU = 60.02；BFScore = 39.23。来源：`clean_ablation_results.csv` A_Dtrain_Etrain 行。

计算：

- 单 iter step time 下降比例：  
  `reduction = (2.350520139120519 - 2.3153213208429517) / 2.350520139120519 = 0.01498 ≈ 1.50%`。来源：`clean_ablation_results.csv` A_Dtrain_Etrain 行的 `step_time_change_pct=-1.50`。
- memory attention time 下降比例：  
  `reduction = (117.48247451335192 - 51.91887534968555) / 117.48247451335192 = 0.5581 ≈ 55.81%`。来源：`memory_token_attention_cost.csv` A_Dtrain_Etrain 行 `attn_time_reduction_pct=55.81`。
- mIoU 变化：`60.02 - 60.26 = -0.24`。来源：`clean_ablation_results.csv`。
- BFScore 变化：`39.23 - 39.59 = -0.36`。来源：`clean_ablation_results.csv`。

### 6.3 Full DensePASS protocol 口径：当前不能完整计算

`step_09_main_results_draft_report.md` 第 61-63 行确认 A_Dtrain_Etrain vs baseline 的变化量为：mIoU −0.10、BFScore −0.11、Step 1.29×、Mem-Attn 2.81×、Mem-Tokens 8.0×、Peak-Mem −1.48 GB。

但是，本次候选文件没有给出 full main result 的原始训练总耗时 43m53s→34m03s，也没有直接给出单 iter 5.27s→4.09s。因此：

- 训练总耗时下降比例：待确认，不能根据本次文件计算。
- 单 iter 耗时下降比例：待确认，不能根据本次文件计算。
- 若后续原始文件确认 5.27s→4.09s，则公式应为：`reduction = (5.27 - 4.09) / 5.27 ≈ 22.39%`；当前不写为论文事实。
- 若后续原始文件确认 43m53s→34m03s，则公式应为：`reduction = (43*60+53 - (34*60+3)) / (43*60+53)`；当前不写为论文事实。

## 7. Paper-ready Claims

以下表述仅针对本次文件已经确认的 memory-efficient / token-efficient training 事实，且建议在论文中注明 single seed 和 fixed-PL/profile 口径：

1. 初步实验表明，在当前 Cityscapes→DensePASS 的 fixed-PL clean ablation 设置下，D+E 组合可以将每次 memory attention 调用的 effective memory tokens 从 32768 降至 4096，实现 8× token reduction，同时 mIoU 从 60.26 变为 60.02、BFScore 从 39.59 变为 39.23。来源：`clean_ablation_results.csv`、`memory_token_attention_cost.csv`、`paper_ablation_table_draft.md`。
2. 在当前 profile 口径下，A_Dtrain_Etrain 的 memory attention time 从 A_ccpm reference 的 117.56s/epoch 降至 51.92s/epoch，下降约 55.8%；总 step time 从 2.4089s/iter 降至 2.3153s/iter，下降约 3.9%。来源：`token_budget_tradeoff.csv`、`step_15_token_budget_results_report.md`、`step_17_training_time_breakdown_report.md`。
3. Token budget curve 显示，E-only keep ratio 从 1.00 降至 0.75、0.50、0.25 时，memory attention time 按 117.56→98.62→79.69→60.54s 的趋势下降；这说明 memory attention cost 与保留 token 数存在一致关系。来源：`step_15_token_budget_results_report.md` 第 51-57、61-66 行。
4. 在 mIoU 基本保持在约 60 左右的情况下，memory token 数从 32768 降至 8192 或 4096 仍能保持可用的分割结果；这支持“memory token 存在一定冗余”的谨慎表述。来源：`token_budget_tradeoff.csv`、`step_15_token_budget_results_report.md` 第 41-47 行。
5. 当前结果更适合表述为“以小幅精度代价换取 memory attention cost 降低”，不应表述为“无精度损失”或“全面优于 baseline”。来源：`paper_ablation_table_draft.md` 第 26 行；`step_15_token_budget_results_report.md` 第 122-128 行。

## 8. Results Not Safe to Claim Yet

- 多 seed 稳定性尚未确认。`step_09_main_results_draft_report.md` 第 34-45 行和 `step_15_token_budget_results_report.md` 第 122-135 行均提示当前为 seed=1 single seed，无方差估计。
- full DensePASS protocol 主结果的绝对 BFScore、训练总耗时、单 iter 耗时尚未从本次允许读取的候选文件直接确认；用户提示的 39.97→39.86、43m53s→34m03s、5.27s→4.09s 不能在本报告中写成已确认事实。
- clean ablation/profile 结果与 full final_2000 主结果存在口径差异：clean ablation 中 A_Dtrain_Etrain mIoU/BFScore 为 60.02/39.23，而用户希望确认的 full main result 为 60.16/39.86，必须分开表述。
- 更多数据集泛化尚未确认。`step_15_token_budget_results_report.md` 第 137-138 行提到可补 Cityscapes→WildPASS 泛化实验，但本任务未确认相关结果。
- compression-only / sparse-only 在 fixed-PL clean ablation 下已有 A_Dtrain/A_Etrain，但 full main protocol 下的 D-only/E-only 或无 A 的纯 D-only/E-only 是否完整可用，当前未确认。
- 显存峰值已有 GPU peak，但 FLOPs、真实训练显存峰值的严格对齐、吞吐量和多 GPU/不同硬件稳定性尚未确认。
- step time 不严格单调：75% token budget 的 step_time 比 100% reference 略高（2.4614s vs 2.4089s），原因可能是 sparse retrieval overhead 尚未被充分摊销；论文中不能声称 total step time 随 token budget 严格单调下降。来源：`step_15` 第 101-110 行、`step_17` 第 128-139 行。
- 不能声称 CCPM 对 memory efficiency 有贡献。`step_16_profile_summary_report.md` 第 83-90 行说明 A 模块不改变 memory token 数量。
- `configs/ablations/omnisam_base_plus4_efficienttam.yaml` 与 `configs/ablations/omnisam_base_plus5_efficientsam2_sparse.yaml` 中 `apply_in_pl: true`，而 run.log 中 token-efficient run 为 `apply_in_train=True, apply_in_pl=False`；最终论文配置需要以后续最终 YAML 或 run log 为准。

## 9. Recommended Tables for Paper v1.0

### 表 1：主结果表

- 建议章节：实验结果 / Main Results。
- 当前可填字段：
  - 方法名：OmniSAM baseline；Memory-efficient configuration / Ours / MTC+SMR / A_Dtrain_Etrain。
  - setting：Cityscapes→DensePASS，seed=1，epoch=1，2000 PL + DensePASS val 100（来源 `step_09`）。
  - 相对变化：mIoU −0.10、BFScore −0.11、Step 1.29×、Mem-Attn 2.81×、Mem-Tokens 8.0×、Peak-Mem −1.48GB（来源 `step_09` 第 61-63 行）。
- 当前必须 TODO 的字段：
  - baseline BFScore 39.97、efficient BFScore 39.86 的原始来源；
  - efficient mIoU 60.16 的原始来源；
  - 训练耗时 43m53s→34m03s；
  - 单 iter 5.27s→4.09s；
  - 是否使用公开 OmniSAM checkpoint / 当前已有 checkpoint 的明确路径。

### 表 2：效率分析表

- 建议章节：Efficiency Analysis / Training Efficiency。
- 当前可填字段：
  - clean/profile 口径：step_time、iter_compute_time、memory_attention_time、memory attention per iter、memory attention share、N_iters=500。
  - A_ccpm reference → A_Dtrain_Etrain：2.4089→2.3153 s/iter；117.56→51.92 s/epoch；235.1→103.8 ms/iter；−55.84% memory attention；−3.89% step time。
- 当前必须 TODO 的字段：
  - full final_2000 的 end-to-end training time 和 per-iter time 原始证据；
  - 是否需要加入 PL generation / validation / E2E 时间，避免与 train-only time 混淆。

### 表 3：memory 消融表

- 建议章节：Ablation Study。
- 当前可填字段：
  - baseline_omnisam、A_ccpm、A_Dtrain、A_Etrain、Dtrain_Etrain、A_Dtrain_Etrain。
  - mIoU、BFScore、memory tokens、memory attention time、GPU peak。
  - 来源：`clean_ablation_results.csv`、`memory_token_attention_cost.csv`、`paper_ablation_table_draft.md`。
- 当前必须 TODO 的字段：
  - 明确表题写 fixed pseudo labels / clean ablation，避免与 full main result 混淆；
  - 如果论文需要“纯 D-only / 纯 E-only（无 A）”或 full protocol 消融，需要补充确认。

### 表 4：token budget trade-off 表

- 建议章节：Efficiency-Accuracy Trade-off / Token Budget Analysis。
- 当前可填字段：
  - 100%、75%、50%、25% E-only curve；final 12.5% D+E endpoint。
  - tokens、token ratio、token reduction、mIoU、BFScore、step time、attention time、GPU peak。
  - 来源：`token_budget_tradeoff.csv`、`step_15_token_budget_results_report.md`、keep025/keep075 run.log。
- 当前必须 TODO 的字段：
  - 明确 final 12.5% 是 D+E combined endpoint，不是 E-only curve 延伸点；
  - 是否放图：`speed_accuracy_tradeoff_miou.png`、`speed_accuracy_tradeoff_bfscore.png`、`token_budget_attention_time.png` 在本任务未读取，只能作为后续整理对象；
  - 若要投稿，建议补 multi-seed 或至少在 caption 中注明 seed=1。
