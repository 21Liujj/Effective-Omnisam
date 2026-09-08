# Source Inventory for OmniSAM Paper v1.0

## 1. Project Root Check
- 当前工作目录：`/root/autodl-tmp/OmniSAM-main`
- 是否判断为 OmniSAM 项目根目录：是
- 判断依据：`run_adaptation.py` 存在; `train.py` 存在; `validation.py` 存在; `adaptation` 存在; `model` 存在; `dataset` 存在; `configs` 存在。
- 当前目录下的主要文件和文件夹：
  - `  `（目录）
  - `.claude`（目录）
  - `.git`（目录）
  - `.gitignore`（文件）
  - `CLAUDE.md`（文件）
  - `README`（文件）
  - `ablation_outputs`（目录）
  - `adaptation`（目录）
  - `assets`（目录）
  - `aux_outputs`（目录）
  - `ckpt_zoo`（目录）
  - `config_mem_pool2_before_attention.yaml`（文件）
  - `configs`（目录）
  - `data`（目录）
  - `data_extra`（目录）
  - `data_pd_cg_src_2000_01_ori`（目录）
  - `data_pd_exp_final_2000`（目录）
  - `dataset`（目录）
  - `debug_wildpass.py`（文件）
  - `docs`（目录）
  - `exp2`（目录）
  - `experiments`（目录）
  - `license.txt`（文件）
  - `logs`（目录）
  - `markdown_archive`（目录）
  - `model`（目录）
  - `requirements.txt`（文件）
  - `run_adaptation.py`（文件）
  - `runs`（目录）
  - `sam`（目录）
  - `sam2`（目录）
  - `scripts`（目录）
  - `tecma_phase5_6_results.tar.gz`（文件）
  - `tecma_phase7_1_per_class_iou.tar.gz`（文件）
  - `tests`（目录）
  - `tools`（目录）
  - `train`（目录）
  - `train.py`（文件）
  - `utils`（目录）
  - `validation.py`（文件）
  - `validation_muti.py`（文件）
  - `数据根目录:`（文件）

## 2. Memory-efficient 相关材料
| 序号 | 文件路径 | 文件类型 | 可能内容 | 对论文 v1.0 的价值 | 是否建议后续深入读取 |
|---|---|---|---|---|---|
| 1 | `  /geo_memory/diagnostics/token_region_summary.json` | JSON/JSONL 结果或配置 | memory 机制/日志；实验配置 | 中：候选材料，建议按需核验 | 待定 |
| 2 | `configs/ablations/omnisam_base_plus4_efficienttam.yaml` | YAML 配置 | 实验配置 | 中-高：可能对应消融/最终配置 | 是 |
| 3 | `configs/ablations/omnisam_base_plus5_efficientsam2_sparse.yaml` | YAML 配置 | 稀疏检索模块或配置；实验配置 | 中-高：可能对应消融/最终配置 | 是 |
| 4 | `docs/efficient_sam2_style_sparse_memory_retrieval_zh.md` | Markdown 文档 | memory 机制/日志；稀疏检索模块或配置；评估指标或验证输出 | 中：候选材料，建议按需核验 | 是 |
| 5 | `experiments` | 文件 | 文件名命中 memory-efficient 关键词，需后续确认 | 中：候选材料，建议按需核验 | 待定 |
| 6 | `experiments/geo_memory/__pycache__/token_diag_writer.cpython-310.pyc` | 文件 | memory 机制/日志 | 中：候选材料，建议按需核验 | 待定 |
| 7 | `experiments/geo_memory/diagnostics/geo_memory_diag_task_02_3.jsonl` | JSON/JSONL 结果或配置 | memory 机制/日志 | 中：候选材料，建议按需核验 | 待定 |
| 8 | `experiments/geo_memory/diagnostics/token_region_sample.json` | JSON/JSONL 结果或配置 | memory 机制/日志；实验配置 | 中：候选材料，建议按需核验 | 待定 |
| 9 | `experiments/geo_memory/logs/task_02_3_token_diag.log` | 日志/文本 | memory 机制/日志 | 高：可能支撑速度/指标事实，需后续读取 | 待定 |
| 10 | `experiments/geo_memory/notes/step_00_2_memory_token_inventory.md` | Markdown 文档 | memory 机制/日志 | 中：候选材料，建议按需核验 | 待定 |
| 11 | `experiments/geo_memory/notes/step_02_1_token_metadata_schema.md` | Markdown 文档 | memory 机制/日志 | 中：候选材料，建议按需核验 | 待定 |
| 12 | `experiments/geo_memory/notes/step_02_4_tecmem_token_distribution.md` | Markdown 文档 | memory 机制/日志 | 中：候选材料，建议按需核验 | 待定 |
| 13 | `experiments/geo_memory/results/geo_memory_diag.jsonl` | JSON/JSONL 结果或配置 | memory 机制/日志；评估指标或验证输出 | 中：候选材料，建议按需核验 | 待定 |
| 14 | `experiments/geo_memory/results/geo_memory_diag_smoke_00_4_summary.json` | JSON/JSONL 结果或配置 | memory 机制/日志；评估指标或验证输出；实验配置 | 中：候选材料，建议按需核验 | 待定 |
| 15 | `experiments/geo_memory/results/token_geometry_diag.jsonl` | JSON/JSONL 结果或配置 | memory 机制/日志；评估指标或验证输出 | 中：候选材料，建议按需核验 | 待定 |
| 16 | `experiments/geo_memory/results/token_writer_sample.json` | JSON/JSONL 结果或配置 | memory 机制/日志；评估指标或验证输出；实验配置 | 中：候选材料，建议按需核验 | 待定 |
| 17 | `experiments/geo_memory/tables/token_region_distribution_tecmem.md` | Markdown 文档 | memory 机制/日志 | 中：候选材料，建议按需核验 | 待定 |
| 18 | `experiments/geo_memory/token_diag_writer.py` | Python 源码 | memory 机制/日志 | 中：候选材料，建议按需核验 | 待定 |
| 19 | `experiments/pl_refiner/configs/R6_memory_plq.yaml` | YAML 配置 | memory 机制/日志；实验配置 | 中-高：可能对应消融/最终配置 | 待定 |
| 20 | `experiments/tecma/figures/speed_accuracy_tradeoff_bfscore.png` | 图像/可视化 | 速度/耗时记录 | 中：候选材料，建议按需核验 | 待定 |
| 21 | `experiments/tecma/figures/speed_accuracy_tradeoff_miou.png` | 图像/可视化 | 速度/耗时记录 | 中：候选材料，建议按需核验 | 待定 |
| 22 | `experiments/tecma/figures/token_budget_attention_time.png` | 图像/可视化 | 速度/耗时记录 | 中：候选材料，建议按需核验 | 待定 |
| 23 | `experiments/tecma/notes/step_09_bfscore_evidence_audit.md` | Markdown 文档 | 文件名命中 memory-efficient 关键词，需后续确认 | 中：候选材料，建议按需核验 | 待定 |
| 24 | `experiments/tecma/notes/step_13_token_budget_plan.md` | Markdown 文档 | 文件名命中 memory-efficient 关键词，需后续确认 | 中：候选材料，建议按需核验 | 待定 |
| 25 | `experiments/tecma/notes/step_14_token_budget_script_report.md` | Markdown 文档 | 文件名命中 memory-efficient 关键词，需后续确认 | 中：候选材料，建议按需核验 | 待定 |
| 26 | `experiments/tecma/notes/step_15_token_budget_preflight_report.md` | Markdown 文档 | 文件名命中 memory-efficient 关键词，需后续确认 | 中：候选材料，建议按需核验 | 待定 |
| 27 | `experiments/tecma/notes/step_15_token_budget_results_report.md` | Markdown 文档 | 评估指标或验证输出 | 中：候选材料，建议按需核验 | 待定 |
| 28 | `experiments/tecma/notes/step_17_training_time_breakdown_report.md` | Markdown 文档 | 速度/耗时记录 | 中：候选材料，建议按需核验 | 待定 |
| 29 | `experiments/tecma/runs/token_budget/_bg_logs/token_budget_seed1_20260617_112455.log` | 日志/文本 | 文件名命中 memory-efficient 关键词，需后续确认 | 高：可能支撑速度/指标事实，需后续读取 | 待定 |
| 30 | `experiments/tecma/scripts/check_configs.py.bak_tokenbudget` | 文件 | 实验配置 | 中：候选材料，建议按需核验 | 待定 |
| 31 | `experiments/tecma/scripts/run_token_budget_experiments.sh` | 文件 | 文件名命中 memory-efficient 关键词，需后续确认 | 中：候选材料，建议按需核验 | 待定 |
| 32 | `experiments/tecma/tables/memory_token_attention_cost.csv` | CSV 表格 | memory 机制/日志 | 中：候选材料，建议按需核验 | 待定 |
| 33 | `experiments/tecma/tables/memory_token_attention_cost.md` | Markdown 文档 | memory 机制/日志 | 中：候选材料，建议按需核验 | 待定 |
| 34 | `experiments/tecma/tables/token_budget_tradeoff.csv` | CSV 表格 | 文件名命中 memory-efficient 关键词，需后续确认 | 中：候选材料，建议按需核验 | 待定 |
| 35 | `experiments/tecma/tables/token_budget_tradeoff.md` | Markdown 文档 | 文件名命中 memory-efficient 关键词，需后续确认 | 中：候选材料，建议按需核验 | 待定 |
| 36 | `experiments/tecma/tables/training_time_breakdown.csv` | CSV 表格 | 速度/耗时记录 | 中：候选材料，建议按需核验 | 待定 |
| 37 | `experiments/tecma/tables/training_time_breakdown.md` | Markdown 文档 | 速度/耗时记录 | 中：候选材料，建议按需核验 | 待定 |
| 38 | `logs/Standford2d3d_results/summaries/standford2d3d_train_time.csv` | CSV 表格 | 速度/耗时记录；评估指标或验证输出 | 高：可能支撑速度/指标事实，需后续读取 | 待定 |
| 39 | `logs/omnisam_time_ablation/plugin_ablation_base_plus4_efficienttam_20260425_153453.log` | 日志/文本 | 速度/耗时记录 | 高：可能支撑速度/指标事实，需后续读取 | 是 |
| 40 | `logs/omnisam_time_ablation/plugin_ablation_base_plus5_efficientsam2_sparse_20260425_153453.log` | 日志/文本 | 稀疏检索模块或配置；速度/耗时记录 | 高：可能支撑速度/指标事实，需后续读取 | 是 |
| 41 | `logs/wildpass_results/reports/memory_accel_meta.json` | JSON/JSONL 结果或配置 | memory 机制/日志；评估指标或验证输出；数据集相关配置/结果；实验配置 | 高：可能支撑速度/指标事实，需后续读取 | 待定 |
| 42 | `logs/wildpass_results/summaries/wildpass_train_time.csv` | CSV 表格 | 速度/耗时记录；评估指标或验证输出；数据集相关配置/结果 | 高：可能支撑速度/指标事实，需后续读取 | 待定 |
| 43 | `logs/wildpass_results/val_ckpts/20260511_160538/final_ckpt/memory_accel_meta.json` | JSON/JSONL 结果或配置 | memory 机制/日志；评估指标或验证输出；数据集相关配置/结果；实验配置 | 高：可能支撑速度/指标事实，需后续读取 | 待定 |
| 44 | `logs/wildpass_results/val_ckpts/20260511_160538/final_ckpt/memory_config_used.yaml` | YAML 配置 | memory 机制/日志；评估指标或验证输出；数据集相关配置/结果；实验配置 | 高：可能支撑速度/指标事实，需后续读取 | 待定 |
| 45 | `logs/wildpass_results/val_ckpts/20260511_160538/paper_ckpt/memory_accel_meta.json` | JSON/JSONL 结果或配置 | memory 机制/日志；评估指标或验证输出；数据集相关配置/结果；实验配置 | 高：可能支撑速度/指标事实，需后续读取 | 待定 |
| 46 | `logs/wildpass_results/val_ckpts/20260511_160538/paper_ckpt/memory_config_used.yaml` | YAML 配置 | memory 机制/日志；评估指标或验证输出；数据集相关配置/结果；实验配置 | 高：可能支撑速度/指标事实，需后续读取 | 待定 |
| 47 | `logs/wildpass_results/val_ckpts/20260511_180208/final_ckpt/memory_accel_meta.json` | JSON/JSONL 结果或配置 | memory 机制/日志；评估指标或验证输出；数据集相关配置/结果；实验配置 | 高：可能支撑速度/指标事实，需后续读取 | 待定 |
| 48 | `logs/wildpass_results/val_ckpts/20260511_180208/paper_ckpt/memory_accel_meta.json` | JSON/JSONL 结果或配置 | memory 机制/日志；评估指标或验证输出；数据集相关配置/结果；实验配置 | 高：可能支撑速度/指标事实，需后续读取 | 待定 |
| 49 | `logs/wildpass_results/val_ckpts/20260511_190132/final_ckpt/memory_accel_meta.json` | JSON/JSONL 结果或配置 | memory 机制/日志；评估指标或验证输出；数据集相关配置/结果；实验配置 | 高：可能支撑速度/指标事实，需后续读取 | 待定 |
| 50 | `logs/wildpass_results/val_ckpts/20260511_190132/paper_ckpt/memory_accel_meta.json` | JSON/JSONL 结果或配置 | memory 机制/日志；评估指标或验证输出；数据集相关配置/结果；实验配置 | 高：可能支撑速度/指标事实，需后续读取 | 待定 |
| 51 | `markdown_archive/docs/efficient_sam2_style_sparse_memory_retrieval.md` | Markdown 文档 | memory 机制/日志；稀疏检索模块或配置；评估指标或验证输出 | 中：候选材料，建议按需核验 | 是 |
| 52 | `markdown_archive/docs/efficienttam_memory_compression.md` | Markdown 文档 | memory 机制/日志；压缩模块或压缩配置 | 中：候选材料，建议按需核验 | 是 |
| 53 | `markdown_archive/docs/efficienttam_memory_compression_zh.md` | Markdown 文档 | memory 机制/日志；压缩模块或压缩配置 | 中：候选材料，建议按需核验 | 是 |
| 54 | `model/__pycache__/memory_bank.cpython-310.pyc` | 文件 | memory 机制/日志 | 高：可用于说明方法实现位置，但需后续读取确认细节 | 是 |
| 55 | `model/memory_bank.py` | Python 源码 | memory 机制/日志 | 高：可用于说明方法实现位置，但需后续读取确认细节 | 是 |
| 56 | `model/modules/__pycache__/memory_compressor.cpython-310.pyc` | 文件 | memory 机制/日志；压缩模块或压缩配置 | 高：可用于说明方法实现位置，但需后续读取确认细节 | 是 |
| 57 | `model/modules/__pycache__/sparse_memory_retriever.cpython-310.pyc` | 文件 | memory 机制/日志；稀疏检索模块或配置 | 高：可用于说明方法实现位置，但需后续读取确认细节 | 是 |
| 58 | `model/modules/memory_compressor.py` | Python 源码 | memory 机制/日志；压缩模块或压缩配置 | 高：可用于说明方法实现位置，但需后续读取确认细节 | 是 |
| 59 | `model/modules/sparse_memory_retriever.py` | Python 源码 | memory 机制/日志；稀疏检索模块或配置 | 高：可用于说明方法实现位置，但需后续读取确认细节 | 是 |
| 60 | `sam/demo/frontend/src/theme/tokens.stylex.ts` | 文件 | 文件名命中 memory-efficient 关键词，需后续确认 | 中：候选材料，建议按需核验 | 待定 |
| 61 | `sam/sam/modeling/__pycache__/memory_attention.cpython-310.pyc` | 文件 | memory 机制/日志 | 中：候选材料，建议按需核验 | 待定 |
| 62 | `sam/sam/modeling/__pycache__/memory_encoder.cpython-310.pyc` | 文件 | memory 机制/日志 | 中：候选材料，建议按需核验 | 待定 |
| 63 | `sam/sam/modeling/memory_attention.py` | Python 源码 | memory 机制/日志 | 中：候选材料，建议按需核验 | 待定 |
| 64 | `sam/sam/modeling/memory_encoder.py` | Python 源码 | memory 机制/日志 | 中：候选材料，建议按需核验 | 待定 |
| 65 | `scripts/extra_exp/standford2d3d/02_train_time_standford2d3d_reuse_pl.sh` | 文件 | 速度/耗时记录 | 中：候选材料，建议按需核验 | 待定 |
| 66 | `scripts/extra_exp/wildpass/02_train_time_wildpass_reuse_pl.sh` | 文件 | 速度/耗时记录；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 67 | `scripts/extra_exp/wildpass/04_val_memory_accel_wildpass.sh` | 文件 | memory 机制/日志；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 68 | `scripts/run_ablation_plus4_efficienttam.sh` | 文件 | 文件名命中 memory-efficient 关键词，需后续确认 | 中：候选材料，建议按需核验 | 是 |
| 69 | `scripts/run_ablation_plus5_efficientsam2_sparse.sh` | 文件 | 稀疏检索模块或配置 | 中：候选材料，建议按需核验 | 是 |
| 70 | `scripts/time_ablation/run_time_ablation_main4.sh` | 文件 | 速度/耗时记录 | 中：候选材料，建议按需核验 | 待定 |
| 71 | `scripts/time_ablation/run_time_ablation_memory_internal.sh` | 文件 | memory 机制/日志；速度/耗时记录 | 中：候选材料，建议按需核验 | 待定 |
| 72 | `scripts/time_ablation/run_time_ablation_one.sh` | 文件 | 速度/耗时记录 | 中：候选材料，建议按需核验 | 待定 |
| 73 | `scripts/time_ablation/run_time_ablation_patch_internal.sh` | 文件 | 速度/耗时记录 | 中：候选材料，建议按需核验 | 待定 |
| 74 | `scripts/time_ablation/run_time_ablation_smoke.sh` | 文件 | 速度/耗时记录 | 中：候选材料，建议按需核验 | 待定 |
| 75 | `scripts/time_ablation/run_time_ablation_staged.sh` | 文件 | 速度/耗时记录 | 中：候选材料，建议按需核验 | 待定 |
| 76 | `tools/summarize_time_ablation.py` | Python 源码 | 速度/耗时记录 | 中：候选材料，建议按需核验 | 待定 |
| 77 | `tools/wildpass_memory_accel.py` | Python 源码 | memory 机制/日志；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 78 | `  /geo_memory` | 目录 | memory 机制/日志 | 中：候选材料，建议按需核验 | 待定 |
| 79 | `experiments/geo_memory` | 目录 | memory 机制/日志 | 中：候选材料，建议按需核验 | 待定 |
| 80 | `experiments/tecma/runs/token_budget` | 目录 | 文件名命中 memory-efficient 关键词，需后续确认 | 中：候选材料，建议按需核验 | 待定 |
| 81 | `logs/Standford2d3d_results/train_time_reuse_pl` | 目录 | 速度/耗时记录；评估指标或验证输出 | 高：可能支撑速度/指标事实，需后续读取 | 待定 |
| 82 | `logs/omnisam_time_ablation` | 目录 | 速度/耗时记录 | 高：可能支撑速度/指标事实，需后续读取 | 待定 |
| 83 | `logs/omnisam_time_ablation_seed1_triplet` | 目录 | 速度/耗时记录 | 高：可能支撑速度/指标事实，需后续读取 | 待定 |
| 84 | `logs/time_ablation` | 目录 | 速度/耗时记录 | 高：可能支撑速度/指标事实，需后续读取 | 待定 |
| 85 | `logs/wildpass_results/train_time_reuse_pl` | 目录 | 速度/耗时记录；评估指标或验证输出；数据集相关配置/结果 | 高：可能支撑速度/指标事实，需后续读取 | 待定 |
| 86 | `logs/wildpass_results/val_memory_accel` | 目录 | memory 机制/日志；评估指标或验证输出；数据集相关配置/结果 | 高：可能支撑速度/指标事实，需后续读取 | 待定 |
| 87 | `markdown_archive/scripts/time_ablation` | 目录 | 速度/耗时记录 | 中：候选材料，建议按需核验 | 待定 |
| 88 | `scripts/time_ablation` | 目录 | 速度/耗时记录 | 中：候选材料，建议按需核验 | 待定 |

## 3. Pseudo-label Refinement 相关材料
| 序号 | 文件路径 | 文件类型 | 可能内容 | 对论文 v1.0 的价值 | 是否建议后续深入读取 |
|---|---|---|---|---|---|
| 1 | `adaptation/plugins/__pycache__/pseudo_conf_mask.cpython-310.pyc` | 文件 | 伪标签生成或筛选 | 高：可用于定位伪标签流程/筛选实现 | 是 |
| 2 | `adaptation/plugins/pseudo_conf_mask.py` | Python 源码 | 伪标签生成或筛选 | 高：可用于定位伪标签流程/筛选实现 | 是 |
| 3 | `adaptation/pseudo_label/__pycache__/prototype_gate.cpython-310.pyc` | 文件 | 伪标签生成或筛选；伪标签优化策略线索 | 高：可能对应伪标签优化实验或设计 | 是 |
| 4 | `adaptation/pseudo_label/prototype_bank.py` | Python 源码 | 伪标签生成或筛选；伪标签优化策略线索 | 高：可能对应伪标签优化实验或设计 | 是 |
| 5 | `adaptation/pseudo_label/prototype_gate.py` | Python 源码 | 伪标签生成或筛选；伪标签优化策略线索 | 高：可能对应伪标签优化实验或设计 | 是 |
| 6 | `assets/text_prototypes_cityscapes13_vitb32.pt` | 模型权重/张量文件 | 伪标签优化策略线索；数据集相关配置/结果；checkpoint/权重 | 高：可能对应伪标签优化实验或设计 | 是 |
| 7 | `assets/text_prototypes_openclip_text_latest.pt` | 模型权重/张量文件 | 伪标签优化策略线索；checkpoint/权重 | 高：可能对应伪标签优化实验或设计 | 是 |
| 8 | `configs/experiments/text_modality/text_ablation_A2_text_branch_gate_soft.yaml` | YAML 配置 | 伪标签优化策略线索；实验配置 | 高：可能对应伪标签优化实验或设计 | 是 |
| 9 | `configs/experiments/text_modality/text_ablation_A3_text_branch_gate_hard.yaml` | YAML 配置 | 伪标签优化策略线索；实验配置 | 高：可能对应伪标签优化实验或设计 | 是 |
| 10 | `data/DensePASS/gtFine/val/100_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 11 | `data/DensePASS/gtFine/val/101_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 12 | `data/DensePASS/gtFine/val/102_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 13 | `data/DensePASS/gtFine/val/103_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 14 | `data/DensePASS/gtFine/val/104_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 15 | `data/DensePASS/gtFine/val/105_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 16 | `data/DensePASS/gtFine/val/106_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 17 | `data/DensePASS/gtFine/val/107_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 18 | `data/DensePASS/gtFine/val/108_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 19 | `data/DensePASS/gtFine/val/109_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 20 | `data/DensePASS/gtFine/val/10_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 21 | `data/DensePASS/gtFine/val/110_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 22 | `data/DensePASS/gtFine/val/111_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 23 | `data/DensePASS/gtFine/val/112_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 24 | `data/DensePASS/gtFine/val/113_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 25 | `data/DensePASS/gtFine/val/114_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 26 | `data/DensePASS/gtFine/val/115_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 27 | `data/DensePASS/gtFine/val/116_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 28 | `data/DensePASS/gtFine/val/117_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 29 | `data/DensePASS/gtFine/val/118_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 30 | `data/DensePASS/gtFine/val/119_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 31 | `data/DensePASS/gtFine/val/11_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 32 | `data/DensePASS/gtFine/val/120_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 33 | `data/DensePASS/gtFine/val/121_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 34 | `data/DensePASS/gtFine/val/122_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 35 | `data/DensePASS/gtFine/val/123_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 36 | `data/DensePASS/gtFine/val/124_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 37 | `data/DensePASS/gtFine/val/125_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 38 | `data/DensePASS/gtFine/val/126_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 39 | `data/DensePASS/gtFine/val/12_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 40 | `data/DensePASS/gtFine/val/147_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 41 | `data/DensePASS/gtFine/val/155_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 42 | `data/DensePASS/gtFine/val/161_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 43 | `data/DensePASS/gtFine/val/162_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 44 | `data/DensePASS/gtFine/val/163_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 45 | `data/DensePASS/gtFine/val/170_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 46 | `data/DensePASS/gtFine/val/183_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 47 | `data/DensePASS/gtFine/val/188_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 48 | `data/DensePASS/gtFine/val/196_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 49 | `data/DensePASS/gtFine/val/197_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 50 | `data/DensePASS/gtFine/val/1_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 51 | `data/DensePASS/gtFine/val/201_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 52 | `data/DensePASS/gtFine/val/202_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 53 | `data/DensePASS/gtFine/val/204_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 54 | `data/DensePASS/gtFine/val/210_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 55 | `data/DensePASS/gtFine/val/21_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 56 | `data/DensePASS/gtFine/val/225_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 57 | `data/DensePASS/gtFine/val/226_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 58 | `data/DensePASS/gtFine/val/228_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 59 | `data/DensePASS/gtFine/val/22_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 60 | `data/DensePASS/gtFine/val/231_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 61 | `data/DensePASS/gtFine/val/246_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 62 | `data/DensePASS/gtFine/val/247_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 63 | `data/DensePASS/gtFine/val/24_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 64 | `data/DensePASS/gtFine/val/252_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 65 | `data/DensePASS/gtFine/val/258_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 66 | `data/DensePASS/gtFine/val/263_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 67 | `data/DensePASS/gtFine/val/265_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 68 | `data/DensePASS/gtFine/val/267_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 69 | `data/DensePASS/gtFine/val/272_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 70 | `data/DensePASS/gtFine/val/281_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 71 | `data/DensePASS/gtFine/val/286_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 72 | `data/DensePASS/gtFine/val/289_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 73 | `data/DensePASS/gtFine/val/294_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 74 | `data/DensePASS/gtFine/val/305_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 75 | `data/DensePASS/gtFine/val/307_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 76 | `data/DensePASS/gtFine/val/309_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 77 | `data/DensePASS/gtFine/val/319_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 78 | `data/DensePASS/gtFine/val/342_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 79 | `data/DensePASS/gtFine/val/343_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 80 | `data/DensePASS/gtFine/val/344_labelTrainIds.png` | 图像/可视化 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 81 | `adaptation/pseudo_label` | 目录 | 伪标签生成或筛选 | 中：候选材料，建议按需核验 | 是 |
| 82 | `data_extra/processed/Stanford360_SPAN8/labels_raw` | 目录 | 伪标签生成或筛选 | 中：候选材料，建议按需核验 | 待定 |
| 83 | `data_extra/processed/Stanford360_SPAN8/labels_trainid` | 目录 | 伪标签生成或筛选 | 中：候选材料，建议按需核验 | 待定 |
| 84 | `data_extra/processed/WildPASS13/labels_raw` | 目录 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 85 | `data_extra/processed/WildPASS13/labels_trainid` | 目录 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 86 | `data_extra/processed/WildPASS13/labels_trainid.backup_20260511_175555` | 目录 | 伪标签生成或筛选；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 87 | `data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1/gtFine_overlap` | 目录 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 88 | `data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1/gtFine_overlap` | 目录 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 89 | `experiments/pl_refiner/fullruns/R2R4_proto_gate_20260625_154537` | 目录 | 伪标签优化策略线索 | 高：可能对应伪标签优化实验或设计 | 是 |
| 90 | `experiments/pl_refiner/sanity_runs/R4_proto_gate_5step` | 目录 | 伪标签优化策略线索 | 高：可能对应伪标签优化实验或设计 | 是 |
| 91 | `experiments/pl_refiner/sanity_runs/R4_proto_threshold_calibration_5step` | 目录 | 伪标签优化策略线索 | 高：可能对应伪标签优化实验或设计 | 是 |
| 92 | `experiments/tecma/eval_predictions_confirmed/A_ccpm` | 目录 | 伪标签优化策略线索；评估指标或验证输出 | 高：可能对应伪标签优化实验或设计 | 是 |
| 93 | `experiments/tecma/runs/A_ccpm_seed1_20260615_201835` | 目录 | 伪标签优化策略线索 | 高：可能对应伪标签优化实验或设计 | 是 |
| 94 | `experiments/tecma/runs/A_ccpm_seed1_20260616_104911` | 目录 | 伪标签优化策略线索 | 高：可能对应伪标签优化实验或设计 | 是 |
| 95 | `experiments/tecma/sanity_runs/task23_A_Dtrain_Etrain_20260615_sanity/DensePASS/sam2_s/Cityscapes_13cls/epoch1/gtFine_overlap` | 目录 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 96 | `experiments/tecma/sanity_runs/task23_baseline_20260615_sanity/DensePASS/sam2_s/Cityscapes_13cls/epoch1/gtFine_overlap` | 目录 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 97 | `logs/pl_threshold_scan` | 目录 | 伪标签优化策略线索 | 高：可能对应伪标签优化实验或设计 | 是 |

## 4. 实验结果与日志材料
| 序号 | 文件路径 | 文件类型 | 可能内容 | 对论文 v1.0 的价值 | 是否建议后续深入读取 |
|---|---|---|---|---|---|
| 1 | `.claude/rules/experiments.md` | Markdown 文档 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 2 | `assets/text_prototypes_cityscapes13_vitb32.pt` | 模型权重/张量文件 | 伪标签优化策略线索；数据集相关配置/结果；checkpoint/权重 | 中：候选材料，建议按需核验 | 待定 |
| 3 | `aux_outputs/depth_anything_v2/WildPASS/wildpass_images.txt` | 日志/文本 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 4 | `configs/extra_exp/standford2d3d/standford360_span8_eval.yaml` | YAML 配置 | 评估指标或验证输出；实验配置 | 中：候选材料，建议按需核验 | 待定 |
| 5 | `configs/extra_exp/wildpass/wildpass13_eval.yaml` | YAML 配置 | 评估指标或验证输出；数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 待定 |
| 6 | `configs/extra_exp/wildpass/wildpass_train_base.yaml` | YAML 配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 待定 |
| 7 | `configs/extra_exp/wildpass/wildpass_train_final_optimized.yaml` | YAML 配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 待定 |
| 8 | `data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 9 | `data_pd_exp_final_2000/base_last_20260425_232941/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 10 | `dataset/Cityscapes.py` | Python 源码 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 11 | `dataset/DensePASS.py` | Python 源码 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 12 | `dataset/WildPASS.py` | Python 源码 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 13 | `dataset/__pycache__/Cityscapes.cpython-310.pyc` | 文件 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 14 | `dataset/__pycache__/DensePASS.cpython-310.pyc` | 文件 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 15 | `dataset/__pycache__/WildPASS.cpython-310.pyc` | 文件 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 16 | `debug_wildpass.py` | Python 源码 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 17 | `docs/claude/experiment-conventions.md` | Markdown 文档 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 18 | `docs/claude/experiment-operations.md` | Markdown 文档 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 19 | `docs/efficient_sam2_style_sparse_memory_retrieval_zh.md` | Markdown 文档 | memory 机制/日志；稀疏检索模块或配置；评估指标或验证输出 | 中-高：可能是已整理报告/表格 | 是 |
| 20 | `experiments/geo_memory/raw_results/GM1_A_Dtrain_Etrain/dry_run_region_metrics.json` | JSON/JSONL 结果或配置 | memory 机制/日志；评估指标或验证输出；实验配置 | 高：可能是论文实验事实来源 | 是 |
| 21 | `experiments/geo_memory/raw_results/GM1_A_Dtrain_Etrain/full_region_metrics.json` | JSON/JSONL 结果或配置 | memory 机制/日志；评估指标或验证输出；实验配置 | 高：可能是论文实验事实来源 | 是 |
| 22 | `experiments/geo_memory/runs/GM1_A_Dtrain_Etrain_task_02_3/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | memory 机制/日志；评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 23 | `experiments/geo_memory/scripts/__pycache__/eval_existing_region_metrics.cpython-310.pyc` | 文件 | memory 机制/日志；评估指标或验证输出 | 中：候选材料，建议按需核验 | 是 |
| 24 | `experiments/geo_memory/scripts/__pycache__/eval_region_metrics.cpython-310.pyc` | 文件 | memory 机制/日志；评估指标或验证输出 | 中：候选材料，建议按需核验 | 是 |
| 25 | `experiments/geo_memory/scripts/eval_existing_region_metrics.py` | Python 源码 | memory 机制/日志；评估指标或验证输出 | 中：候选材料，建议按需核验 | 是 |
| 26 | `experiments/geo_memory/scripts/eval_region_metrics.py` | Python 源码 | memory 机制/日志；评估指标或验证输出 | 中：候选材料，建议按需核验 | 是 |
| 27 | `experiments/geo_memory/unit_tests/test_geometry_ops_report.json` | JSON/JSONL 结果或配置 | memory 机制/日志；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 28 | `experiments/geo_memory/unit_tests/test_geometry_ops_report.md` | Markdown 文档 | memory 机制/日志 | 中-高：可能是已整理报告/表格 | 是 |
| 29 | `experiments/pl_refiner/fullruns/R0_identity_20260625_135318/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 30 | `experiments/pl_refiner/fullruns/R2R4_proto_gate_20260625_154537/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 伪标签优化策略线索；评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 31 | `experiments/pl_refiner/fullruns/R2_class_balanced_20260625_135318/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 32 | `experiments/pl_refiner/notes/step_02_identity_refiner_report.md` | Markdown 文档 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 33 | `experiments/pl_refiner/notes/step_03_config_dryrun_report.md` | Markdown 文档 | 实验配置 | 中-高：可能是已整理报告/表格 | 是 |
| 34 | `experiments/pl_refiner/sanity_runs/R0_off/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 35 | `experiments/pl_refiner/sanity_runs/R1_conf_mask_l3/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 36 | `experiments/pl_refiner/sanity_runs/R1_identity/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 37 | `experiments/pl_refiner/sanity_runs/R2_class_balanced_l3/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 38 | `experiments/pl_refiner/sanity_runs/R3_plq_basic_5step/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 39 | `experiments/pl_refiner/sanity_runs/R4_proto_gate_5step/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 伪标签优化策略线索；评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 40 | `experiments/pl_refiner/sanity_runs/R4_proto_similarity_5step/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 41 | `experiments/pl_refiner/sanity_runs/R4_proto_threshold_calibration_5step/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 伪标签优化策略线索；评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 42 | `experiments/pl_refiner/tables/pl_fullrun_diag_summary.csv` | CSV 表格 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 43 | `experiments/pl_refiner/tables/pl_fullrun_main_metrics.csv` | CSV 表格 | 评估指标或验证输出 | 中-高：可能是已整理报告/表格 | 是 |
| 44 | `experiments/pl_refiner/tables/pl_fullrun_per_class_iou.csv` | CSV 表格 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 45 | `experiments/tecma/figures/failure_cases_diagnostic/failure_case_candidates.csv` | CSV 表格 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 46 | `experiments/tecma/figures/qualitative_diagnostic/diagnostic_sample_manifest.csv` | CSV 表格 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 47 | `experiments/tecma/notes/phase0_report.md` | Markdown 文档 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 48 | `experiments/tecma/notes/phase1_report.md` | Markdown 文档 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 49 | `experiments/tecma/notes/step_05_profile_implementation_report.md` | Markdown 文档 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 50 | `experiments/tecma/notes/step_07_main_experiment_script.md` | Markdown 文档 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 51 | `experiments/tecma/notes/step_09_main_results_draft_report.md` | Markdown 文档 | 评估指标或验证输出 | 中-高：可能是已整理报告/表格 | 是 |
| 52 | `experiments/tecma/notes/step_14_token_budget_script_report.md` | Markdown 文档 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 53 | `experiments/tecma/notes/step_15_token_budget_preflight_report.md` | Markdown 文档 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 54 | `experiments/tecma/notes/step_15_token_budget_results_report.md` | Markdown 文档 | 评估指标或验证输出 | 中-高：可能是已整理报告/表格 | 是 |
| 55 | `experiments/tecma/notes/step_16_profile_summary_report.md` | Markdown 文档 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 56 | `experiments/tecma/notes/step_17_training_time_breakdown_report.md` | Markdown 文档 | 速度/耗时记录 | 中-高：可能是已整理报告/表格 | 是 |
| 57 | `experiments/tecma/notes/step_19_per_class_iou_diagnostic_report.md` | Markdown 文档 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 58 | `experiments/tecma/notes/step_19_per_class_iou_report.md` | Markdown 文档 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 59 | `experiments/tecma/notes/step_20_21_confirmed_prediction_visualization_report.md` | Markdown 文档 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 60 | `experiments/tecma/notes/step_20_21_diagnostic_visualization_report.md` | Markdown 文档 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 61 | `experiments/tecma/runs/A_Dtrain_Etrain_PLon_seed1_20260616_031742/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 62 | `experiments/tecma/runs/A_Dtrain_Etrain_PLon_seed1_20260616_031742/run.log` | 日志/文本 | 实验结果/日志候选材料 | 高：可能是论文实验事实来源 | 是 |
| 63 | `experiments/tecma/runs/A_Dtrain_Etrain_seed1_20260616_121135/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 64 | `experiments/tecma/runs/A_Dtrain_Etrain_seed1_20260616_121135/run.log` | 日志/文本 | 实验结果/日志候选材料 | 高：可能是论文实验事实来源 | 是 |
| 65 | `experiments/tecma/runs/A_Dtrain_Etrain_seed1_sanity32_165302/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 66 | `experiments/tecma/runs/A_Dtrain_Etrain_seed1_sanity32_165302/run.log` | 日志/文本 | 实验结果/日志候选材料 | 高：可能是论文实验事实来源 | 是 |
| 67 | `experiments/tecma/runs/A_Dtrain_seed1_20260615_215927/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 68 | `experiments/tecma/runs/A_Dtrain_seed1_20260615_215927/run.log` | 日志/文本 | 实验结果/日志候选材料 | 高：可能是论文实验事实来源 | 是 |
| 69 | `experiments/tecma/runs/A_Dtrain_seed1_20260616_111019/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 70 | `experiments/tecma/runs/A_Dtrain_seed1_20260616_111019/run.log` | 日志/文本 | 实验结果/日志候选材料 | 高：可能是论文实验事实来源 | 是 |
| 71 | `experiments/tecma/runs/A_Etrain_seed1_20260615_234524/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 72 | `experiments/tecma/runs/A_Etrain_seed1_20260615_234524/run.log` | 日志/文本 | 实验结果/日志候选材料 | 高：可能是论文实验事实来源 | 是 |
| 73 | `experiments/tecma/runs/A_Etrain_seed1_20260616_113023/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 74 | `experiments/tecma/runs/A_Etrain_seed1_20260616_113023/run.log` | 日志/文本 | 实验结果/日志候选材料 | 高：可能是论文实验事实来源 | 是 |
| 75 | `experiments/tecma/runs/A_ccpm_seed1_20260615_201835/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 伪标签优化策略线索；评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 76 | `experiments/tecma/runs/A_ccpm_seed1_20260615_201835/run.log` | 日志/文本 | 伪标签优化策略线索 | 高：可能是论文实验事实来源 | 是 |
| 77 | `experiments/tecma/runs/A_ccpm_seed1_20260616_104911/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 伪标签优化策略线索；评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 78 | `experiments/tecma/runs/A_ccpm_seed1_20260616_104911/run.log` | 日志/文本 | 伪标签优化策略线索 | 高：可能是论文实验事实来源 | 是 |
| 79 | `experiments/tecma/runs/Dtrain_Etrain_seed1_20260616_012942/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 80 | `experiments/tecma/runs/Dtrain_Etrain_seed1_20260616_012942/run.log` | 日志/文本 | 实验结果/日志候选材料 | 高：可能是论文实验事实来源 | 是 |
| 81 | `experiments/tecma/runs/Dtrain_Etrain_seed1_20260616_115113/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 82 | `experiments/tecma/runs/Dtrain_Etrain_seed1_20260616_115113/run.log` | 日志/文本 | 实验结果/日志候选材料 | 高：可能是论文实验事实来源 | 是 |
| 83 | `experiments/tecma/runs/_subsets/Cityscapes_13cls_seed1_n2000.txt` | 日志/文本 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 84 | `experiments/tecma/runs/_subsets/Cityscapes_13cls_seed1_n8.txt` | 日志/文本 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 85 | `experiments/tecma/runs/baseline_omnisam_seed1_20260616_102834/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 86 | `experiments/tecma/runs/baseline_omnisam_seed1_20260616_102834/run.log` | 日志/文本 | 实验结果/日志候选材料 | 高：可能是论文实验事实来源 | 是 |
| 87 | `experiments/tecma/runs/baseline_omnisam_seed1_sanity32_164244/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 88 | `experiments/tecma/runs/baseline_omnisam_seed1_sanity32_164244/run.log` | 日志/文本 | 实验结果/日志候选材料 | 高：可能是论文实验事实来源 | 是 |
| 89 | `experiments/tecma/runs/token_budget/A_Etrain_keep025_seed1_20260617_114629/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 90 | `experiments/tecma/runs/token_budget/A_Etrain_keep025_seed1_20260617_114629/run.log` | 日志/文本 | 实验结果/日志候选材料 | 高：可能是论文实验事实来源 | 是 |
| 91 | `experiments/tecma/runs/token_budget/A_Etrain_keep075_seed1_20260617_112455/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 92 | `experiments/tecma/runs/token_budget/A_Etrain_keep075_seed1_20260617_112455/run.log` | 日志/文本 | 实验结果/日志候选材料 | 高：可能是论文实验事实来源 | 是 |
| 93 | `experiments/tecma/runs/token_budget/_subsets/Cityscapes_13cls_seed1_n2000.txt` | 日志/文本 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 94 | `experiments/tecma/sanity_runs/task23_A_Dtrain_Etrain_20260615_sanity/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 95 | `experiments/tecma/sanity_runs/task23_baseline_20260615_sanity/checkpoints/Cityscapes_13cls/sam2_s/metrics.jsonl` | JSON/JSONL 结果或配置 | 评估指标或验证输出；数据集相关配置/结果；checkpoint/权重 | 高：可能是论文实验事实来源 | 是 |
| 96 | `experiments/tecma/tables/clean_ablation_results.csv` | CSV 表格 | 评估指标或验证输出 | 中-高：可能是已整理报告/表格 | 是 |
| 97 | `experiments/tecma/tables/memory_token_attention_cost.csv` | CSV 表格 | memory 机制/日志 | 中-高：可能是已整理报告/表格 | 是 |
| 98 | `experiments/tecma/tables/paper_ablation_table_draft.md` | Markdown 文档 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 99 | `experiments/tecma/tables/per_class_iou.csv` | CSV 表格 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 100 | `experiments/tecma/tables/per_class_iou_diagnostic.csv` | CSV 表格 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 101 | `experiments/tecma/tables/profile_summary.csv` | CSV 表格 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 102 | `experiments/tecma/tables/token_budget_tradeoff.csv` | CSV 表格 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 103 | `experiments/tecma/tables/training_time_breakdown.csv` | CSV 表格 | 速度/耗时记录 | 中-高：可能是已整理报告/表格 | 是 |
| 104 | `logs/Standford2d3d_results/summaries/standford2d3d_train_time.csv` | CSV 表格 | 速度/耗时记录；评估指标或验证输出 | 中-高：可能是已整理报告/表格 | 是 |
| 105 | `logs/Standford2d3d_results/summaries/standford2d3d_val_ckpts.csv` | CSV 表格 | 评估指标或验证输出 | 中-高：可能是已整理报告/表格 | 是 |
| 106 | `logs/abcd_summary.csv` | CSV 表格 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 107 | `logs/cdsweep_20260422_235848.phase1_results.csv` | CSV 表格 | 评估指标或验证输出 | 中-高：可能是已整理报告/表格 | 是 |
| 108 | `logs/cdsweep_20260422_235848.topk.csv` | CSV 表格 | 实验结果/日志候选材料 | 中-高：可能是已整理报告/表格 | 是 |
| 109 | `logs/cdsweep_only_20260422_235545.phase1_results.csv` | CSV 表格 | 评估指标或验证输出 | 中-高：可能是已整理报告/表格 | 是 |
| 110 | `logs/cdsweep_only_clean_20260422_235636.phase1_results.csv` | CSV 表格 | 评估指标或验证输出 | 中-高：可能是已整理报告/表格 | 是 |
| 111 | `logs/omnisam_final_2000/subsets/Cityscapes_13cls_seed1_n2000.txt` | 日志/文本 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 112 | `logs/omnisam_final_2000/subsets/Cityscapes_13cls_seed1_warmup100.txt` | 日志/文本 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 113 | `logs/omnisam_time_ablation/plugin_ablation_subsets/Cityscapes_13cls_seed0_small50.txt` | 日志/文本 | 速度/耗时记录；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 114 | `logs/omnisam_time_ablation/plugin_ablation_subsets/Cityscapes_13cls_seed1_small20.txt` | 日志/文本 | 速度/耗时记录；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 115 | `logs/omnisam_time_ablation/plugin_ablation_subsets/Cityscapes_13cls_seed1_small50.txt` | 日志/文本 | 速度/耗时记录；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 116 | `logs/omnisam_time_ablation_seed1_triplet/plugin_ablation_subsets/Cityscapes_13cls_seed1_small200.txt` | 日志/文本 | 速度/耗时记录；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 117 | `logs/pl_threshold_scan/scan_results.jsonl` | JSON/JSONL 结果或配置 | 伪标签优化策略线索；评估指标或验证输出 | 中：候选材料，建议按需核验 | 待定 |
| 118 | `logs/uda_densepass_20260408_094541.log` | 日志/文本 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 119 | `logs/uda_densepass_20260408_143548.log` | 日志/文本 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 120 | `logs/uda_densepass_originalish_20260410_133124.log` | 日志/文本 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 121 | `aux_outputs/depth_anything_v2/WildPASS` | 目录 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 122 | `aux_outputs/depth_anything_v2/WildPASS_debug_strict` | 目录 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 123 | `configs/extra_exp/wildpass` | 目录 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 待定 |
| 124 | `data/DensePASS` | 目录 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 125 | `data/WildPASS` | 目录 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 126 | `data/cityscapes` | 目录 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 127 | `data_extra/processed/WildPASS13` | 目录 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 128 | `data_extra/raw/WildPASS` | 目录 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 129 | `data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/DensePASS` | 目录 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 130 | `data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls` | 目录 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 131 | `data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/checkpoints/Cityscapes_13cls` | 目录 | 数据集相关配置/结果；checkpoint/权重 | 中：候选材料，建议按需核验 | 待定 |
| 132 | `data_pd_exp_final_2000/base_last_20260425_232941/DensePASS` | 目录 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 133 | `data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls` | 目录 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 134 | `data_pd_exp_final_2000/base_last_20260425_232941/checkpoints/Cityscapes_13cls` | 目录 | 数据集相关配置/结果；checkpoint/权重 | 中：候选材料，建议按需核验 | 待定 |
| 135 | `dataset/cityscapes_list` | 目录 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 136 | `dataset/densepass_list` | 目录 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 137 | `dataset/wildpass_list` | 目录 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 138 | `docs/reports` | 目录 | 实验结果/日志候选材料 | 中：候选材料，建议按需核验 | 是 |
| 139 | `exp2/Cityscapes` | 目录 | 数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |
| 140 | `experiments/geo_memory/raw_results/GM1_A_Dtrain_Etrain/dry_run/gt_densepass13` | 目录 | memory 机制/日志；评估指标或验证输出；数据集相关配置/结果 | 中：候选材料，建议按需核验 | 待定 |

## 5. Checkpoint 与配置材料
| 序号 | 文件路径 | 文件类型 | 可能内容 | 对论文 v1.0 的价值 | 是否建议后续深入读取 |
|---|---|---|---|---|---|
| 1 | `  /geo_memory/diagnostics/token_region_summary.json` | JSON/JSONL 结果或配置 | memory 机制/日志；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 2 | `assets/openclip_class_prompts_10.json` | JSON/JSONL 结果或配置 | 实验配置 | 中：候选材料，建议按需核验 | 是 |
| 3 | `assets/openclip_class_prompts_13_optimized.json` | JSON/JSONL 结果或配置 | 实验配置 | 中：候选材料，建议按需核验 | 是 |
| 4 | `assets/text_prototypes_cityscapes13_vitb32.pt` | 模型权重/张量文件 | 伪标签优化策略线索；数据集相关配置/结果；checkpoint/权重 | 高：可用于确认 ckpt 范围，注意不要覆盖 | 是 |
| 5 | `assets/text_prototypes_openclip_text_latest.pt` | 模型权重/张量文件 | 伪标签优化策略线索；checkpoint/权重 | 高：可用于确认 ckpt 范围，注意不要覆盖 | 是 |
| 6 | `aux_outputs/depth_anything_v2/WildPASS/manifest.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 7 | `aux_outputs/depth_anything_v2/WildPASS/metadata.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 8 | `aux_outputs/depth_anything_v2/WildPASS_debug_strict/manifest.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 9 | `aux_outputs/depth_anything_v2/WildPASS_debug_strict/metadata.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 10 | `aux_outputs/openclip_text/class_prompt_metadata.json` | JSON/JSONL 结果或配置 | 实验配置 | 中：候选材料，建议按需核验 | 是 |
| 11 | `aux_outputs/openclip_text_v2/class_prompt_metadata.json` | JSON/JSONL 结果或配置 | 实验配置 | 中：候选材料，建议按需核验 | 是 |
| 12 | `ckpt_zoo/CKPT_INDEX.txt` | 日志/文本 | 配置或权重候选材料 | 中：候选材料，建议按需核验 | 待定 |
| 13 | `ckpt_zoo/index.json` | JSON/JSONL 结果或配置 | 实验配置 | 中：候选材料，建议按需核验 | 是 |
| 14 | `config_mem_pool2_before_attention.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 15 | `configs/ablations/omnisam_A_plus4train_plus5train.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 16 | `configs/ablations/omnisam_A_plus4train_plus5train_plus_depth.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 17 | `configs/ablations/omnisam_all5_full.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 18 | `configs/ablations/omnisam_base.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 19 | `configs/ablations/omnisam_base_plus1_confmask.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 20 | `configs/ablations/omnisam_base_plus4_efficienttam.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 21 | `configs/ablations/omnisam_base_plus5_efficientsam2_sparse.yaml` | YAML 配置 | 稀疏检索模块或配置；实验配置 | 高：可用于复现实验设置 | 是 |
| 22 | `configs/ablations/omnisam_plus_depth_prior.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 23 | `configs/depth_prior/base_depth_adapter.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 24 | `configs/depth_prior/base_depth_conf.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 25 | `configs/depth_prior/base_depth_conf_adapter.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 26 | `configs/depth_prior/base_depth_none.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 27 | `configs/experiments/depth_prior/ablation50_A0_base.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 28 | `configs/experiments/depth_prior/ablation50_A1_base_accel.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 29 | `configs/experiments/depth_prior/ablation50_A2_base_accel_depth.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 30 | `configs/experiments/text_modality/text_ablation_A0_baseline.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 31 | `configs/experiments/text_modality/text_ablation_A1_text_branch.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 32 | `configs/experiments/text_modality/text_ablation_A2_text_branch_gate_soft.yaml` | YAML 配置 | 伪标签优化策略线索；实验配置 | 高：可用于复现实验设置 | 是 |
| 33 | `configs/experiments/text_modality/text_ablation_A3_text_branch_gate_hard.yaml` | YAML 配置 | 伪标签优化策略线索；实验配置 | 高：可用于复现实验设置 | 是 |
| 34 | `configs/experiments/text_modality/text_ablation_A4_shuffle_proto.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 35 | `configs/experiments/text_modality/text_ablation_A5_random_proto.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 36 | `configs/experiments/text_modality_crossattn/crossattn_A0_baseline.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 37 | `configs/experiments/text_modality_crossattn/crossattn_B1_real_proto.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 38 | `configs/experiments/text_modality_crossattn/crossattn_B4_shuffle_proto.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 39 | `configs/experiments/text_modality_crossattn/crossattn_B5_random_proto.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 40 | `configs/experiments/text_modality_grounding/grounding_A0_baseline.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 41 | `configs/experiments/text_modality_grounding/grounding_C1_real_proto.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 42 | `configs/experiments/text_modality_grounding/grounding_C4_shuffle_proto.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 43 | `configs/experiments/text_modality_grounding/grounding_C5_random_proto.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 44 | `configs/extra_exp/standford2d3d/standford360_span8_eval.yaml` | YAML 配置 | 评估指标或验证输出；实验配置 | 高：可用于复现实验设置 | 是 |
| 45 | `configs/extra_exp/standford2d3d/stanford360_span8_train_base.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 46 | `configs/extra_exp/standford2d3d/stanford360_span8_train_final_optimized.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 47 | `configs/extra_exp/wildpass/wildpass13_eval.yaml` | YAML 配置 | 评估指标或验证输出；数据集相关配置/结果；实验配置 | 高：可用于复现实验设置 | 是 |
| 48 | `configs/extra_exp/wildpass/wildpass_train_base.yaml` | YAML 配置 | 数据集相关配置/结果；实验配置 | 高：可用于复现实验设置 | 是 |
| 49 | `configs/extra_exp/wildpass/wildpass_train_final_optimized.yaml` | YAML 配置 | 数据集相关配置/结果；实验配置 | 高：可用于复现实验设置 | 是 |
| 50 | `configs/psr_pld/base.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 51 | `configs/psr_pld/scan_template.yaml` | YAML 配置 | 实验配置 | 高：可用于复现实验设置 | 是 |
| 52 | `data/cityscapes/gtFine/test/berlin/berlin_000000_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 53 | `data/cityscapes/gtFine/test/berlin/berlin_000001_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 54 | `data/cityscapes/gtFine/test/berlin/berlin_000002_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 55 | `data/cityscapes/gtFine/test/berlin/berlin_000003_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 56 | `data/cityscapes/gtFine/test/berlin/berlin_000004_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 57 | `data/cityscapes/gtFine/test/berlin/berlin_000005_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 58 | `data/cityscapes/gtFine/test/berlin/berlin_000006_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 59 | `data/cityscapes/gtFine/test/berlin/berlin_000007_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 60 | `data/cityscapes/gtFine/test/berlin/berlin_000008_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 61 | `data/cityscapes/gtFine/test/berlin/berlin_000009_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 62 | `data/cityscapes/gtFine/test/berlin/berlin_000010_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 63 | `data/cityscapes/gtFine/test/berlin/berlin_000011_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 64 | `data/cityscapes/gtFine/test/berlin/berlin_000012_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 65 | `data/cityscapes/gtFine/test/berlin/berlin_000013_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 66 | `data/cityscapes/gtFine/test/berlin/berlin_000014_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 67 | `data/cityscapes/gtFine/test/berlin/berlin_000015_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 68 | `data/cityscapes/gtFine/test/berlin/berlin_000016_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 69 | `data/cityscapes/gtFine/test/berlin/berlin_000017_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 70 | `data/cityscapes/gtFine/test/berlin/berlin_000018_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 71 | `data/cityscapes/gtFine/test/berlin/berlin_000019_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 72 | `data/cityscapes/gtFine/test/berlin/berlin_000020_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 73 | `data/cityscapes/gtFine/test/berlin/berlin_000021_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 74 | `data/cityscapes/gtFine/test/berlin/berlin_000022_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 75 | `data/cityscapes/gtFine/test/berlin/berlin_000023_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 76 | `data/cityscapes/gtFine/test/berlin/berlin_000024_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 77 | `data/cityscapes/gtFine/test/berlin/berlin_000025_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 78 | `data/cityscapes/gtFine/test/berlin/berlin_000026_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 79 | `data/cityscapes/gtFine/test/berlin/berlin_000027_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 80 | `data/cityscapes/gtFine/test/berlin/berlin_000028_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 81 | `data/cityscapes/gtFine/test/berlin/berlin_000029_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 82 | `data/cityscapes/gtFine/test/berlin/berlin_000030_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 83 | `data/cityscapes/gtFine/test/berlin/berlin_000031_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 84 | `data/cityscapes/gtFine/test/berlin/berlin_000032_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 85 | `data/cityscapes/gtFine/test/berlin/berlin_000033_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 86 | `data/cityscapes/gtFine/test/berlin/berlin_000034_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 87 | `data/cityscapes/gtFine/test/berlin/berlin_000035_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 88 | `data/cityscapes/gtFine/test/berlin/berlin_000036_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 89 | `data/cityscapes/gtFine/test/berlin/berlin_000037_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 90 | `data/cityscapes/gtFine/test/berlin/berlin_000038_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 91 | `data/cityscapes/gtFine/test/berlin/berlin_000039_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 92 | `data/cityscapes/gtFine/test/berlin/berlin_000040_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 93 | `data/cityscapes/gtFine/test/berlin/berlin_000041_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 94 | `data/cityscapes/gtFine/test/berlin/berlin_000042_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 95 | `data/cityscapes/gtFine/test/berlin/berlin_000043_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 96 | `data/cityscapes/gtFine/test/berlin/berlin_000044_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 97 | `data/cityscapes/gtFine/test/berlin/berlin_000045_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 98 | `data/cityscapes/gtFine/test/berlin/berlin_000046_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 99 | `data/cityscapes/gtFine/test/berlin/berlin_000047_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 100 | `data/cityscapes/gtFine/test/berlin/berlin_000048_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 101 | `data/cityscapes/gtFine/test/berlin/berlin_000049_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 102 | `data/cityscapes/gtFine/test/berlin/berlin_000050_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 103 | `data/cityscapes/gtFine/test/berlin/berlin_000051_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 104 | `data/cityscapes/gtFine/test/berlin/berlin_000052_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 105 | `data/cityscapes/gtFine/test/berlin/berlin_000053_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 106 | `data/cityscapes/gtFine/test/berlin/berlin_000054_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 107 | `data/cityscapes/gtFine/test/berlin/berlin_000055_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 108 | `data/cityscapes/gtFine/test/berlin/berlin_000056_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 109 | `data/cityscapes/gtFine/test/berlin/berlin_000057_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 110 | `data/cityscapes/gtFine/test/berlin/berlin_000058_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 111 | `data/cityscapes/gtFine/test/berlin/berlin_000059_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 112 | `data/cityscapes/gtFine/test/berlin/berlin_000060_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 113 | `data/cityscapes/gtFine/test/berlin/berlin_000061_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 114 | `data/cityscapes/gtFine/test/berlin/berlin_000062_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 115 | `data/cityscapes/gtFine/test/berlin/berlin_000063_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 116 | `data/cityscapes/gtFine/test/berlin/berlin_000064_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 117 | `data/cityscapes/gtFine/test/berlin/berlin_000065_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 118 | `data/cityscapes/gtFine/test/berlin/berlin_000066_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 119 | `data/cityscapes/gtFine/test/berlin/berlin_000067_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 120 | `data/cityscapes/gtFine/test/berlin/berlin_000068_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 121 | `data/cityscapes/gtFine/test/berlin/berlin_000069_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 122 | `data/cityscapes/gtFine/test/berlin/berlin_000070_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 123 | `data/cityscapes/gtFine/test/berlin/berlin_000071_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 124 | `data/cityscapes/gtFine/test/berlin/berlin_000072_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 125 | `data/cityscapes/gtFine/test/berlin/berlin_000073_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 126 | `data/cityscapes/gtFine/test/berlin/berlin_000074_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 127 | `data/cityscapes/gtFine/test/berlin/berlin_000075_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 128 | `data/cityscapes/gtFine/test/berlin/berlin_000076_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 129 | `data/cityscapes/gtFine/test/berlin/berlin_000077_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 130 | `data/cityscapes/gtFine/test/berlin/berlin_000078_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 131 | `data/cityscapes/gtFine/test/berlin/berlin_000079_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 132 | `data/cityscapes/gtFine/test/berlin/berlin_000080_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 133 | `data/cityscapes/gtFine/test/berlin/berlin_000081_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 134 | `data/cityscapes/gtFine/test/berlin/berlin_000082_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 135 | `data/cityscapes/gtFine/test/berlin/berlin_000083_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 136 | `data/cityscapes/gtFine/test/berlin/berlin_000084_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 137 | `data/cityscapes/gtFine/test/berlin/berlin_000085_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 138 | `data/cityscapes/gtFine/test/berlin/berlin_000086_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 139 | `data/cityscapes/gtFine/test/berlin/berlin_000087_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 140 | `data/cityscapes/gtFine/test/berlin/berlin_000088_000019_gtFine_polygons.json` | JSON/JSONL 结果或配置 | 数据集相关配置/结果；实验配置 | 中：候选材料，建议按需核验 | 是 |
| 141 | `ckpt_zoo` | 目录 | 配置或权重候选材料 | 中：候选材料，建议按需核验 | 待定 |
| 142 | `configs` | 目录 | 实验配置 | 中：候选材料，建议按需核验 | 是 |
| 143 | `data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/checkpoints` | 目录 | checkpoint/权重 | 高：可用于确认 ckpt 范围，注意不要覆盖 | 是 |
| 144 | `data_pd_exp_final_2000/base_last_20260425_232941/checkpoints` | 目录 | checkpoint/权重 | 高：可用于确认 ckpt 范围，注意不要覆盖 | 是 |
| 145 | `experiments/geo_memory/runs/GM1_A_Dtrain_Etrain_task_02_3/checkpoints` | 目录 | memory 机制/日志；checkpoint/权重 | 高：可用于确认 ckpt 范围，注意不要覆盖 | 是 |
| 146 | `experiments/pl_refiner/configs` | 目录 | 实验配置 | 中：候选材料，建议按需核验 | 是 |
| 147 | `experiments/pl_refiner/fullruns/R0_identity_20260625_135318/checkpoints` | 目录 | checkpoint/权重 | 高：可用于确认 ckpt 范围，注意不要覆盖 | 是 |
| 148 | `experiments/pl_refiner/fullruns/R2R4_proto_gate_20260625_154537/checkpoints` | 目录 | 伪标签优化策略线索；checkpoint/权重 | 高：可用于确认 ckpt 范围，注意不要覆盖 | 是 |
| 149 | `experiments/pl_refiner/fullruns/R2_class_balanced_20260625_135318/checkpoints` | 目录 | checkpoint/权重 | 高：可用于确认 ckpt 范围，注意不要覆盖 | 是 |
| 150 | `experiments/pl_refiner/sanity_runs/R0_off/checkpoints` | 目录 | checkpoint/权重 | 高：可用于确认 ckpt 范围，注意不要覆盖 | 是 |
| 151 | `experiments/pl_refiner/sanity_runs/R1_conf_mask_l3/checkpoints` | 目录 | checkpoint/权重 | 高：可用于确认 ckpt 范围，注意不要覆盖 | 是 |
| 152 | `experiments/pl_refiner/sanity_runs/R1_identity.interrupted_20260623_144317/checkpoints` | 目录 | checkpoint/权重 | 高：可用于确认 ckpt 范围，注意不要覆盖 | 是 |
| 153 | `experiments/pl_refiner/sanity_runs/R1_identity/checkpoints` | 目录 | checkpoint/权重 | 高：可用于确认 ckpt 范围，注意不要覆盖 | 是 |
| 154 | `experiments/pl_refiner/sanity_runs/R2_class_balanced_l3/checkpoints` | 目录 | checkpoint/权重 | 高：可用于确认 ckpt 范围，注意不要覆盖 | 是 |
| 155 | `experiments/pl_refiner/sanity_runs/R3_plq_basic_5step/checkpoints` | 目录 | checkpoint/权重 | 高：可用于确认 ckpt 范围，注意不要覆盖 | 是 |
| 156 | `experiments/pl_refiner/sanity_runs/R4_proto_gate_5step/checkpoints` | 目录 | 伪标签优化策略线索；checkpoint/权重 | 高：可用于确认 ckpt 范围，注意不要覆盖 | 是 |
| 157 | `experiments/pl_refiner/sanity_runs/R4_proto_similarity_5step/checkpoints` | 目录 | checkpoint/权重 | 高：可用于确认 ckpt 范围，注意不要覆盖 | 是 |
| 158 | `experiments/pl_refiner/sanity_runs/R4_proto_threshold_calibration_5step/checkpoints` | 目录 | 伪标签优化策略线索；checkpoint/权重 | 高：可用于确认 ckpt 范围，注意不要覆盖 | 是 |
| 159 | `experiments/tecma/configs` | 目录 | 实验配置 | 中：候选材料，建议按需核验 | 是 |
| 160 | `experiments/tecma/runs/A_Dtrain_Etrain_PLon_seed1_20260616_031742/checkpoints` | 目录 | checkpoint/权重 | 高：可用于确认 ckpt 范围，注意不要覆盖 | 是 |

## 6. 已发现但暂时不确定价值的材料
- `CLAUDE.md`：仅从文件名/类型看可能是说明文档或记录，但未命中本次重点关键词，暂未判断其对论文 v1.0 的直接价值。
- `aux_outputs/depth_anything_v2/WildPASS/process_log.txt`：仅从文件名/类型看可能是说明文档或记录，但未命中本次重点关键词，暂未判断其对论文 v1.0 的直接价值。
- `aux_outputs/depth_anything_v2/WildPASS_debug_strict/process_log.txt`：仅从文件名/类型看可能是说明文档或记录，但未命中本次重点关键词，暂未判断其对论文 v1.0 的直接价值。
- `ckpt_zoo/README.txt`：仅从文件名/类型看可能是说明文档或记录，但未命中本次重点关键词，暂未判断其对论文 v1.0 的直接价值。
- `data_extra/processed/Stanford360_SPAN8/lists/train.txt`：仅从文件名/类型看可能是说明文档或记录，但未命中本次重点关键词，暂未判断其对论文 v1.0 的直接价值。
- `data_extra/processed/Stanford360_SPAN8/lists/val.txt`：仅从文件名/类型看可能是说明文档或记录，但未命中本次重点关键词，暂未判断其对论文 v1.0 的直接价值。
- `data_extra/processed/WildPASS13/lists/val.txt`：仅从文件名/类型看可能是说明文档或记录，但未命中本次重点关键词，暂未判断其对论文 v1.0 的直接价值。
- `data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/unused_path.txt`：仅从文件名/类型看可能是说明文档或记录，但未命中本次重点关键词，暂未判断其对论文 v1.0 的直接价值。
- `data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/used_path.txt`：仅从文件名/类型看可能是说明文档或记录，但未命中本次重点关键词，暂未判断其对论文 v1.0 的直接价值。
- `data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/unused_path.txt`：仅从文件名/类型看可能是说明文档或记录，但未命中本次重点关键词，暂未判断其对论文 v1.0 的直接价值。
- `data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/used_path.txt`：仅从文件名/类型看可能是说明文档或记录，但未命中本次重点关键词，暂未判断其对论文 v1.0 的直接价值。
- `dataset/cityscapes_list/label.txt`：仅从文件名/类型看可能是说明文档或记录，但未命中本次重点关键词，暂未判断其对论文 v1.0 的直接价值。
- `dataset/cityscapes_list/train.txt`：仅从文件名/类型看可能是说明文档或记录，但未命中本次重点关键词，暂未判断其对论文 v1.0 的直接价值。
- `dataset/cityscapes_list/val.txt`：仅从文件名/类型看可能是说明文档或记录，但未命中本次重点关键词，暂未判断其对论文 v1.0 的直接价值。
- `dataset/densepass_list/label.txt`：仅从文件名/类型看可能是说明文档或记录，但未命中本次重点关键词，暂未判断其对论文 v1.0 的直接价值。
- `dataset/densepass_list/train.txt`：仅从文件名/类型看可能是说明文档或记录，但未命中本次重点关键词，暂未判断其对论文 v1.0 的直接价值。
- `dataset/densepass_list/val.txt`：仅从文件名/类型看可能是说明文档或记录，但未命中本次重点关键词，暂未判断其对论文 v1.0 的直接价值。
- `dataset/s2d3d_pan_list/label.txt`：仅从文件名/类型看可能是说明文档或记录，但未命中本次重点关键词，暂未判断其对论文 v1.0 的直接价值。
- `dataset/s2d3d_pan_list/train.txt`：仅从文件名/类型看可能是说明文档或记录，但未命中本次重点关键词，暂未判断其对论文 v1.0 的直接价值。
- `dataset/s2d3d_pan_list/trainval.txt`：仅从文件名/类型看可能是说明文档或记录，但未命中本次重点关键词，暂未判断其对论文 v1.0 的直接价值。
- `dataset/s2d3d_pan_list/val.txt`：仅从文件名/类型看可能是说明文档或记录，但未命中本次重点关键词，暂未判断其对论文 v1.0 的直接价值。
- `dataset/s2d3d_pin_list/label.txt`：仅从文件名/类型看可能是说明文档或记录，但未命中本次重点关键词，暂未判断其对论文 v1.0 的直接价值。
- `dataset/s2d3d_pin_list/train.txt`：仅从文件名/类型看可能是说明文档或记录，但未命中本次重点关键词，暂未判断其对论文 v1.0 的直接价值。
- `dataset/s2d3d_pin_list/trainval.txt`：仅从文件名/类型看可能是说明文档或记录，但未命中本次重点关键词，暂未判断其对论文 v1.0 的直接价值。
- `dataset/s2d3d_pin_list/val.txt`：仅从文件名/类型看可能是说明文档或记录，但未命中本次重点关键词，暂未判断其对论文 v1.0 的直接价值。

## 7. 下一步建议
只建议下一步，不在本子任务执行：
- 子任务 0-2：从 memory-efficient 相关材料中提取可写入论文的确定实验事实；
- 子任务 0-3：从 pseudo-label refinement 相关材料中提取可写入论文的确定实验事实；
- 子任务 0-4：扫描 checkpoint inventory，确认当前可直接复现实验的数据集和 ckpt 范围。
