# OmniSAM 表格格式与术语统一审计（1-10d）

## 1. 内部实验名与论文方法名映射

| 内部 run ID | 实际启用配置 | 正式表格名称 |
|---|---|---|
| `baseline_omnisam` | A/置信掩码关，D/MTC 关，E/SMR 关 | OmniSAM |
| `A_ccpm` | 仅启用 `pseudo_conf_mask`；不改变 Memory Tokens | OmniSAM + Confidence Mask |
| `A_Dtrain` | Confidence Mask + train-only D/MTC | OmniSAM + Confidence Mask + MTC |
| `A_Etrain` | Confidence Mask + train-only E/SMR | OmniSAM + Confidence Mask + SMR |
| `Dtrain_Etrain` | train-only D/MTC + E/SMR；不启用 Confidence Mask | OmniSAM + MTC+SMR |
| `A_Dtrain_Etrain` | Confidence Mask + train-only D/MTC + E/SMR | OmniSAM + Confidence Mask + MTC+SMR |

映射证据：

- `experiments/tecma/runs/*/run.log` 的 `[plugins] enabled=[...]`：A 组只出现 `pseudo_conf_mask`；D 组出现 `efficient_tam_memory`；E 组出现 `efficient_sam2_sparse`。
- `paper/OmniSAM_Paper_v1_0_CN/notes/memory_experiment_facts.md` 的模块配置与 clean-ablation 表，确认 D 的 token 压缩行为对应 MTC，E 的 keep-ratio 稀疏检索对应 SMR。
- `experiments/tecma/tables/clean_ablation_results.csv` 和 `token_budget_tradeoff.csv` 提供六组配置、Memory Tokens、mIoU、BFScore、Step Time 与 Memory Attention Time。
- full-protocol run 的工程 ID 为 `A_Dtrain_Etrain`，实际同时启用辅助置信掩码、MTC 与 SMR。论文主名称采用 `Ours (MTC+SMR)`，因为直接构成本文 memory-efficient 主贡献的是 MTC 与 SMR；`sections/04_experiments.tex` 已透明说明该配置还使用辅助置信掩码。

A/Confidence Mask 并非所有 clean-ablation 配置共享：`baseline_omnisam` 与 `Dtrain_Etrain` 均未启用。因此正式消融表保留其有无，以区分两组 MTC+SMR 配置，不能将六组都简写为同一方法。

## 2. 主结果表与效率表合并

`tables/main_results.tex` 现同时报告 mIoU、BFScore、训练耗时、Iter 时间和 22.4\% 降幅。各列只对事实上的最佳值加粗：OmniSAM 的 mIoU/BFScore，以及 Ours 的时间列。

`tables/efficiency_analysis.tex` 未删除、数字未修改，仅在顶部增加 deprecated 注释。`sections/04_experiments.tex` 不再 input 该文件，也不再引用 `tab:efficiency_analysis`；主结果与效率小节均引用 `tab:main_results`。

正文实际保留 4 张表：

1. 主结果与训练效率；
2. Memory 模块消融；
3. Token Budget；
4. PQR 辅助消融。

## 3. 新 caption 与 label

| 表格 | Caption 摘要 | Label |
|---|---|---|
| 主结果 | Cityscapes→DensePASS full protocol 下的分割性能与训练效率比较 | `tab:main_results` |
| Memory 消融 | 固定伪标签消融设置下的 memory-efficient 模块比较 | `tab:memory_ablation` |
| Token Budget | 不同 Memory Token 预算下的精度与效率权衡 | `tab:token_budget` |
| PQR | 固定伪标签设置下的辅助伪标签质量消融 | `tab:pseudo_label_ablation` |

全部正式 caption 已改为中文；只保留必要的英文模块/指标术语。

## 4. 时间术语区分

- **Iter 时间**：full protocol 主结果中的端到端单迭代时间，表中为 5.27s 与 4.09s。
- **Step Time**：固定伪标签 profile 的单步总时间，包含稀疏选择、编码解码及其他训练开销；其 token-budget 曲线不严格单调。
- **Memory Attention Time**：固定伪标签 profile 中 memory attention 模块的累计时间，用于解释 Memory Tokens 与 attention 开销的直接关系。

三者未合并或互换。正文采用“单随机种子”，并将 fixed-PL clean ablation/profile 统一表述为“固定伪标签消融设置”。

## 5. 排版与修改边界

- `main.tex` 已加载 `booktabs` 与 `graphicx`，本任务未修改。
- 各正式表使用 `\small`、3pt/4pt `\tabcolsep`、1.08 `\arraystretch` 和 booktabs。
- 未使用 `\resizebox`，未使用颜色、显著性星号或不存在的标准差。
- Token Budget 表已将 Step Time 和 Memory Attention Time 拆为独立列；12.5\% 使用 dagger 注释说明其为 MTC+SMR 组合端点。
- 没有修改任何实验数字、CSV、日志、预测、图片、参考文献或实验代码。
- Figure 2、Figure 3、Figure 4 的文件、caption、label 和引用均保留。

## 6. Overleaf 宽度风险

未在远端编译。Memory 消融表的方法列较长，Token Budget 表为六列；已通过短列名和缩小列间距控制宽度，但仍建议在 Overleaf 编译后检查是否越界。若确有越界，应优先进一步缩短方法名或表头，最后才考虑 `\resizebox`。
