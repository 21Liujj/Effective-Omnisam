# OmniSAM 中文论文正文压缩审计（1-10c）

> 本轮仅压缩八个正文 section，并对 `tables/pseudo_label_ablation.tex` 做文字层面的最小清理。未运行实验或 LaTeX 编译，未修改实验数字、图片、参考文献、模型或实验代码。

## 1. 分章节压缩统计

| Section | 修改前总字符 | 修改后总字符 | 总字符减少 | 修改前中文字符 | 修改后中文字符 | 中文字符减少 |
|---|---:|---:|---:|---:|---:|---:|
| `sections/00_abstract.tex` | 920 | 655 | 28.8\% | 389 | 295 | 24.2\% |
| `sections/01_introduction.tex` | 3314 | 2011 | 39.3\% | 1915 | 1146 | 40.2\% |
| `sections/02_related_work.tex` | 3032 | 1529 | 49.6\% | 1661 | 949 | 42.9\% |
| `sections/03_method.tex` | 10726 | 3434 | 68.0\% | 3806 | 1126 | 70.4\% |
| `sections/04_experiments.tex` | 5107 | 1867 | 63.4\% | 2018 | 699 | 65.4\% |
| `sections/05_ablation.tex` | 7129 | 2307 | 67.6\% | 2276 | 657 | 71.1\% |
| `sections/06_discussion.tex` | 2937 | 972 | 66.9\% | 1450 | 588 | 59.4\% |
| `sections/07_conclusion.tex` | 1383 | 613 | 55.7\% | 485 | 290 | 40.2\% |

- 八章总字符：34548 → 13388，减少 **61.2\%**。
- 八章中文字符：14000 → 5750，减少 **58.9\%**。

说明：总字符统计包含 LaTeX 命令、英文术语、公式和空白。原稿含大量代码说明、重复口径和工程状态段落，因此在完整保留公式、图表环境和关键事实后，实际减少幅度高于预估的 25--30\%；继续填充会重新引入已删除的重复和施工语言。

## 2. 小节合并与主线收敛

- Related Work 从 5 个小节合并为 3 个：全景语义分割与域适应、SAM/SAM2 与 memory-based 建模、token-efficient 建模与伪标签可靠性。
- Method 从 7 个小节收敛为 4 个：问题定义与整体框架、MTC、SMR、训练目标与辅助 PQR；删除独立的基线流程和实现说明章节。
- Experiments 合并实验设置与数据协议，保留评价口径、实现、主结果、效率和定性结果；Figure 4 原位保留。
- Ablation 保留 Memory module、Token budget、PQR 三个小节，删除重复的长篇消融小结；Figure 2 与 Figure 3 原位保留。
- Discussion 从 5 个小节合并为 3 个：效率--精度权衡、局限性、后续工作。
- Conclusion 收敛为三段：问题与方法、定量结果、限制与未来方向。

## 3. 施工痕迹清理

正文和允许修改的 PQR 表中已清除：`v1.0`、`v1.1`、本地 checkpoint/审计表述、`placeholder`、`TODO`、`NOT_IMPLEMENTED_YET`、内部 Python 文件名、代码类名和 R6 未完成说明。代码类名统一改为论文模块名 MTC、SMR 与 PQR。

## 4. PQR 压缩

- PQR 不再占据独立贡献条目，只在方法的训练目标中保留类别均衡与原型引导的抽象公式。
- 消融正文压缩为一段：R0=60.02、R2=60.07（仅 +0.05）、R2R4=60.05（未进一步改善），结论限定为辅助探索。
- `tables/pseudo_label_ablation.tex` 保留 R0、R2、R2R4 三行、全部数值、行列结构和 label；删除未完成实验说明，并将 caption 改为保守的辅助可靠性消融表述。

## 5. 保留项与静态检查

- 保留所有修改前已有的 `\label{}`、`\ref{}`、`\cite{}`、`\input{}` 和图片引用；原正文没有 `\cite{}`，因此不存在引用丢失。
- Method 保留问题定义、MTC token 表示与复杂度、SMR top-k 与复杂度、MTC→SMR 串联、PQR 掩码与联合训练损失等核心公式。工程 fallback/cache/异常分支和模式枚举已删除。
- Figure 2、Figure 3、Figure 4 的文件路径、caption、label 和正文引用均保留。
- Full protocol、fixed-PL profile 与 PQR 的硬性数字均通过字符串级静态核对；没有新增无损加速、显著精度提升、多 seed/多数据集完成、推理加速或显存降低等未经验证结论。
- 所有修改文件的 LaTeX environment 与花括号数量基本配对；未安装或运行 LaTeX。

## 6. 仍建议后续处理

- 中英文术语、表格配置名与 caption 风格仍可在 1-10d 统一。
- Related Work 尚无正式 `\cite{}`；若投稿要求完整文献综述，应在后续引用核验任务中补充真实且匹配的文献，而非恢复泛泛枚举。
- Figure 1 pipeline 仍由用户后续绘制，本轮未处理。
