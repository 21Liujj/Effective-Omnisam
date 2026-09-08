# OmniSAM 中文论文 v1.1 参考文献与引用审计（子任务 1-10e）

## 1. 执行范围与引用系统

- 审计日期：2026-07-14
- `refs.bib` 修改前条目数：5（均为待核验占位条目）
- `refs.bib` 修改后条目数：20
- 引用系统：BibTeX
- 文献样式：`\bibliographystyle{plain}`
- 文献库：`\bibliography{refs}`
- 正文引用命令：普通 `\cite{}`
- `main.tex`：未修改；现有 BibTeX 配置完整且无需调整

## 2. 最终文献清单

| Key | Title | Venue | Year |
|---|---|---|---:|
| `zhong2025omnisam` | OmniSAM: Omnidirectional Segment Anything Model for UDA in Panoramic Semantic Segmentation | ICCV | 2025 |
| `ma2021densepass` | DensePASS: Dense Panoramic Semantic Segmentation via Unsupervised Domain Adaptation with Attention-Augmented Context Exchange | IEEE ITSC | 2021 |
| `cordts2016cityscapes` | The Cityscapes Dataset for Semantic Urban Scene Understanding | CVPR | 2016 |
| `yang2021omnirange` | Capturing Omni-Range Context for Omnidirectional Segmentation | CVPR | 2021 |
| `zhang2022bending` | Bending Reality: Distortion-Aware Transformers for Adapting to Panoramic Semantic Segmentation | CVPR | 2022 |
| `zheng2023dppass` | Both Style and Distortion Matter: Dual-Path Unsupervised Domain Adaptation for Panoramic Semantic Segmentation | CVPR | 2023 |
| `zheng2023datr` | Look at the Neighbor: Distortion-Aware Unsupervised Domain Adaptation for Panoramic Semantic Segmentation | ICCV | 2023 |
| `zhang2024goodsam` | GoodSAM: Bridging Domain and Capacity Gaps via Segment Anything Model for Distortion-Aware Panoramic Semantic Segmentation | CVPR | 2024 |
| `kirillov2023segment` | Segment Anything | ICCV | 2023 |
| `ravi2025sam2` | SAM 2: Segment Anything in Images and Videos | ICLR | 2025 |
| `tsai2018adaptsegnet` | Learning to Adapt Structured Output Space for Semantic Segmentation | CVPR | 2018 |
| `zou2018cbst` | Unsupervised Domain Adaptation for Semantic Segmentation via Class-Balanced Self-Training | ECCV | 2018 |
| `hoyer2022daformer` | DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation | CVPR | 2022 |
| `hoyer2022hrda` | HRDA: Context-Aware High-Resolution Domain-Adaptive Semantic Segmentation | ECCV | 2022 |
| `hoyer2023mic` | MIC: Masked Image Consistency for Context-Enhanced Domain Adaptation | CVPR | 2023 |
| `rao2021dynamicvit` | DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification | NeurIPS | 2021 |
| `bolya2023tome` | Token Merging: Your ViT But Faster | ICLR | 2023 |
| `lu2023tokensharing` | Content-Aware Token Sharing for Efficient Semantic Segmentation With Vision Transformers | CVPR | 2023 |
| `zheng2024sfuda360` | Semantics, Distortion, and Style Matter: Towards Source-Free UDA for Panoramic Segmentation | CVPR | 2024 |
| `zheng2025sfuda360pp` | 360SFUDA++: Towards Source-Free UDA for Panoramic Segmentation by Learning Reliable Category Prototypes | IEEE TPAMI, 47(2):1190--1204 | 2025 |

## 3. 正式版本与 arXiv 核验链接

以下条目在 `refs.bib` 中仅保留正式会议或期刊版本；链接只用于本次人工核验对应关系，没有作为重复 arXiv 条目写入。

| Key | 正式版本 | arXiv 核验链接 |
|---|---|---|
| `zhong2025omnisam` | ICCV 2025 | https://arxiv.org/abs/2503.07098 |
| `ma2021densepass` | IEEE ITSC 2021 | https://arxiv.org/abs/2108.06383 |
| `cordts2016cityscapes` | CVPR 2016 | https://arxiv.org/abs/1604.01685 |
| `yang2021omnirange` | CVPR 2021 | https://arxiv.org/abs/2103.05687 |
| `zhang2022bending` | CVPR 2022 | https://arxiv.org/abs/2203.01452 |
| `zheng2023dppass` | CVPR 2023 | https://arxiv.org/abs/2303.14360 |
| `zheng2023datr` | ICCV 2023 | https://arxiv.org/abs/2308.05493 |
| `zhang2024goodsam` | CVPR 2024 | https://arxiv.org/abs/2403.16370 |
| `kirillov2023segment` | ICCV 2023 | https://arxiv.org/abs/2304.02643 |
| `ravi2025sam2` | ICLR 2025 | https://arxiv.org/abs/2408.00714 |
| `tsai2018adaptsegnet` | CVPR 2018 | https://arxiv.org/abs/1802.10349 |
| `zou2018cbst` | ECCV 2018 | https://arxiv.org/abs/1810.07911 |
| `hoyer2022daformer` | CVPR 2022 | https://arxiv.org/abs/2111.14887 |
| `hoyer2022hrda` | ECCV 2022, pp. 372--391 | https://arxiv.org/abs/2204.13132 |
| `hoyer2023mic` | CVPR 2023 | https://arxiv.org/abs/2212.01322 |
| `rao2021dynamicvit` | NeurIPS 2021 | https://arxiv.org/abs/2106.02034 |
| `bolya2023tome` | ICLR 2023 | https://arxiv.org/abs/2210.09461 |
| `lu2023tokensharing` | CVPR 2023, pp. 23631--23640 | https://arxiv.org/abs/2306.02095 |
| `zheng2024sfuda360` | CVPR 2024 | https://arxiv.org/abs/2403.12505 |
| `zheng2025sfuda360pp` | IEEE TPAMI 2025, 47(2):1190--1204, DOI 10.1109/TPAMI.2024.3490619 | https://arxiv.org/abs/2404.16501 |

清单未提供且本任务未进一步确认的会议页码没有猜测或写入。作者特殊字符采用 BibTeX 转义：`Rei{\ss}`、`R{\"a}dle`、`Doll{\'a}r`。

## 4. 正文引用变更

- Introduction：新增 7 组 citation。
- Related Work：新增 11 组 citation。
- Method：新增 4 组 citation。
- Experiments：新增 4 组 citation。
- Abstract：未加入引用。
- 引用均使用普通 `\cite{}`，没有上标、脚注或尾注引用。
- 单组引用最多 4 篇，没有使用 `\nocite{*}`。
- 仅在已有句子中补充引用；没有新增长背景段落。

## 5. 静态检查结果

- 最终 BibTeX 条目数为 20，key 均唯一。
- 20 个 BibTeX 条目均已在允许修改的正文中被合理引用。
- 所有 `\cite{}` key 均能在 `refs.bib` 中解析。
- 未使用 BibTeX 条目：无。
- 缺失 citation key：无。
- 空 `\cite{}`：无。
- `\cite{TODO}`：无。
- `\nocite{*}`：无。
- 正式版与 arXiv 重复条目：无。
- OmniSAM：ICCV 2025 `@inproceedings`。
- SAM 2：ICLR 2025 `@inproceedings`，key 为 `ravi2025sam2`。
- Source-Free Panoramic UDA：CVPR 2024 `@inproceedings`。
- 360SFUDA++：TPAMI 2025 `@article`，47(2):1190--1204。
- CBST：采用完整 ECCV 正式标题。
- BibTeX 与四个修改章节的花括号基本配对检查通过。
- `main.tex` 未修改。
- 未修改实验数字、表格、图片或公式；未执行训练、验证、推理或 LaTeX 编译。

## 6. Overleaf 编译重点检查

1. 按 XeLaTeX -> BibTeX -> XeLaTeX -> XeLaTeX 的顺序完整编译，确认交叉引用和文献编号稳定。
2. 检查 `plain` 样式下多条普通 `\cite{}` 的显示是否满足最终版式要求；标准 `plain` 通常显示离散编号，不保证自动压缩为 `[2--4]`。本任务按要求沿用现有样式，没有引入 natbib、biblatex 或 biber。
3. 检查中文正文与方括号编号之间的间距，以及超链接颜色是否符合投稿模板。
4. 检查 BibTeX 输出中的 `Reiß`、`Rädle`、`Dollár` 是否正确渲染。
5. 检查受花括号保护的专有名词大小写：OmniSAM、UDA、DensePASS、Cityscapes、SAM 2、GoodSAM、DAFormer、HRDA、MIC、DynamicViT、360SFUDA++。
6. 若后续切换到正式投稿模板，应只调整模板要求的 bibliography style，不应迁移到 biblatex/biber，也不应更改 citation key。
