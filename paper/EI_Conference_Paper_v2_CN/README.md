# OmniSAM EI会议论文 v2.0 编译说明

## 论文信息

- **标题**: 基于记忆优化和边界感知推理的高效全景语义分割
- **类型**: EI会议论文(中文LaTeX)
- **版本**: v2.0 (基于2026年8月实验数据)
- **页数**: 预计6-8页

## 文件结构

```
EI_Conference_Paper_v2_CN/
├── main.tex                    # 主文档
├── sections/
│   ├── 00_abstract.tex        # 摘要
│   ├── 01_introduction.tex    # 引言
│   ├── 02_related_work.tex    # 相关工作
│   ├── 03_method.tex          # 方法
│   ├── 04_experiments.tex     # 实验
│   └── 05_conclusion.tex      # 结论
├── tables/                     # 表格(LaTeX代码)
├── figures/                    # 图片
├── refs.bib                    # 参考文献
└── README.md                   # 本文件
```

## 核心贡献

### 1. 记忆Token压缩(MTC)
- 8× Token压缩率(32768→4096)
- 仅损失0.24% mIoU

### 2. 稀疏记忆检索(SMR)
- 55.8%注意力时间降低
- 1.2GB GPU显存节省

### 3. 无缝循环记忆(SCM)
- +0.33%整体mIoU提升
- +5.04%交通标志类别提升(26.07→31.11)

## 关键实验数据

### 多随机种子验证(5 seeds: 1,2,3,42,100)
- **基线**: 60.22±0.02% mIoU
- **MTC+SMR**: 60.00±0.05% mIoU (4 seeds)
- **统计显著性**: 标准差<0.05%

### SCM循环填充消融
| Pad Width | mIoU | Δ |
|-----------|------|---|
| 256 | 60.60 | +0.09 |
| 512 | 60.72 | +0.21 |
| 768 | 60.80 | +0.29 |
| **1024** | **60.84** | **+0.33** |

## 编译方法

### 方法1: XeLaTeX(推荐)
```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

### 方法2: latexmk自动化
```bash
latexmk -xelatex -synctex=1 -interaction=nonstopmode main.tex
```

### 方法3: Overleaf
1. 将整个目录打包上传到Overleaf
2. 设置编译器为XeLaTeX
3. 点击"Recompile"

## 系统要求

- **TeX发行版**: TeX Live 2020或更高版本
- **中文支持**: ctex宏包(XeLaTeX编译)
- **字体**: 系统需安装中文字体(宋体、黑体等)

## 注意事项

1. **必须使用XeLaTeX编译**，pdflatex不支持中文
2. 首次编译需运行2-3次以生成正确的交叉引用
3. 修改参考文献后需重新运行bibtex

## 待补充内容

### 高优先级
- [ ] 补充作者信息和单位
- [ ] 补充参考文献(refs.bib)
- [ ] 添加可视化对比图(边界区域)
- [ ] 生成训练曲线图

### 中优先级
- [ ] 补充消融实验的详细数据
- [ ] 添加更多定性结果
- [ ] 补充补充材料(如有)

### 低优先级
- [ ] 优化表格格式
- [ ] 调整图表布局
- [ ] 检查引用格式

## 数据来源

所有实验数据来自:
- **实验清单**: `/root/autodl-tmp/OmniSAM-main/experiments/tecma/notes/full_experiment_inventory_20260903.md`
- **表格数据**: `/root/autodl-tmp/OmniSAM-main/experiments/tecma/tables/`
- **SCM总结**: `/root/autodl-tmp/OmniSAM-main/experiments/tecma/results/SCM_EXPERIMENT_SUMMARY.md`

## 论文投稿检查清单

- [ ] 标题和作者信息完整
- [ ] 摘要字数符合要求(150-200字)
- [ ] 所有表格和图片编号正确
- [ ] 参考文献格式统一
- [ ] 全文无拼写错误
- [ ] 数学公式编号正确
- [ ] 页数符合会议要求(6-8页)
- [ ] PDF生成成功且可正常打开

## 联系方式

如有问题，请联系: [作者邮箱]

---

**生成日期**: 2026-09-03  
**最后更新**: 2026-09-03
