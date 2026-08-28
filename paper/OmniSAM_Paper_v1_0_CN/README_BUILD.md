# OmniSAM 中文论文 v1.0 LaTeX 构建说明

推荐在本目录执行：

```bash
latexmk -xelatex main.tex
```

或手动执行：

```bash
xelatex -interaction=nonstopmode -file-line-error main.tex
xelatex -interaction=nonstopmode -file-line-error main.tex
```

当前远端环境检查结果：`xelatex` 和 `latexmk` 未安装，因此本任务只能创建可编译母版，无法在当前环境实际生成 PDF。
