#!/bin/bash
# OmniSAM EI会议论文 v2.0 编译脚本

set -e

echo "=================================="
echo "OmniSAM EI Conference Paper v2.0"
echo "编译开始..."
echo "=================================="

# 清理旧的编译文件
echo "清理旧文件..."
rm -f main.aux main.bbl main.blg main.log main.out main.synctex.gz main.pdf

# 第一次编译
echo "第一次 XeLaTeX 编译..."
xelatex -interaction=nonstopmode main.tex

# BibTeX 编译
echo "BibTeX 编译..."
bibtex main || true

# 第二次编译
echo "第二次 XeLaTeX 编译..."
xelatex -interaction=nonstopmode main.tex

# 第三次编译
echo "第三次 XeLaTeX 编译..."
xelatex -interaction=nonstopmode main.tex

echo "=================================="
echo "编译完成!"
echo "输出文件: main.pdf"
echo "=================================="

# 检查PDF是否生成
if [ -f main.pdf ]; then
    echo "✓ PDF生成成功"
    ls -lh main.pdf
else
    echo "✗ PDF生成失败，请检查错误日志"
    exit 1
fi
