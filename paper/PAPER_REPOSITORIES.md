# 论文仓库说明

## 1. OmniSAM_Paper_v1_0_CN (第一版初稿 - 保留不动)
- **用途**: 原始初稿，作为参考
- **状态**: 只读，不再修改
- **内容**: Memory优化初步分析

## 2. EI_Conference_Paper_v2_CN (EI会议小论文 - 中文版)
- **用途**: EI会议投稿，中文版
- **页数**: 6-8页
- **核心**: Memory优化(训练) + SCM优化(推理)
- **状态**: 正在撰写

## 3. EI_Conference_Paper_v2_EN (EI会议小论文 - 英文版)
- **用途**: EI会议投稿，英文版
- **页数**: 6-8页
- **核心**: Memory优化(训练) + SCM优化(推理)
- **状态**: 正在撰写

## 4. Master_Thesis_v1 (硕士大论文)
- **用途**: 硕士学位论文
- **页数**: 50-80页
- **核心**: 完整研究体系，包含所有实验
- **状态**: 正在撰写

## 创建时间
2026-09-03

## 论文定位

### EI小论文（中英双版）
- **标题**: Efficient Panoramic Semantic Segmentation with Memory Optimization and Boundary-Aware Inference
- **中文标题**: 基于记忆优化和边界感知推理的高效全景语义分割
- **核心贡献**:
  1. Memory Token Compression (8× 压缩)
  2. Sparse Memory Retrieval (稀疏检索)
  3. Seamless Circular Memory (边界优化, +5.04% traffic signs)
- **结果**: 训练时间-55.8%, 推理精度+0.33%, 5-seed统计验证

### 硕士大论文
- **标题**: 面向全景图像的高效语义分割关键技术研究
- **核心内容**:
  1. 完整的理论分析
  2. 所有实验结果（包括SG-DRF 74.59%）
  3. 深度消融研究
  4. 跨数据集分析
  5. 详细的相关工作综述
