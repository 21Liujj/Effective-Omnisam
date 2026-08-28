# OmniSAM 中的 EfficientTAM 风格 Memory Compression（中文记录）

本文档记录了本次在 OmniSAM 中落地的 memory 加速改造：基于 EfficientTAM 思路，在不引入学习式压缩器的前提下，对历史 spatial memory token 做非学习式池化压缩，以降低 memory attention 计算开销。

## 1. 改造目标与边界

- 不改 backbone 与 semantic decoder 主结构。
- 默认行为保持不变：压缩开关关闭时，结果与原工程一致。
- 主路径采用 `before_attention`：
  - memory bank 存原始 dense memory；
  - 读取历史 memory 后、进入 memory attention 前进行压缩。
- `before_bank` 仅保留为可选实验项（非默认）。
- 全程不引入：
  - 2D Spatial Perceiver
  - learnable latent queries
  - 额外训练式 compressor/蒸馏目标

## 2. 核心流程

默认流程（`before_attention`）：

1. 从 memory bank 读取历史 memory tokens。
2. 仅对 spatial memory features 与对应 pos enc 做池化压缩。
3. 保持当前帧 query 不变。
4. 保持非空间 token（如特殊 token）语义安全。
5. 压缩后 memory 送入 memory attention。
6. 记录 token 压缩比例与耗时变化。

## 3. 代码落点

- `model/modules/memory_compressor.py`
  - `SpatialMemoryCompressor`（纯非学习式）
  - 支持 `none / pool / spatial / hybrid`
  - 对 feature 与 pos enc 同步压缩
  - 提供完整 stats 与保护逻辑

- `model/memory_bank.py`
  - 支持 `before_attention`（默认）与 `before_bank`（可选）
  - 训练/伪标签阶段的运行上下文切换
  - 压缩异常自动回退 no-op
  - memory attention 前后计时（CUDA 同步）

- `run_adaptation.py`、`utils/build.py`
  - 新增并打通 `mem_compression_*` 配置
  - 保持旧参数兼容映射

- `adaptation/strategies.py`、`adaptation/pl_generator.py`、`adaptation/trainer.py`、`adaptation/engine.py`
  - train/pl 两阶段独立配置生效
  - 指标与耗时汇总到现有 jsonl

## 4. 配置项（核心）

在 `run_adaptation.py` 中新增（CLI 直接可用）：

- `mem_compression_enabled`
- `mem_compression_mode`：`none|pool|spatial|hybrid`
- `mem_compression_location`：`before_attention|before_bank`
- `mem_compression_kernel_size`
- `mem_compression_stride`
- `mem_compression_keep_global_tokens`
- `mem_compression_apply_to_pos_enc`
- `mem_compression_min_tokens`
- `mem_compression_warmup_iters`
- `mem_compression_enable_in_train`
- `mem_compression_enable_in_pseudo_label`
- `mem_compression_train_kernel_size`
- `mem_compression_pl_kernel_size`
- `mem_compression_verbose`
- `mem_compression_log_interval`

模板配置见：

- `config_mem_pool2_before_attention.yaml`

## 5. 模式说明

- `none`
  - 不压缩，完全 baseline

- `pool`
  - EfficientTAM 风格标准 avg pooling（推荐首选）

- `spatial`
  - 非学习式空间下采样（stride 或 target size）

- `hybrid`
  - local pooling + 少量 non-learnable global summary tokens
  - 不含学习参数

## 6. 稳定性与精度保护

- token 过少（`min_tokens`）时跳过压缩。
- 使用 `mem_hw` 元数据校验形状一致性（`L == H*W`）。
- 不压 current query。
- feature 与 pos 使用同规则同步压缩，避免错位。
- AMP/fp16 与 dtype/device 保持一致，无 CPU 回拷。
- 压缩失败时回退 no-op，并记录 skip reason。

## 7. 日志与计时指标

在线指标（train/pl）包含：

- `memory_tokens_before`
- `memory_tokens_after`
- `memory_token_ratio`
- `memory_attention_time`
- `memory_attention_time_ms`
- `memory_attention_time_avg`
- `memory_compress_mode`
- `memory_compress_location`

结构化输出：

- `metrics/pl_timing.jsonl`
  - 伪标签总耗时、memory token 统计、memory attention 时间、峰值显存

- `metrics/train_timing.jsonl`
  - 训练耗时、memory token 统计、memory attention 时间、峰值显存

- `checkpoints/.../metrics.jsonl`
  - 验证指标 + 时间 + 压缩统计的统一汇总

## 8. 推荐实验顺序

优先确保以下两组稳定：

1. baseline  
   - `--no-mem_compression_enabled`

2. pool-2-before-attention  
   - `--mem_compression_enabled`
   - `--mem_compression_mode pool`
   - `--mem_compression_location before_attention`
   - `--mem_compression_train_kernel_size 2`
   - `--mem_compression_pl_kernel_size 2`

后续可加：

3. pool-4-before-attention  
4. hybrid-2-before-attention  
5. pool-2-before-bank（补充实验）

## 9. 组会/论文可用表述

本文将 EfficientTAM-style spatial memory token pooling 引入 OmniSAM 的全景 patch sequence 建模流程：在 memory bank 读取历史 patch memory 后、进入 memory attention 前，对 spatial memory features 及其 positional embeddings 进行非学习式池化压缩，从而减少 cross-attention 中 K/V token 数量，降低伪标签生成与 UDA 训练阶段的 memory attention 计算开销。该方法不引入额外学习式压缩器，区别于 EdgeTAM 的 Perceiver-style memory compression，具备实现简单、训练风险低、易于消融验证的特点。

