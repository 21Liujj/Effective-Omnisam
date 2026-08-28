# EfficientTAM-Style Memory Compression in OmniSAM

This document records the memory acceleration improvement integrated into OmniSAM based on EfficientTAM-style non-learnable spatial memory pooling.

## 1. Scope and Objective

- Keep OmniSAM backbone/decoder interfaces unchanged.
- Reduce memory-attention cross-attention cost by compressing historical spatial memory tokens (K/V side only).
- Preserve baseline behavior when compression is disabled.
- Keep implementation low-risk and easy to ablate:
  - no Perceiver
  - no learnable latent queries
  - no extra trainable compressor

## 2. Core Pipeline

Default path (`before_attention`):

1. Read dense historical memory tokens from memory bank.
2. Compress only spatial memory features and aligned positional embeddings.
3. Keep current-frame query tokens unchanged.
4. Keep non-spatial/special tokens semantics safe.
5. Feed compressed memory to memory attention.
6. Log token ratio and timing.

Optional path (`before_bank`):

- Compress memory before writing into bank for memory footprint experiments.
- Not recommended as default due to earlier information loss.

## 3. Implementation Files

- `model/modules/memory_compressor.py`
  - `SpatialMemoryCompressor` (non-learnable)
  - modes: `none`, `pool`, `spatial`, `hybrid`
  - synchronized compression for memory feature and position embedding
  - stats and safety checks

- `model/memory_bank.py`
  - compression location support: `before_attention` (default), `before_bank` (optional)
  - runtime stage routing (`train` / `pseudo_label`)
  - robust fallback on compression error
  - memory-attention timing instrumentation

- `run_adaptation.py` and `utils/build.py`
  - config surface and wiring
  - baseline-safe defaults

- `adaptation/strategies.py`, `adaptation/pl_generator.py`, `adaptation/trainer.py`, `adaptation/engine.py`
  - stage-aware runtime context
  - structured logging for compression and timing

## 4. Config Parameters

Primary config namespace uses `mem_compression_*` CLI fields in `run_adaptation.py`.

Key options:

- `mem_compression_enabled` (default `False`)
- `mem_compression_mode` (`none|pool|spatial|hybrid`)
- `mem_compression_location` (`before_attention|before_bank`)
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

Compatibility:

- Legacy flags (`use_memory_compression`, `memory_compress_*`) are still accepted and mapped.

Reference template:

- `config_mem_pool2_before_attention.yaml`

## 5. Compression Modes

- `none`
  - no-op, full baseline behavior

- `pool`
  - EfficientTAM-style average pooling (`avg_pool2d`)
  - recommended default for first stable experiments

- `spatial`
  - non-learnable spatial downsample
  - configurable by kernel/stride or target size

- `hybrid`
  - local pooled spatial tokens + small non-learnable global summary tokens
  - no learnable positional parameters in compressor

## 6. Safety Guards

- Skip compression when token count is small (`min_tokens`).
- Enforce shape consistency (`L == H*W`) using stored `mem_hw` metadata.
- Keep query tokens unchanged.
- Keep dtype/device consistent (AMP/fp16 safe, no CPU copy).
- On compression exceptions, fall back to no-op path with explicit skip reason.
- Baseline default remains unchanged (`enabled=False`).

## 7. Logging and Metrics

### Online metrics (training/pseudo-label loops)

- `memory_tokens_before`
- `memory_tokens_after`
- `memory_token_ratio`
- `memory_attention_time`
- `memory_attention_time_ms`
- `memory_attention_time_avg`
- `memory_compress_mode`
- `memory_compress_location`

### JSONL summaries

- `metrics/pl_timing.jsonl`
  - pseudo-label total time
  - memory token stats and attention timing in pseudo-label stage
  - peak GPU memory during pseudo-label stage

- `metrics/train_timing.jsonl`
  - per-epoch train timing
  - memory token stats and attention timing in train stage
  - peak GPU memory during train stage

- `checkpoints/.../metrics.jsonl`
  - merged validation + timing + compression summary

Timing notes:

- CUDA timing is synchronized around memory attention to reduce async measurement error.

## 8. Recommended Ablations

Run at least:

1. baseline
   - `--no-mem_compression_enabled`

2. pool-2-before-attention
   - `--mem_compression_enabled`
   - `--mem_compression_mode pool`
   - `--mem_compression_location before_attention`
   - `--mem_compression_train_kernel_size 2`
   - `--mem_compression_pl_kernel_size 2`

Optional:

3. pool-4-before-attention
4. hybrid-2-before-attention
5. pool-2-before-bank

## 9. Expected Paper/Meeting Positioning

OmniSAM introduces EfficientTAM-style spatial memory token pooling into panoramic patch-sequence modeling: after reading historical patch memory from the memory bank and before memory attention, spatial memory features and their positional embeddings are compressed by non-learnable pooling to reduce K/V token count in cross-attention. This reduces memory-attention overhead in pseudo-label generation and UDA training without introducing extra trainable compressors, differing from EdgeTAM-style Perceiver compression.

