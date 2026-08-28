# Efficient-SAM2-Style Sparse Memory Retrieval (SMR) in OmniSAM

This document records the memory acceleration improvement integrated into OmniSAM based on Efficient-SAM2-style sparse memory retrieval (SMR): selecting only salient historical **spatial memory tokens** as K/V for memory attention.

## 1. Scope and Objective

- Keep OmniSAM backbone/decoder interfaces unchanged.
- Reduce memory-attention cross-attention cost by **retrieving a top-k subset** of historical spatial memory tokens (K/V side only).
- Preserve baseline behavior when SMR is disabled (`enabled=False`).
- Keep implementation low-risk and easy to ablate:
  - no Perceiver
  - no learnable latent queries
  - no extra trainable compressor
- Keep special semantics safe:
  - do **not** compress current-frame query tokens
  - do **not** touch object pointer / special tokens semantics (SMR operates on historical spatial tokens only in this repo)

## 2. Core Pipeline

Default path (`before_attention`):

1. Read dense historical spatial memory tokens from memory bank (`maskmem_features`) and aligned positional embeddings (`maskmem_pos_enc`).
2. Compute a saliency score (stable default: `feature_norm`).
3. Select top-k indices under keep ratio constraints (`keep_ratio`, `min_tokens`, `max_tokens`).
4. Gather both `memory_tokens` and `pos_tokens` using the **same** indices (pos-enc alignment safety).
5. Feed retrieved sparse memory to SAM2 memory attention.
6. Log token keep ratio, cache hit ratio, and timing.

Combination with EfficientTAM-style pooling (optional):

- Order is configurable:
  - `pool_then_smr`: pool first, then SMR on the pooled tokens (default)
  - `smr_then_pool`: SMR first, then pool on the retrieved tokens

## 3. Implementation Files

- `model/modules/sparse_memory_retriever.py`
  - `SparseMemoryRetriever` (non-learnable selection)
  - modes: `none`, `feature_norm`, `attn_topk` (fallback placeholder), `confidence_topk`, `hybrid`, `random`
  - safe fallback guards (min tokens, NaN, empty selection)
  - first-recollection cache (stores selected indices on CPU, detached)

- `model/memory_bank.py`
  - SMR insertion point: `before_attention`
  - stage-aware routing (`train` / `pseudo_label`) via `set_runtime_context(...)`
  - robust fallback on selection error
  - reuses existing memory-attention timing instrumentation

- `run_adaptation.py` and `utils/build.py`
  - config surface and wiring
  - baseline-safe defaults (`enabled=False`)

- `adaptation/strategies.py`, `adaptation/pl_generator.py`, `adaptation/engine.py`
  - stage-aware runtime context already exists
  - structured logging for SMR ratios/timing and cache hit ratio

## 4. Config Parameters

Primary config namespace uses `sparse_memory_retrieval_*` CLI fields in `run_adaptation.py`.

Key options:

- `sparse_memory_retrieval_enabled` (default `False`)
- `sparse_memory_retrieval_location` (currently only `before_attention`)
- `sparse_memory_retrieval_mode` (`none|feature_norm|attn_topk|confidence_topk|hybrid|random`)
- `sparse_memory_retrieval_fallback_mode` (`feature_norm|random`)
- `sparse_memory_retrieval_min_tokens`
- `sparse_memory_retrieval_max_tokens` (0 disables cap)
- `sparse_memory_retrieval_cache_enabled`
- `sparse_memory_retrieval_cache_update_interval`
- `sparse_memory_retrieval_apply_in_train`
- `sparse_memory_retrieval_apply_in_pseudo_label`
- `sparse_memory_retrieval_train_keep_ratio`
- `sparse_memory_retrieval_pl_keep_ratio`
- `sparse_memory_retrieval_verbose`
- `sparse_memory_retrieval_log_interval`
- `sparse_memory_retrieval_order_with_pooling` (`pool_then_smr|smr_then_pool`)

Reserved (no-op / future hooks in this version):

- `sparse_memory_retrieval_return_attn` (attention-guided `attn_topk` needs upstream attn score exposure)
- `sparse_memory_retrieval_preserve_overlap_tokens`
- `sparse_memory_retrieval_preserve_boundary_tokens`
- `sparse_memory_retrieval_preserve_confident_tokens`

## 5. Retrieval Modes (Current Status)

- `none`
  - no-op, full baseline behavior

- `feature_norm` (stable default)
  - score = mean over batch of L2 norm per token
  - does not require attention weights or extra heads

- `attn_topk` (placeholder + safe fallback)
  - intended: use cross-attention response to identify salient memory tokens
  - current repo status: SAM2 attention uses `scaled_dot_product_attention` without returning weights, so when `attn_scores` is not provided, SMR **automatically falls back** to `fallback_mode` (default `feature_norm`)

- `confidence_topk` (optional)
  - uses confidence map when provided; otherwise falls back

- `hybrid` (lightweight)
  - combines feature norm with optional priors when provided (attn/conf/overlap/boundary); otherwise degrades to feature norm

- `random`
  - negative control / sanity check

## 6. First Recollection Cache

Goal:

- Avoid re-computing selection for the same memory entry across repeated retrieval.

Current implementation:

- Cache key: `memory_id` string generated in `MemoryBank` (e.g. `source:<frame_idx>` / `target:<frame_idx>`).
- Cache payload: selected indices stored as `cpu()` + `detach()` tensor (prevents GPU memory leak and gradient tracking).
- Cache cleanup:
  - on memory reset (`_init_output_dict_source/_target`), corresponding cached entries are removed.

Notes:

- Cache update is throttled by `sparse_memory_retrieval_cache_update_interval`.

## 7. Safety Guards

- Skip SMR when token count is small (`min_tokens`).
- Enforce at least `min_tokens` and at least 1 token.
- Gather `memory_tokens` and `pos_tokens` with identical indices.
- On exceptions / NaN / empty selection, fall back to full tokens.
- Baseline default remains unchanged (`enabled=False`).

## 8. Logging and Metrics

### Online metrics (training)

Reported through `adaptation/strategies.py` metrics:

- `smr_enabled`
- `smr_mode`
- `smr_tokens_before`
- `smr_tokens_after`
- `smr_keep_ratio`
- `smr_selection_time`
- `smr_cache_hit`
- `smr_cache_hit_ratio`
- plus existing memory attention timing:
  - `memory_attention_time`, `memory_attention_time_ms`, `memory_attention_time_avg`

### JSONL summaries

- `metrics/pl_timing.jsonl`
  - adds: `smr_tokens_before/after`, `smr_keep_ratio`, `smr_cache_hit_ratio`, `smr_selection_time`

- `metrics/train_timing.jsonl`
  - adds: `smr_enabled`, `smr_mode`, `smr_cache_hit_ratio` (epoch-level summary)

- `checkpoints/.../metrics.jsonl`
  - adds SMR fields for automation scripts

Timing notes:

- Memory attention timing is synchronized around the attention call in `MemoryBank`.
- SMR selection time is measured inside `SparseMemoryRetriever.retrieve(...)`.

## 9. Recommended Ablations

Run at least:

1. baseline
   - `--no-mem_compression_enabled`
   - `--no-sparse_memory_retrieval_enabled`

2. SMR conservative (stable)
   - `--sparse_memory_retrieval_enabled`
   - `--sparse_memory_retrieval_mode feature_norm`
   - `--sparse_memory_retrieval_train_keep_ratio 0.7`
   - `--sparse_memory_retrieval_pl_keep_ratio 0.5`

Optional:

3. SMR aggressive for pseudo-labels
   - `--sparse_memory_retrieval_train_keep_ratio 0.7`
   - `--sparse_memory_retrieval_pl_keep_ratio 0.4` (or 0.3)

4. SMR + pooling
   - `--mem_compression_enabled`
   - `--sparse_memory_retrieval_enabled`
   - `--sparse_memory_retrieval_order_with_pooling pool_then_smr`

5. Random retrieval sanity check
   - `--sparse_memory_retrieval_mode random`

## 10. Expected Paper/Meeting Positioning

On top of EfficientTAM-style non-learnable memory pooling, OmniSAM further adopts Efficient-SAM2-style sparse memory retrieval to exploit token-level redundancy in memory attention: after reading historical patch spatial memories, we estimate token importance (e.g., feature norm, optional attention/confidence/overlap priors) and keep only top-k salient tokens as K/V for memory attention. Unlike EdgeTAM-style learnable Perceiver compression, this method introduces no additional trainable compressor and remains baseline-safe when disabled.

