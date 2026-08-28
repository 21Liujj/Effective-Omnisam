# Phase 2 / Task 2-1 — Token Geometry Metadata Schema

Date: 2026-06-29
Run Level: L1 design / dry-run
Scope: memory token geometry diagnostics only
Output: `experiments/geo_memory/notes/step_02_1_token_metadata_schema.md`

## 0. Task Boundary

This task only designs the metadata schema for future Geo-TeCMA / Geo-Memory diagnostics.

- No training is run.
- No pseudo-label generation is run.
- No model architecture is changed.
- No memory token selection, sparse retrieval, memory attention, or loss behavior is changed.
- Phase 1 / Task 1-4 available-subset region metrics are used only as preliminary motivation, not as official full-val conclusions.
- Phase 2 must not depend on GM0/GM1 full prediction completeness; it focuses on memory token geometry diagnostics, region distribution, and score / kept-dropped statistics.

## 1. Design Goals

For each memory token, record:

1. where the token came from in batch / image / patch / frame space;
2. how the token maps to the 2D memory grid via `mem_hw`;
3. whether the token has reliable ERP geometry;
4. which coarse region the token belongs to: `seam`, `polar`, `seam_polar`, `equator`, or `unknown`;
5. what score was used before any future geometry reweighting;
6. what geometry score would be assigned in diagnosis;
7. whether the token would be kept or dropped by existing selection / compression stages;
8. which stage produced or consumed the token.

This schema is intentionally diagnostic-first. Fields that are unavailable in current code must be written as `null` or `"N/A"` with a clear `reason_unavailable`; they must not be fabricated from tensor order when geometry is unreliable.

## 2. Record Granularity

Two JSONL granularities are recommended.

### 2.1 Token Record

One row per memory token after flattening a memory feature grid from `(H_mem, W_mem)` to `(L, B, C)`.

Suggested file for future Task 2-2 / 2-3:

```text
experiments/geo_memory/results/token_geometry_diag.jsonl
```

### 2.2 Frame Summary Record

One row per frame / memory write or read event. This avoids huge logs during early diagnostics and can summarize token records by region and selection state.

Suggested file for future Task 2-2 / 2-3:

```text
experiments/geo_memory/results/token_geometry_summary.jsonl
```

Task 2-1 only defines these schemas; it does not implement the writers.

## 3. Token Metadata Schema

| Field | Type | Required | Source / Derivation | Notes |
|---|---:|---:|---|---|
| `schema_version` | string | yes | constant | Start with `"geo_memory_token_v1"`. |
| `run_level` | string | yes | constant/config | `"L1_schema"`, `"L2_dry_run"`, etc. |
| `stage` | string | yes | runtime context | `train`, `pseudo_label`, `val`, or `smoke`. |
| `global_step` | int/null | yes | trainer context | `null` if unavailable. |
| `epoch` | int/null | optional | trainer context | `null` if unavailable. |
| `memory_event` | string | yes | memory bank hook | `write_new_memory`, `read_before_attention`, `before_bank_compress`, `before_attention_compress`, `sparse_retrieve`, or `selection_summary`. |
| `mem_type` | string | yes | model call | `source` or `target`. |
| `batch_id` | int | yes | batch dimension | Index inside the current batch. |
| `image_id` | string/null | optional | dataset metadata | Use real ID if available; otherwise `null`. |
| `patch_id` | int/null | optional | dataset/view metadata | Prefer true dataset patch ID if available. |
| `view_id` | int/null | optional | dataset/view metadata | Alias or richer form of patch/view index if available. |
| `tensor_order` | int | yes | strategy geo context | Tensor position in current frame sequence. |
| `frame_idx` | int | yes | model forward argument | The frame index passed to `model(frame_idx, ...)`. |
| `frame_cnt` | int | yes | strategy context | Number of patches/views in current sequence. |
| `random_reverse_possible` | bool | yes | dataset/config context | True for WildPASS-style loaders where order can be randomly reversed. |
| `tensor_order_inference_allowed` | bool | yes | `plugins.geo_memory.allow_tensor_order_inference` | Must default to false. |
| `true_erp_x_range` | list[int,int]/null | yes | reliable metadata only | Must be `null` when tensor-order inference is unsafe. |
| `true_erp_y_range` | list[int,int]/null | yes | geometry context | Usually `[0, erp_height]` for full-height ERP patches. |
| `true_erp_x_range_reliable` | bool | yes | geometry context | Controls whether seam labels may be trusted. |
| `inferred_erp_x_range_from_tensor_order` | list[int,int]/null | diagnostic | tensor-order formula | Diagnostic only; not a trusted seam source unless allowed and safe. |
| `erp_width` | int | yes | config | Current default `4096`. |
| `erp_height` | int | yes | config | Current default `1024`. |
| `window_size` | int | yes | config | Current default `1024`. |
| `stride` | int | yes | config | Current default `384`. |
| `mem_hw` | list[int,int] | yes | memory bank | `[mem_h, mem_w]`; e.g. `[64,64]`. |
| `memory_num_tokens` | int | yes | memory bank | Expected `mem_h * mem_w` before stage-specific compression. |
| `token_id` | int | yes | flattened token index | Row-major index in current token set. |
| `token_row` | int | yes | `token_id // mem_w` | Must satisfy `0 <= token_row < mem_h`. |
| `token_col` | int | yes | `token_id % mem_w` | Must satisfy `0 <= token_col < mem_w`. |
| `token_x` | float/null | optional | token grid to patch/ERP mapping | Center x in patch or ERP coords if reliable. |
| `token_y` | float/null | optional | token grid to patch/ERP mapping | Center y in patch/ERP coords; latitude diagnostics use token row. |
| `source_box` | list[float,float,float,float]/null | optional | token cell footprint | `(x0,y0,x1,y1)` in patch or ERP coordinate space if available. |
| `latitude_rad` | float/null | diagnostic | token row / ERP height | Optional; can use `compute_latitude_map` aligned to token grid. |
| `latitude_prior` | float/null | diagnostic | `abs(sin(latitude))` | High near poles, low near equator. |
| `latitude_band` | string | yes | token row | `top`, `middle`, `bottom`, or `unknown`. |
| `seam_region` | string | yes | reliable x-range only | `left_seam`, `right_seam`, `non_seam`, or `unknown`. |
| `assigned_region` | string | yes | region priority | One of `seam`, `polar`, `seam_polar`, `equator`, `unknown`. |
| `region_priority` | string | yes | schema constant | `seam_polar > seam > polar > equator`; `unknown` only for unmappable/unsafe geometry. |
| `base_score` | float/null | optional | existing selection score | Feature norm / attention / importance score if available. |
| `base_score_source` | string | yes | score source | `feature_norm`, `attention`, `to_cat_importance`, `none`, or `unknown`. |
| `geo_score` | float/null | optional | future diagnosis-only geo score | Must not affect selection in Task 2. |
| `geo_score_source` | string | yes | geo scoring rule | e.g. `latitude_prior`, `seam_prior`, `none`. |
| `final_score` | float/null | optional | future combined score | Diagnostic only until explicit selection task. |
| `kept_or_dropped` | string | yes | stage result | `kept`, `dropped`, or `not_applicable`. |
| `drop_reason` | string/null | optional | stage result | `spatial_compression`, `sparse_retrieval`, `memory_select_topk`, `none`, or error reason. |
| `selection_rank` | int/null | optional | top-k stage | Rank before keep/drop if available. |
| `keep_ratio` | float/null | optional | stage config/stats | Sparse/compression keep ratio when available. |
| `compression_stage` | string | yes | memory/compressor hook | `none`, `before_bank`, `before_attention`, `sparse_retrieval`, `memory_select`. |
| `compression_mode` | string | yes | memory bank / compressor | Current examples: `none`, pooling mode from `SpatialMemoryCompressor`. |
| `original_hw` | list[int,int]/null | optional | compression stats | From `last_compression_stats.original_hw`. |
| `compressed_hw` | list[int,int]/null | optional | compression stats | From `last_compression_stats.compressed_hw`. |
| `original_num_tokens` | int/null | optional | compression stats | Before compression. |
| `compressed_num_tokens` | int/null | optional | compression stats | After compression. |
| `compression_ratio` | float/null | optional | compression stats | `compressed_num_tokens / original_num_tokens`. |
| `sparse_tokens_before` | int/null | optional | sparse stats | From `last_sparse_tokens_before`. |
| `sparse_tokens_after` | int/null | optional | sparse stats | From `last_sparse_tokens_after`. |
| `sparse_keep_ratio` | float/null | optional | sparse stats | From `last_sparse_keep_ratio`. |
| `reason_unavailable` | dict | yes | writer | Map unavailable optional fields to reasons. |
| `selection_effect` | string | yes | constant for Phase 2 | Must be `"diagnosis_only"`. |

## 4. Region Assignment Rules

### 4.1 Token Row and Column Mapping

Memory features are flattened from `(H_mem, W_mem)` to `(L, B, C)`. For each token:

```text
token_row = token_id // mem_w
token_col = token_id % mem_w
```

Validation checks:

```text
0 <= token_id < mem_h * mem_w
0 <= token_row < mem_h
0 <= token_col < mem_w
memory_num_tokens == mem_h * mem_w
```

If any check fails, `assigned_region="unknown"` and the failure is recorded in `reason_unavailable`.

### 4.2 Latitude / Polar / Equator

Latitude-aware labels must use token-row mapping, not patch-level labels, because DensePASS/WildPASS ERP patches cover full image height.

Recommended diagnostic bands:

```text
polar_band = round(mem_h * polar_band_ratio)
top:    token_row < polar_band
bottom: token_row >= mem_h - polar_band
middle: otherwise
```

`latitude_band` is then:

```text
top / middle / bottom
```

Polar membership:

```text
is_polar = latitude_band in {top, bottom}
```

### 4.3 Seam

Seam membership must be trusted only when horizontal ERP coordinates are reliable.

```text
if true_erp_x_range_reliable is false:
    seam_region = unknown
    is_seam = false for official region assignment unless a later task provides reliable source_box/ERP coords
else:
    use true_erp_x_range / token_x / source_box to decide left_seam, right_seam, or non_seam
```

For WildPASS-like random reverse order, tensor order alone is not reliable; seam stays `unknown` unless true coordinates are explicitly available.

### 4.4 Priority

When valid seam and polar labels overlap:

```text
seam_polar > seam > polar > equator
```

`unknown` is reserved for unmappable geometry, invalid token grid, unreliable seam-only requests, or missing required context.

## 5. Existing Field Sources in Current Code

| Current location | Useful fields for schema | Notes |
|---|---|---|
| `adaptation/strategies.py::_build_geo_context` | `tensor_order`, `frame_idx`, `frame_cnt`, `true_erp_x_range`, `true_erp_y_range`, `true_erp_x_range_reliable`, `random_reverse_possible`, `seam_region` | Existing diagnosis-only context. |
| `adaptation/strategies.py::_geo_diag_record` | `mem_hw`, `memory_num_tokens`, token mapping strings, reliability flags | Frame-level record already exists. |
| `model/memory_bank.py::_encode_new_memory` | `mem_hw`, `num_tokens`, `geometry_meta` | Natural hook for write-side token metadata. |
| `model/memory_bank.py::forward` | memory chunks, effective memory length, selection mode, before-attention compression, sparse retrieval | Natural hook for read-side kept/dropped summaries. |
| `model/memory_bank.py::last_compression_stats` | `original_num_tokens`, `compressed_num_tokens`, `compression_ratio`, `original_hw`, `compressed_hw`, `compression_mode`, `compression_location` | Frame/chunk-level compression summary. |
| `model/memory_bank.py::last_sparse_stats` | sparse retrieval stats | Token-level indices may require extra diagnostic capture in Task 2-2. |
| `model/modules/memory_compressor.py` | compression statistics | Current pooling compression may not expose per-token drop IDs; record summaries first. |
| `model/modules/sparse_memory_retriever.py` | selected indices / score mode if returned | Potential source for `base_score`, `kept_or_dropped`, `selection_rank`. |

## 6. Example Token Record

```json
{
  "schema_version": "geo_memory_token_v1",
  "run_level": "L2_dry_run",
  "stage": "train",
  "global_step": 12,
  "epoch": 0,
  "memory_event": "write_new_memory",
  "mem_type": "target",
  "batch_id": 0,
  "image_id": null,
  "patch_id": null,
  "view_id": null,
  "tensor_order": 3,
  "frame_idx": 3,
  "frame_cnt": 9,
  "random_reverse_possible": false,
  "tensor_order_inference_allowed": false,
  "true_erp_x_range": null,
  "true_erp_y_range": [0, 1024],
  "true_erp_x_range_reliable": false,
  "inferred_erp_x_range_from_tensor_order": [1152, 2176],
  "erp_width": 4096,
  "erp_height": 1024,
  "window_size": 1024,
  "stride": 384,
  "mem_hw": [64, 64],
  "memory_num_tokens": 4096,
  "token_id": 130,
  "token_row": 2,
  "token_col": 2,
  "token_x": null,
  "token_y": 40.0,
  "source_box": null,
  "latitude_rad": 1.448,
  "latitude_prior": 0.992,
  "latitude_band": "top",
  "seam_region": "unknown",
  "assigned_region": "polar",
  "region_priority": "seam_polar > seam > polar > equator",
  "base_score": null,
  "base_score_source": "none",
  "geo_score": 0.992,
  "geo_score_source": "latitude_prior",
  "final_score": null,
  "kept_or_dropped": "not_applicable",
  "drop_reason": null,
  "selection_rank": null,
  "keep_ratio": null,
  "compression_stage": "none",
  "compression_mode": "none",
  "original_hw": [64, 64],
  "compressed_hw": [64, 64],
  "original_num_tokens": 4096,
  "compressed_num_tokens": 4096,
  "compression_ratio": 1.0,
  "sparse_tokens_before": null,
  "sparse_tokens_after": null,
  "sparse_keep_ratio": null,
  "reason_unavailable": {
    "image_id": "dataset metadata not provided to strategy yet",
    "true_erp_x_range": "tensor_order inference disabled or unsafe",
    "source_box": "token footprint writer not implemented in Task 2-1",
    "base_score": "no selection score at write_new_memory stage"
  },
  "selection_effect": "diagnosis_only"
}
```

## 7. Frame Summary Schema

Frame summaries are the preferred first implementation target because they control log size.

| Field | Type | Notes |
|---|---:|---|
| `schema_version` | string | `"geo_memory_summary_v1"`. |
| `stage` | string | `train`, `pseudo_label`, `val`, or `smoke`. |
| `global_step` | int/null | Current step if available. |
| `mem_type` | string | `source` / `target`. |
| `batch_id` | int/null | `null` if aggregated over batch. |
| `image_id` | string/null | Optional. |
| `tensor_order` | int | Current tensor order. |
| `frame_idx` | int | Current frame index. |
| `frame_cnt` | int | Current sequence length. |
| `true_erp_x_range_reliable` | bool | Whether seam stats can be trusted. |
| `random_reverse_possible` | bool | Dataset/order risk flag. |
| `mem_hw` | list[int,int] | Memory grid. |
| `memory_num_tokens` | int | Expected token total. |
| `token_total_checked` | int | Count emitted or summarized. |
| `token_grid_valid` | bool | Whether mapping bounds and totals pass. |
| `region_counts` | dict | Counts for `seam`, `polar`, `seam_polar`, `equator`, `unknown`. |
| `region_ratios` | dict | Region counts divided by valid token total. |
| `latitude_band_counts` | dict | `top`, `middle`, `bottom`, `unknown`. |
| `kept_counts_by_region` | dict | For stages with keep/drop info. |
| `dropped_counts_by_region` | dict | For stages with keep/drop info. |
| `base_score_stats_by_region` | dict | `mean`, `std`, `min`, `max`, `count`, `nan_count`. |
| `geo_score_stats_by_region` | dict | Same statistics, diagnosis-only. |
| `compression_stats` | dict | Copy of current compression stats. |
| `sparse_stats` | dict | Copy of current sparse stats. |
| `selection_effect` | string | Must be `"diagnosis_only"` in Phase 2. |

## 8. Required Aggregations for Phase 2 Diagnostics

Future Task 2-2 / 2-3 should produce these summaries without requiring full-val predictions:

1. `region_distribution`: token counts and ratios for `seam`, `polar`, `seam_polar`, `equator`, `unknown`.
2. `latitude_band_distribution`: top / middle / bottom counts; sum must equal `memory_num_tokens` for valid grids.
3. `seam_reliability_distribution`: counts of records with `true_erp_x_range_reliable=true/false`.
4. `kept_dropped_by_region`: keep/drop counts and ratios grouped by assigned region.
5. `score_stats_by_region`: `base_score`, `geo_score`, and optional `final_score` stats grouped by region.
6. `compression_stage_stats`: before/after token counts, ratios, and stage labels.
7. `sparse_retrieval_stats`: tokens before/after, keep ratio, selected index coverage if available.
8. `mapping_validity`: token row/col bounds checks and `mem_h * mem_w == memory_num_tokens` checks.

These diagnostics are independent of GM0/GM1 prediction-mask completeness.

## 9. Unavailable Field Policy

Do not infer or fabricate missing geometry.

- If true image IDs are unavailable, set `image_id=null` and record `reason_unavailable.image_id`.
- If true ERP x coordinates are unavailable or unreliable, set `true_erp_x_range=null`, `true_erp_x_range_reliable=false`, and `seam_region="unknown"`.
- If only tensor order is available and `random_reverse_possible=true`, tensor-order horizontal inference is diagnostic only and must not drive seam assignment.
- If per-token base scores are unavailable, set `base_score=null`, `base_score_source="none"`, and aggregate only token counts.
- If compression uses pooling and does not expose exact source-token IDs, record compression summary fields first; per-source-token kept/dropped mapping can remain `not_applicable` until a later implementation task.

## 10. Validation Checklist for Later Implementation

Before using this schema for any conclusion:

- `memory_num_tokens == mem_h * mem_w` for uncompressed token grids.
- every emitted `token_id` maps to in-bound `token_row/token_col`.
- top + middle + bottom token counts equal total token count.
- region count sum equals total token count for valid summaries.
- WildPASS-like records with random reverse and no true coordinates keep `seam_region="unknown"`.
- `selection_effect` remains `"diagnosis_only"` until an explicit later task enables selection changes.
- reports clearly separate Train / PL / Val / smoke stages.

## 11. Task 2-1 Conclusion

The schema is ready for diagnosis-only implementation in the next Phase 2 task. It is designed around memory token geometry, region distribution, score statistics, and kept/dropped summaries, and does not rely on full prediction-mask availability or official full-val evaluation.
