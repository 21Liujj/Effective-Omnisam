# Phase 2 / Task 2-2 — Diagnosis-only Token Writer

Date: 2026-06-29
Run Level: L2 dry-run / diagnosis-only implementation
Scope: capped summary/token writer for Geo-Memory diagnostics

## 0. Boundary

This task implements only a diagnosis writer for memory-token geometry summaries and capped token samples.

- No training is run by this task.
- No pseudo-label generation is run by this task.
- No memory compression behavior is changed.
- No sparse retrieval behavior is changed.
- No memory attention input is changed.
- No loss input or loss formula is changed by the writer.
- The writer follows `experiments/geo_memory/notes/step_02_1_token_metadata_schema.md`.
- Unavailable per-token fields are written as `null` / `not_applicable` with `reason_unavailable` instead of being fabricated.

## 1. Implemented Files

| File | Change |
|---|---|
| `experiments/geo_memory/token_diag_writer.py` | New standalone diagnosis-only helper that builds `geo_memory_summary_v1` and capped `geo_memory_token_v1` records. |
| `model/memory_bank.py` | Stores latest memory-grid metadata and existing stats in `last_geo_memory_diag`; does not alter memory tensors. |
| `adaptation/strategies.py` | Calls writer only when `plugins.geo_memory.token_writer.enabled=true`; collects returned summary/sample records. |
| `adaptation/trainer.py` | Applies `first_n_batches`, `max_summary_records`, and `max_records` caps before passing records to engine. |
| `adaptation/engine.py` | Writes summary/sample JSONL only when the writer is explicitly enabled. |
| `adaptation/plugins/manager.py` | Adds default-disabled `geo_memory.token_writer` config. |
| `configs/ablations/omnisam_base.yaml` | Mirrors default-disabled writer config. |
| `docs/claude/module-map.md` | Registers Task 2-2 writer defaults and code paths. |

## 2. Default Write Limits

```yaml
geo_memory:
  token_writer:
    enabled: false
    summary_jsonl: experiments/geo_memory/results/token_geometry_summary.jsonl
    token_jsonl: experiments/geo_memory/results/token_geometry_diag.jsonl
    max_records: 5
    max_summary_records: 200
    sample_rate: 1.0
    first_n_batches: 1
```

The default state is disabled. When enabled, token-level output is capped by `max_records`, and collection is limited by `first_n_batches` to avoid large JSONL files.

## 3. Identity / No-op Evidence

Static and synthetic validation were used instead of training.

| Evidence item | Result |
|---|---|
| `py_compile` | Passed for `model/memory_bank.py`, `adaptation/strategies.py`, `adaptation/trainer.py`, `adaptation/engine.py`, `adaptation/plugins/manager.py`, and `experiments/geo_memory/token_diag_writer.py`. |
| `git diff --check` | Passed for all Task 2-2 touched files. |
| Writer default | `token_writer.enabled=false`, so no writer path runs unless explicitly enabled. |
| Selected token counts | Writer reads `num_tokens`, `last_memory_tokens_before`, `last_memory_tokens_after`; it never writes to `to_cat_memory`, `feats`, `memory`, `memory_pos_embed`, or selected indices. |
| Sparse retrieval counts | Writer reads `last_sparse_tokens_before`, `last_sparse_tokens_after`, `last_sparse_keep_ratio`; it never calls or modifies `sparse_retriever.retrieve`. |
| Memory attention input | Writer is called after model forward diagnostics are available; it does not feed into `memory_attention`. |
| Loss input | Writer records only diagnostics after `pred_t` / memory stats exist; it does not change `pred_s`, `pred_t`, labels, masks, `loss_sup_s`, `loss_unsup`, or `loss_proto`. |

Synthetic writer check:

```text
summary_schema=geo_memory_summary_v1
token_records=5
token_total_checked=4096
region_counts={seam:0, polar:1280, seam_polar:0, equator:2816, unknown:0}
latitude_band_counts={top:640, middle:2816, bottom:640, unknown:0}
```

This confirms the cap emits 5 token samples while the summary still checks all 4096 tokens for a `[64,64]` memory grid.

## 4. Unavailable Fields Policy

Current implementation intentionally writes unavailable fields as null or not applicable:

- `image_id=null`: dataset metadata is not provided to strategy yet.
- `patch_id=null`: dataset patch metadata is not provided to strategy yet.
- `view_id=null`: dataset view metadata is not provided to strategy yet.
- `source_box=null`: token footprint writer is not implemented in Task 2-2.
- `base_score=null`: per-token selection score is not exposed by current memory path.
- `final_score=null`: diagnosis writer does not change selection scores.
- `kept_or_dropped=not_applicable`: `write_new_memory` stage does not expose per-token keep/drop state.
- `selection_rank=null`: per-token top-k rank is not exposed by current memory path.
- `keep_ratio=null`: per-token keep ratio is not applicable at `write_new_memory` stage.
- `true_erp_x_range=null` when unreliable: tensor-order horizontal inference is disabled or unsafe.
- `token_x=null` when x-range is unreliable: true ERP x range is unavailable or unreliable.

## 5. Sample Token Records

The synthetic validation emitted 5 records; below are 3 representative abbreviated samples. These records follow the schema fields from Task 2-1; fields omitted here are present in the generated record and validated by key inspection.

```json
{"schema_version":"geo_memory_token_v1","run_level":"L2_dry_run","stage":"train","memory_event":"write_new_memory","mem_type":"target","token_id":0,"token_row":0,"token_col":0,"mem_hw":[64,64],"memory_num_tokens":4096,"latitude_band":"top","seam_region":"unknown","assigned_region":"polar","base_score":null,"geo_score":0.9999247018391445,"kept_or_dropped":"not_applicable","selection_effect":"diagnosis_only"}
```

```json
{"schema_version":"geo_memory_token_v1","run_level":"L2_dry_run","stage":"train","memory_event":"write_new_memory","mem_type":"target","token_id":1,"token_row":0,"token_col":1,"mem_hw":[64,64],"memory_num_tokens":4096,"latitude_band":"top","seam_region":"unknown","assigned_region":"polar","base_score":null,"geo_score":0.9999247018391445,"kept_or_dropped":"not_applicable","selection_effect":"diagnosis_only"}
```

```json
{"schema_version":"geo_memory_token_v1","run_level":"L2_dry_run","stage":"train","memory_event":"write_new_memory","mem_type":"target","token_id":2,"token_row":0,"token_col":2,"mem_hw":[64,64],"memory_num_tokens":4096,"latitude_band":"top","seam_region":"unknown","assigned_region":"polar","base_score":null,"geo_score":0.9999247018391445,"kept_or_dropped":"not_applicable","selection_effect":"diagnosis_only"}
```

## 6. Sample Summary Record

```json
{"schema_version":"geo_memory_summary_v1","stage":"train","mem_type":"target","frame_idx":0,"frame_cnt":9,"true_erp_x_range_reliable":false,"random_reverse_possible":true,"mem_hw":[64,64],"memory_num_tokens":4096,"token_total_checked":4096,"token_grid_valid":true,"region_counts":{"seam":0,"polar":1280,"seam_polar":0,"equator":2816,"unknown":0},"latitude_band_counts":{"top":640,"middle":2816,"bottom":640,"unknown":0},"kept_counts_by_region":null,"dropped_counts_by_region":null,"base_score_stats_by_region":null,"selection_effect":"diagnosis_only"}
```

## 7. Validation Commands

```bash
/root/miniconda3/envs/omnisam/bin/python -m py_compile   model/memory_bank.py adaptation/strategies.py adaptation/trainer.py adaptation/engine.py   adaptation/plugins/manager.py experiments/geo_memory/token_diag_writer.py

git diff --check --   model/memory_bank.py adaptation/strategies.py adaptation/trainer.py adaptation/engine.py   adaptation/plugins/manager.py configs/ablations/omnisam_base.yaml   docs/claude/module-map.md experiments/geo_memory/token_diag_writer.py
```

Both checks passed.

## 8. Conclusion

Task 2-2 implements a capped, default-disabled diagnosis writer for memory token geometry. It records region distribution and available score/selection statistics without depending on GM0/GM1 full prediction completeness and without changing compression, sparse retrieval, memory attention, or loss behavior.

