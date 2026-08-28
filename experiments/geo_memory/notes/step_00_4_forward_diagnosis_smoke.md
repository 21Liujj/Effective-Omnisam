# Phase 0 / Task 0-4 — Forward Diagnosis Smoke Test

Date: 2026-06-29
Repo: `/root/autodl-tmp/OmniSAM-main`
Mode: minimal forward-diagnosis smoke only. No training, no backward, no pseudo-label generation.

## Scope

- The smoke test exercises Geo-TeCMA diagnosis record construction and JSONL writing only.
- It does not change token selection, sparse retrieval, memory attention, model loss, or optimizer state.
- It writes synthetic diagnosis records for DensePASS-like and WildPASS-like contexts to verify metadata semantics before any training run.

## Artifacts

- JSONL: `experiments/geo_memory/results/geo_memory_diag.jsonl`
- Summary: `experiments/geo_memory/results/geo_memory_diag_smoke_00_4_summary.json`

## Counts

- Scenarios: DensePASS-like and WildPASS-like.
- Batches per scenario: 2.
- Frames per batch: 9.
- Memory frames per batch: 9.
- Total records: 36.
- Per-scenario/batch counts:
  - DensePASS-like batch 0: 9
  - DensePASS-like batch 1: 9
  - WildPASS-like batch 0: 9
  - WildPASS-like batch 1: 9

The record count matches the expected patch/view count and memory frame count for each batch.

## Reliability Distribution

- DensePASS-like: `true_erp_x_range_reliable=True` for 18 / 18 records.
- WildPASS-like: `true_erp_x_range_reliable=False` for 18 / 18 records.

## Seam Region Distribution

- DensePASS-like: `left_seam=4`, `non_seam=10`, `right_seam=4`.
- WildPASS-like: `unknown=18`.

WildPASS-like horizontal ERP coordinates are treated as unreliable because random reverse can invalidate tensor-order inference. Therefore `true_erp_x_range=null` and `seam_region=unknown`; no seam label is fabricated.

## Token Mapping Check

- `mem_hw=[64, 64]`, `memory_num_tokens=4096` in the smoke records.
- Mapping checked:
  - `token_row = token_idx // mem_w`
  - `token_col = token_idx % mem_w`
- Max mapped row/col for the record shape: row 63, col 63.
- Violations: 0.
- No `mem_hw`, token count, row mapping, or col mapping fields were `None`.

## Validation Commands

```bash
/root/miniconda3/envs/omnisam/bin/python -m py_compile   adaptation/plugins/manager.py   model/memory_bank.py   adaptation/strategies.py   adaptation/trainer.py   adaptation/engine.py

git diff --check --   adaptation/plugins/manager.py   model/memory_bank.py   adaptation/strategies.py   adaptation/trainer.py   adaptation/engine.py   configs/ablations/omnisam_base.yaml   docs/claude/module-map.md   experiments/geo_memory/notes/step_00_3_geometry_metadata_diagnosis.md
```

Both checks passed after adding the `seam_region` diagnosis field.
