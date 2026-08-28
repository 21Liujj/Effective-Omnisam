# Phase 0 / Task 0-3 — Geometry Metadata Diagnosis-Only

Date: 2026-06-29
Repo: `/root/autodl-tmp/OmniSAM-main`
Mode: implementation, diagnosis-only. No training was run by Claude.

## Goal

Implement the next safe step after Task 0-2: geometry metadata logging only. This does not enable Geo-TeCMA token selection and does not change loss, memory attention, sparse retrieval, or pseudo-label filtering.

## User constraints implemented

1. Geometry metadata separates `tensor_order`, `frame_idx`, `true_erp_x_range`, and `true_erp_y_range`.
2. Seam / horizontal position is not trusted from tensor order when random reverse may affect the sample.
3. Latitude / polar diagnosis is based on `mem_hw` token-row mapping, not patch-level labels.
4. Seam-aware selection remains disabled; `seam_selection_enabled` is forced to diagnosis-only behavior.
5. The next enabled behavior is geometry metadata logging / diagnosis-only, not Geo-TeCMA selection.

## Code changes

- `adaptation/plugins/manager.py`: added default `plugins.geo_memory` config with `enabled: false`.
- `configs/ablations/omnisam_base.yaml`: added matching default YAML block.
- `model/memory_bank.py`: extended runtime context with `geo_context`, stores `geometry_meta`, exposes `get_last_geo_memory_diag()`.
- `adaptation/strategies.py`: builds per-frame target/source geometry context and records diagnosis after memory encoding; does not alter `mask_t`, sparse selection, memory attention, or losses.
- `adaptation/trainer.py`: collects per-frame geometry diagnosis records per epoch.
- `adaptation/engine.py`: writes records to `experiments/geo_memory/results/geo_memory_diag.jsonl` by default.
- `docs/claude/module-map.md`: registered `plugins.geo_memory` defaults and diagnosis-only behavior.

## JSONL fields

Each record includes `tensor_order`, `frame_idx`, `frame_cnt`, `true_erp_x_range`, `true_erp_y_range`, `inferred_erp_x_range_from_tensor_order`, `true_erp_x_range_reliable`, `random_reverse_possible`, `tensor_order_inference_allowed`, `mem_hw`, `memory_num_tokens`, `token_row_mapping`, `token_col_mapping`, `latitude_source`, `seam_selection_enabled`, and `selection_effect`.

## Safety notes

For WildPASS-like adaptation, `random_reverse_possible=true` and `true_erp_x_range_reliable=false` unless a later dataset change records actual order. This prevents silently treating tensor order as physical ERP seam position.

`true_erp_y_range` remains `[0, erp_height]` because the current panorama patching is width-only and each patch covers the full ERP height.

## Verification

Recommended user-side checks:

```bash
python -m py_compile adaptation/plugins/manager.py model/memory_bank.py adaptation/strategies.py adaptation/trainer.py adaptation/engine.py
```

For a short diagnosis run, enable only:

```yaml
plugins:
  geo_memory:
    enabled: true
    mode: diagnosis
    seam_selection_enabled: false
```

Then inspect:

```bash
head -n 5 experiments/geo_memory/results/geo_memory_diag.jsonl
```

## Cleanup / Verification

- Removed duplicate geometry helper copy from `NoMemStrategy`; it inherits the `WithMemStrategy` implementation.
- Cleared trailing whitespace reported by `git diff --check`.
- Syntax check passed with `/root/miniconda3/envs/omnisam/bin/python -m py_compile adaptation/plugins/manager.py model/memory_bank.py adaptation/strategies.py adaptation/trainer.py adaptation/engine.py`.
- `git diff --check` passed for the Task 0-3 touched files.
- Confirmed only one `_build_geo_context` definition remains in `adaptation/strategies.py`.

No training, PL generation, or selection-changing experiment was run by Claude.
