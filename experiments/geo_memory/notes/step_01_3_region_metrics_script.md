# Phase 1 / Task 1-3 — Region Metrics Script

Date: 2026-06-29
Repo: `/root/autodl-tmp/OmniSAM-main`
Run Level: L1 scripts, no training

## 1. What Was Done

Implemented standalone region metrics evaluation script:

- `experiments/geo_memory/scripts/eval_region_metrics.py`

The script evaluates overall / seam / polar / equator mIoU and per-class IoU by region for matched prediction and GT masks.

## 2. Geometry Definition Reuse

Region masks reuse `experiments.geo_memory.geometry_ops.build_region_mask`. The script does not define a separate seam / polar / equator logic, so region labels stay aligned with Task 1-1 / Task 1-2.

Region groups:

- `seam`: `seam` + `seam_polar`
- `polar`: `polar` + `seam_polar`
- `equator`: `equator`
- `overall`: all non-ignored pixels

## 3. Missing Prediction / GT Handling

If prediction directory or GT directory is missing or not provided, the script returns `N/A` and writes outputs. It does not crash and does not fabricate metrics.

## 4. ignore_index Logic

Pixels with `gt == ignore_index` are excluded before intersection / union computation. They do not contribute to valid pixels, intersection, or union in any region.

## 5. Synthetic Self-Test

The script includes a synthetic 8 x 8 pred/GT self-test that verifies:

- overall mIoU calculation;
- seam mIoU calculation;
- polar mIoU calculation;
- equator mIoU calculation;
- `ignore_index` exclusion from intersection / union.

## 6. Outputs

Sample outputs:

- `experiments/geo_memory/results/region_metrics_sample.json`
- `experiments/geo_memory/tables/region_metrics_sample.md`

## 7. per-class IoU Status

Per-class IoU by region is implemented in the JSON structure under each region's `per_class_iou`. The markdown sample summarizes region mIoU and valid pixels for readability.

## 8. Training / Model Impact

- Training run: no.
- Full training: no.
- Model changed: no.
- Memory selection changed: no.
- Sparse retrieval changed: no.
- Memory attention changed: no.
- Loss changed: no.
- PL generation: no.

## 9. Current Risk

Low. The script is standalone under `experiments/geo_memory/scripts/` and only imports `geometry_ops` plus standard evaluation dependencies.

## 10. Next Step

Task 1-4 can use this script to attempt diagnostics on existing baseline / memory-line prediction masks. If masks are missing, it should record `N/A` rather than invent metrics.
