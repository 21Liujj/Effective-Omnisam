# Phase 1 / Task 1-1 — Geometry Ops Implementation

Date: 2026-06-29
Repo: `/root/autodl-tmp/OmniSAM-main`
Run Level: L1 unit-test

## 1. What Was Done

Implemented standalone Geo-Memory geometry ops for ERP/token-grid diagnostics. The repository has no `omnisam/` root package, so the planned path `omnisam/geo_memory/geometry_ops.py` is mapped to `experiments/geo_memory/geometry_ops.py` under the existing experiment namespace.

Implemented functions:

1. `compute_latitude_map(H, W)`
2. `compute_latitude_prior(H, W)`
3. `compute_seam_distance_map(H, W)`
4. `compute_seam_prior(H, W, seam_width, tau)`
5. `build_region_mask(H, W, seam_width, polar_band_ratio)`
6. `assign_region_from_xy(x, y, H, W)`
7. `assign_region_from_box(box, H, W)`
8. `resize_region_map_to_token_grid(region_map, token_H, token_W)`
9. `compute_token_row_region_counts(token_H, token_W, polar_band_ratio)` for token-row latitude diagnostics.

## 2. Modified / Added Files

- `experiments/geo_memory/__init__.py`
- `experiments/geo_memory/geometry_ops.py`
- `experiments/geo_memory/unit_tests/test_geometry_ops.py`
- `experiments/geo_memory/unit_tests/test_geometry_ops_report.json`
- `experiments/geo_memory/unit_tests/test_geometry_ops_report.md`
- `experiments/geo_memory/notes/step_01_1_geometry_ops_impl.md`

## 3. Outputs

Unit test output files:

- `experiments/geo_memory/unit_tests/test_geometry_ops_report.json`
- `experiments/geo_memory/unit_tests/test_geometry_ops_report.md`

## 4. Region Priority

`build_region_mask` uses mutually exclusive labels with explicit priority:

1. `seam_polar` if both seam and polar.
2. `seam` if seam-only.
3. `polar` if polar-only.
4. `equator` for remaining valid cells.
5. `unknown` is reserved for unmappable coordinates and is not assigned to valid cells.

This prevents one position from receiving mixed/conflicting labels.

## 5. Token-Grid Latitude Support

`compute_latitude_map`, `compute_latitude_prior`, and `build_region_mask` accept arbitrary `H, W`, including token grids such as `64 x 64`. Latitude/polar/equator diagnostics can therefore be computed at token-grid level and depend only on token row.

The unit test specifically validates `1024 x 1024 -> 64 x 64` with `resize_region_map_to_token_grid` because current smoke records use `mem_hw=[64, 64]`.

## 6. Full-Height Patch Warning

`assign_region_from_xy` is reliable for point coordinates. `assign_region_from_box` is only a coarse center-point diagnostic. When a patch or box covers full ERP height, box-level polar/equator judgment is unreliable because the box spans polar and equator rows. Later latitude prior must use token-row labels based on `mem_hw`, not patch-level or box-level labels.

## 7. Unit Test Summary

The unit test checks:

- top/bottom rows have stronger high-latitude prior than middle rows;
- middle rows are equator-like;
- left/right columns have stronger seam prior than middle columns;
- corner region is `seam_polar`;
- region priority is mutually exclusive;
- `1024 x 1024 -> 64 x 64` categorical resize works and has no NaN;
- `64 x 64` region counts sum to `4096`;
- `assign_region_from_xy` and `assign_region_from_box` cover seam / polar / seam_polar / equator / unknown.

## 8. Training / Model Impact

- Training run: no.
- Full training: no.
- PL generation: no.
- Memory selection changed: no.
- Sparse retrieval changed: no.
- Memory attention changed: no.
- Loss changed: no.
- Inference changed: no.
- Dependency on old memory line: no.

## 9. Current Risk

Low. The new code is standalone under `experiments/geo_memory/` and is not imported by training/model paths.

## 10. Next Step

Proceed to Phase 1 / Task 1-2 if needed: run/extend geometry ops unit-test reporting, still without training or model changes.
