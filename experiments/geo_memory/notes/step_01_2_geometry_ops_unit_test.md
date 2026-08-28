# Phase 1 / Task 1-2 — Geometry Ops Unit Test

Date: 2026-06-29
Repo: `/root/autodl-tmp/OmniSAM-main`
Run Level: L1 unit-test

## Scope

This step runs and extends Geo-Memory geometry ops unit tests only. It does not run training, does not modify model paths, does not modify memory selection, and does not touch sparse retrieval, memory attention, or loss.

## Tests Inherited From Task 1-1

The following checks were introduced during Task 1-1 and re-run here as regression checks rather than counted as new design work:

- latitude prior: top/bottom rows are stronger than middle rows;
- seam prior: left/right columns are stronger than middle columns;
- region mask priority: corner is `seam_polar`, top middle is `polar`, left middle is `seam`, center is `equator`;
- 64 x 64 token-grid region ids are mutually exclusive and sum to 4096;
- 1024 x 1024 to 64 x 64 categorical resize preserves expected regions and has no NaN;
- normal point/box assignment covers seam, polar, seam_polar, equator, and unknown.

## New / Rechecked In Task 1-2

Task 1-2 added explicit full-height patch/box checks:

- non-seam full-height box `(30, 0, 34, 64)` returns `unknown`, not `polar` or `equator`;
- seam full-height box `(0, 0, 2, 64)` returns `seam` only;
- `assign_region_from_box` now encodes the rule that full-height boxes must not be collapsed into a single polar/equator label.

Reason: when a panoramic patch/box covers the full ERP height, it contains top polar rows, middle equator rows, and bottom polar rows simultaneously. Therefore box-level polar/equator assignment is only auxiliary; subsequent latitude prior must use token-row labels based on `mem_hw`.

## Count Checks

64 x 64 region counts:

```json
{
  "seam": 384,
  "polar": 896,
  "seam_polar": 128,
  "equator": 2688
}
```

Sum: 4096 / 4096.

1024 x 1024 to 64 x 64 resized region counts:

```json
{
  "seam": 384,
  "polar": 896,
  "seam_polar": 128,
  "equator": 2688
}
```

Sum: 4096 / 4096. Resize output has no NaN.

Token-row top/middle/bottom counts:

```json
{
  "top": 512,
  "middle": 3072,
  "bottom": 512
}
```

Sum: 4096 / 4096.

## Validation

Passed:

```bash
/root/miniconda3/envs/omnisam/bin/python -m py_compile   experiments/geo_memory/geometry_ops.py   experiments/geo_memory/unit_tests/test_geometry_ops.py

/root/miniconda3/envs/omnisam/bin/python   experiments/geo_memory/unit_tests/test_geometry_ops.py
```

## Impact Statement

- Training run: no.
- Full training: no.
- Model changed: no.
- Memory selection changed: no.
- Sparse retrieval changed: no.
- Memory attention changed: no.
- Loss changed: no.
- PL generation: no.
