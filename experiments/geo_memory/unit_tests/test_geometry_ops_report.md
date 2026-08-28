# Geometry Ops Unit Test Report

Status: passed

## Key Checks
- top/bottom rows marked high-latitude relative to middle rows
- left/right columns have stronger seam prior than middle columns
- 64x64 region ids have clear priority and sum to 4096
- 1024x1024 -> 64x64 categorical resize preserves expected regions
- assign_region_from_xy/box returns seam, polar, seam_polar, equator, unknown
- full-height boxes are not collapsed to polar/equator; non-seam full-height returns unknown
- token-row top/middle/bottom counts sum to 4096

## 64x64 Region Counts

{
  "seam": 384,
  "polar": 896,
  "seam_polar": 128,
  "equator": 2688
}

## 1024x1024 -> 64x64 Region Counts

{
  "seam": 384,
  "polar": 896,
  "seam_polar": 128,
  "equator": 2688
}

## Token Row Counts

{
  "top": 512,
  "middle": 3072,
  "bottom": 512
}

No training, memory selection, sparse retrieval, attention, or loss code was run.
