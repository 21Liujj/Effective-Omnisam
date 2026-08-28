# Region Metrics Sample

Status: `ok`

## Synthetic Self-Test

Status: `passed`

| Region | mIoU | Valid Pixels |
|---|---:|---:|
| overall | 0.9231678486997636 | 63 |
| seam | 1.0 | 15 |
| polar | 0.4838709677419355 | 31 |
| equator | 0.9131944444444444 | 24 |

## Directory Evaluation

Status: `N/A`
Reason: `no prediction masks found in /tmp/geo_missing_pred`

## Implementation Notes

- Region masks reuse `experiments.geo_memory.geometry_ops.build_region_mask`.
- `ignore_index` pixels are excluded from intersection and union before IoU computation.
- Overall region mIoU and per-class IoU by region are implemented for matched masks.
- Missing prediction / GT inputs return `N/A` instead of failing or inventing metrics.
- No training, model, memory selection, sparse retrieval, attention, or loss code is touched.
