# Token Region Distribution — TeCMA / GM1 diagnosis

Run Level: L2 diagnosis.

Source metadata:
- `experiments/geo_memory/diagnostics/token_region_sample.json`
- `experiments/geo_memory/results/token_geometry_diag.jsonl`
- `experiments/geo_memory/results/token_writer_sample.json`

| Region | Total Tokens | Kept Tokens | Dropped Tokens | Retention Ratio | Avg Base Score | Notes |
|---|---:|---:|---:|---:|---:|---|
| seam | 0 | N/A | N/A | N/A | N/A | sample has 0 tokens in this region; kept/dropped not exposed by step_02_3 metadata; base_score unavailable |
| polar | 1280 | N/A | N/A | N/A | N/A | kept/dropped not exposed by step_02_3 metadata; base_score unavailable |
| seam_polar | 0 | N/A | N/A | N/A | N/A | sample has 0 tokens in this region; kept/dropped not exposed by step_02_3 metadata; base_score unavailable |
| equator | 2816 | N/A | N/A | N/A | N/A | kept/dropped not exposed by step_02_3 metadata; base_score unavailable |
| unknown | 0 | N/A | N/A | N/A | N/A | sample has 0 tokens in this region; kept/dropped not exposed by step_02_3 metadata; base_score unavailable |

## Token Keep Ratio over Sampled Iterations

| Source | Selected Tokens Before | Selected Tokens After | Sparse Tokens Before | Sparse Tokens After | Interpretable as TeCMA keep ratio? |
|---|---:|---:|---:|---:|---|
| `token_writer_sample.json.identity_noop_evidence` | 4096 | 4096 | 4096 | 4096 | No; diagnosis/identity evidence, not per-token TeCMA keep/drop output |

## Diagnostic Geo Score by Region

This is latitude-prior diagnostic information, not TeCMA base_score.

| Region | Count | Avg Geo Score | NaN Count |
|---|---:|---:|---:|
| seam | 0 | N/A | 0 |
| polar | 1280 | 0.960418 | 0 |
| seam_polar | 0 | N/A | 0 |
| equator | 2816 | 0.489532 | 0 |
| unknown | 0 | N/A | 0 |

## Interpretation

- The available step_02_3 metadata covers region assignment for one sampled target memory write event.
- It does not expose per-token `kept_or_dropped`, `selection_rank`, or `base_score`; therefore kept/dropped distributions, retention ratios, and avg base_score are reported as N/A rather than inferred.
- Region coverage exists for `polar` and `equator`; seam-related regions remain unavailable because true ERP x range is unreliable and tensor-order horizontal inference is disabled.
- Current evidence is sufficient to validate latitude/row mapping diagnostics, but insufficient to claim original TeCMA token retention bias by region.
