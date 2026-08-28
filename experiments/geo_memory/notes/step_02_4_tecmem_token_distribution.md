# Step 02-4 — Analyze Original TeCMA Token Retention Distribution

Run Level: L2 diagnosis.

## 1. 本次做了什么

- 基于 step_02_3 已生成的 token geometry metadata，统计原 TeCMA / GM1 诊断样本的 region distribution。
- 检查是否可以统计 kept token region distribution、dropped token region distribution、retention ratio by region、avg base_score by region、sampled token keep ratio。
- 严格按方案要求：缺失项写 N/A，不补假数据。

## 2. 修改了哪些文件

- `experiments/geo_memory/tables/token_region_distribution_tecmem.md`
- `experiments/geo_memory/notes/step_02_4_tecmem_token_distribution.md`

## 3. 生成了哪些输出

- 主表：`experiments/geo_memory/tables/token_region_distribution_tecmem.md`
- 本 step 报告：`experiments/geo_memory/notes/step_02_4_tecmem_token_distribution.md`

## 4. 输入来源

- `experiments/geo_memory/diagnostics/token_region_sample.json`
- `experiments/geo_memory/results/token_geometry_diag.jsonl`
- `experiments/geo_memory/results/token_writer_sample.json`

## 5. 核心统计表

| Region | Total Tokens | Kept Tokens | Dropped Tokens | Retention Ratio | Avg Base Score | Notes |
|---|---:|---:|---:|---:|---:|---|
| seam | 0 | N/A | N/A | N/A | N/A | sample has 0 tokens in this region; kept/dropped not exposed by step_02_3 metadata; base_score unavailable |
| polar | 1280 | N/A | N/A | N/A | N/A | kept/dropped not exposed by step_02_3 metadata; base_score unavailable |
| seam_polar | 0 | N/A | N/A | N/A | N/A | sample has 0 tokens in this region; kept/dropped not exposed by step_02_3 metadata; base_score unavailable |
| equator | 2816 | N/A | N/A | N/A | N/A | kept/dropped not exposed by step_02_3 metadata; base_score unavailable |
| unknown | 0 | N/A | N/A | N/A | N/A | sample has 0 tokens in this region; kept/dropped not exposed by step_02_3 metadata; base_score unavailable |

## 6. 必须统计项完成情况

| Required Item | Status | Evidence / Reason |
|---|---|---|
| kept token region distribution | N/A | step_02_3 records use `kept_or_dropped=not_applicable` or omit aggregate kept counts |
| dropped token region distribution | N/A | step_02_3 records do not expose dropped token state |
| retention ratio by region | N/A | kept/dropped counts unavailable, cannot compute |
| avg base_score by region | N/A | `base_score=null`, `base_score_source=none`; summary base stats unavailable |
| token keep ratio over sampled iterations | N/A | identity/no-op evidence has selected token counts, but not original TeCMA per-token keep/drop output |

## 7. 可用诊断信息

- Sampled event token grid: `[64, 64]`; total checked tokens: `4096`.
- Region counts: `{'seam': 0, 'polar': 1280, 'seam_polar': 0, 'equator': 2816, 'unknown': 0}`.
- Region ratios: `{'seam': 0.0, 'polar': 0.3125, 'seam_polar': 0.0, 'equator': 0.6875, 'unknown': 0.0}`.
- Diagnostic `geo_score` is available from latitude prior, but it is not TeCMA `base_score`.

| Region | Count | Avg Geo Score | NaN Count |
|---|---:|---:|---:|
| seam | 0 | N/A | 0 |
| polar | 1280 | 0.960418 | 0 |
| seam_polar | 0 | N/A | 0 |
| equator | 2816 | 0.489532 | 0 |
| unknown | 0 | N/A | 0 |

## 8. 是否跑训练

- 否。

## 9. 是否完整训练

- 否。

## 10. 是否改动 memory selection

- 否。只读取已有 metadata 并生成 markdown 统计。

## 11. 是否改动 inference

- 否。

## 12. 是否依赖旧 memory 线

- 是。本 step 目标是分析 GM1 / TeCMA-base 的 token region 分布；但当前 step_02_3 metadata 尚未暴露 TeCMA selection 的 per-token keep/drop 与 base_score。

## 13. 当前风险

- 当前 hook 位于 `write_new_memory` 诊断阶段，无法证明原 TeCMA selection 对 seam / polar / equator 的保留偏好。
- `true_erp_x_range_reliable=false` 且 `token_x=null`，seam / seam_polar 暂不可可靠统计。
- `base_score` 与 `selection_rank` 未暴露，不能计算 avg base_score 或 retention ratio。
- 如果后续论文需要 token retention claim，必须在 selection/compression 后增加不改变选择结果的 hook。

## 14. 下一步建议

- 在不改变 selection 的前提下，把诊断 hook 挪到或补到 D/E selection/compression 后，记录 `base_score`、`kept_or_dropped`、`selection_rank`。
- 为 patch/view 传递可靠 `source_box` 或 `true_erp_x_range`，否则 seam prior 只能保持 N/A 或降级为 view-level 诊断。
- 重新执行极小样本 L2 diagnosis 后再重跑 2-4，届时才能输出 retention_ratio_by_region。

## 15. 结论

- 目前所有 region 分布没有足够证据说明原 TeCMA 存在 seam / polar retention 问题；必须如实写 N/A。
- 已确认 latitude/polar/equator row-level 诊断可用，但 seam 与 original selection retention 诊断尚未闭环。
