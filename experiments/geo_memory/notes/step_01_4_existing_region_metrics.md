# Step 01-4 Existing Region Metrics Diagnosis

## 本次做了什么

- 先完成 input inventory，再对可用 matched 样本做 dry-run，dry-run 通过后才运行该 method 的 available-mask region metrics。
- 扫描 GM0 baseline_omnisam 与 GM1 tecmem_base / A_Dtrain_Etrain 的 prediction mask 路径，并使用 DensePASS GT。
- 对每个 method 统计 matched / missing_pred / missing_gt / size_mismatch。
- 检查 label id 范围、ignore_index、DensePASS13 class mapping 与原 OmniSAM validation 保持一致。

## 输出文件

- `experiments/geo_memory/results/region_metrics_existing.json`
- `experiments/geo_memory/tables/region_metrics_existing.md`
- `experiments/geo_memory/notes/step_01_4_existing_region_metrics.md`
- dry-run/full raw outputs under `experiments/geo_memory/raw_results/GM*/`

## Class mapping / ignore_index

- DensePASS13 mapping from dataset/DensePASS.py: [0,1,2,3,4,5,6,7,8,9,10,11,11,12,12,12,255,12,12]; ignore_index=255; num_classes=13; valid GT follows validation.py fast_hist filter GT∈[0,13).
- Raw DensePASS GT `labelTrainIds` is mapped to 13 classes before region metrics; using raw 19-class GT directly would be inconsistent.
- `ignore_index=255` pixels are excluded from intersection/union and polar error.

## 关键结论

- GM0 baseline_omnisam prediction masks were not found in the scanned final/confirmed prediction paths, so region metrics are recorded as N/A rather than fabricated.
- GM1 A_Dtrain_Etrain has 12 matched confirmed prediction masks; 88 GT masks have no corresponding confirmed prediction mask, so metrics are diagnostic on available masks only.
- 因 GM0 缺少 comparable prediction masks，当前不能可靠回答 memory 压缩后 seam/polar 是否相对 GM0 退化；该项标记为证据不足。
- GM1 available-mask diagnostics can motivate Geo-TeCMA region-wise analysis, but claims must be limited to the matched subset unless GM0/full predictions are exported later.

## 禁止项确认

- training_run: false
- full_training_run: false
- model_changed: false
- memory_selection_changed: false
- loss_changed: false
- inference_graph_changed: false
