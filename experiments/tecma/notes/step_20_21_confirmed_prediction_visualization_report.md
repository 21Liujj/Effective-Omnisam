# Step 20/21 Confirmed Prediction Visualization Report — TeCMA Phase 7.2 + 7.3

**生成时间**: 2026-06-18

## 四配置 Prediction Masks 来源

| Config | Checkpoint | val_mIoU |
|---|---|---:|
| A_ccpm | `runs/A_ccpm_seed1_20260616_104911/...best_iou60.26.pth` | 60.26 |
| A_Dtrain | `runs/A_Dtrain_seed1_20260616_111019/...best_iou60.12.pth` | 60.12 |
| A_Etrain_keep025 | `runs/token_budget/A_Etrain_keep025.../...best_iou60.10.pth` | 60.10 |
| A_Dtrain_Etrain | `runs/A_Dtrain_Etrain_seed1_20260616_121135/...best_iou60.02.pth` | 60.02 |

**推断策略**: 标准推断（D/E 不激活），对比纯 training effect。

---

## 12 样本 Confirmed Panel 状态

| Sample | Panel 路径 | Missing Configs | Status |
|---|---|---|---|
| 122 | `qualitative_confirmed/sample_122/confirmed_panel.png` | — | ✅ OK |
| 155 | `qualitative_confirmed/sample_155/confirmed_panel.png` | — | ✅ OK |
| 289 | `qualitative_confirmed/sample_289/confirmed_panel.png` | — | ✅ OK |
| 101 | `qualitative_confirmed/sample_101/confirmed_panel.png` | — | ✅ OK |
| 111 | `qualitative_confirmed/sample_111/confirmed_panel.png` | — | ✅ OK |
| 263 | `qualitative_confirmed/sample_263/confirmed_panel.png` | — | ✅ OK |
| 68 | `qualitative_confirmed/sample_68/confirmed_panel.png` | — | ✅ OK |
| 319 | `qualitative_confirmed/sample_319/confirmed_panel.png` | — | ✅ OK |
| 305 | `qualitative_confirmed/sample_305/confirmed_panel.png` | — | ✅ OK |
| 24 | `qualitative_confirmed/sample_24/confirmed_panel.png` | — | ✅ OK |
| 210 | `qualitative_confirmed/sample_210/confirmed_panel.png` | — | ✅ OK |
| 363 | `qualitative_confirmed/sample_363/confirmed_panel.png` | — | ✅ OK |

---

## 6 Confirmed Failure Cases

| Sample | Type | Panel | Missing Configs | Status |
|---|---|---|---|---|
| 122 | small_object_missing | `failure_cases_confirmed/confirmed_122_small_object_missing.png` | — | ✅ OK |
| 319 | thin_structure_broken | `failure_cases_confirmed/confirmed_319_thin_structure_broken.png` | — | ✅ OK |
| 305 | boundary_erosion | `failure_cases_confirmed/confirmed_305_boundary_erosion.png` | — | ✅ OK |
| 210 | seam_inconsistency | `failure_cases_confirmed/confirmed_210_seam_inconsistency.png` | — | ✅ OK |
| 101 | class_confusion | `failure_cases_confirmed/confirmed_101_class_confusion.png` | — | ✅ OK |
| 24 | pseudo-label_noise_suspected | `failure_cases_confirmed/confirmed_24_pseudo-label_noise_suspected.png` | — | ✅ OK |

---

## 严格非操作记录

- 未训练模型。
- 未生成伪标签。
- 未覆盖已有文件（`qualitative_diagnostic/`、`failure_cases_diagnostic/` 保持不变）。
- 推断仅使用 `eval_predictions_confirmed/` 中预先导出的预测 masks。
