# Step 20/21 Confirmed Export Resume Status — TeCMA Phase 7.2 + 7.3

**生成时间**: 2026-06-18（最终更新）  
**状态**: ✅ **全部完成**

---

## 最终验收状态

| Item | Status | Path | Complete? |
|---|---|---|---|
| 四配置 checkpoint 定位 | ✅ | `runs/*/checkpoints/*/best_iou*.pth` | ✅ |
| 四配置 prediction mask 导出 (4×12=48) | ✅ | `eval_predictions_confirmed/{cfg}/` | ✅ |
| 12 样本 confirmed side-by-side panel | ✅ | `figures/qualitative_confirmed/sample_{ID}/confirmed_panel.png` | ✅ |
| 12 样本个体 color 图 (4 cfg × 12) | ✅ | `figures/qualitative_confirmed/sample_{ID}/pred_{cfg}_color.png` | ✅ |
| 6 confirmed failure case panels | ✅ | `figures/failure_cases_confirmed/confirmed_{ID}_{type}.png` | ✅ |
| confirmed visualization report | ✅ | `notes/step_20_21_confirmed_prediction_visualization_report.md` | ✅ |
| 已有诊断产物（保留，未覆盖） | ✅ | `figures/qualitative_diagnostic/` | ✅ |
| 已有诊断报告（保留，未覆盖） | ✅ | `notes/step_20_21_diagnostic_visualization_report.md` | ✅ |

---

## Checkpoint 信息

| Config | Best Checkpoint | val_mIoU |
|---|---|---:|
| A_ccpm | `runs/A_ccpm_seed1_20260616_104911/…/best_iou60.26.pth` | 60.26 |
| A_Dtrain | `runs/A_Dtrain_seed1_20260616_111019/…/best_iou60.12.pth` | 60.12 |
| A_Etrain_keep025 | `runs/token_budget/A_Etrain_keep025_…/best_iou60.10.pth` | 60.10 |
| A_Dtrain_Etrain | `runs/A_Dtrain_Etrain_seed1_20260616_121135/…/best_iou60.02.pth` | 60.02 |

---

## 产物清单

### Prediction Masks（48 个，inference-only，不训练）

```
eval_predictions_confirmed/
  val_subset12.txt                      (12 样本列表)
  export_summary.json                   (导出摘要)
  A_ccpm/{24,68,101,…,363}_pred_argmax.png   (12 个)
  A_Dtrain/…                                 (12 个)
  A_Etrain_keep025/…                         (12 个)
  A_Dtrain_Etrain/…                          (12 个)
```

### Confirmed Panels（12 样本 × 7 文件 = 84 文件）

```
figures/qualitative_confirmed/sample_{ID}/
  confirmed_panel.png                   (6列：Input|GT|A_ccpm|A_Dtrain|A_Etrain_keep025|A_Dtrain_Etrain)
  input_panorama.png
  ground_truth_color.png
  pred_A_ccpm_color.png
  pred_A_Dtrain_color.png
  pred_A_Etrain_keep025_color.png
  pred_A_Dtrain_Etrain_color.png
```

### Confirmed Failure Cases（6 个）

```
figures/failure_cases_confirmed/
  confirmed_122_small_object_missing.png
  confirmed_319_thin_structure_broken.png
  confirmed_305_boundary_erosion.png
  confirmed_210_seam_inconsistency.png
  confirmed_101_class_confusion.png
  confirmed_24_pseudo-label_noise_suspected.png
```

---

## 技术说明

- **推断策略**: 标准推断（D/E 不在 inference 时激活），展示纯 training effect
- **推断脚本**: `experiments/tecma/scripts/export_confirmed_predictions.py`
- **可视化脚本**: `experiments/tecma/scripts/generate_confirmed_panels.py`
- **断点恢复**: 发现第一次会话已生成所有 48 个 prediction masks，本次只补生成了 panels
- **严格非操作**: 未训练、未生成伪标签、未覆盖已有文件
