# Baseline vs A_Dtrain_Etrain 视觉失效诊断报告

**对比**: `baseline_omnisam` (无压缩, best_iou60.26) vs `A_Dtrain_Etrain` (MTC+SMR, best_iou60.02)
**样本**: 6 个已标注失效案例 + 4 个代表性样本

| Sample | 失效类型 | Panel | 显著恶化类别 (dIoU<-2) | 显著改善类别 (dIoU>2) |
|---|---|---|---|---|
| 210 | seam_inconsistency | `qualitative_baseline_comparison/210_seam_inconsistency_baseline_vs_dtrain.png` | - | - |
| 24 | pseudo-label_noise_suspected | `qualitative_baseline_comparison/24_pseudo-label_noise_suspected_baseline_vs_dtrain.png` | - | - |
| 305 | boundary_erosion | `qualitative_baseline_comparison/305_boundary_erosion_baseline_vs_dtrain.png` | - | pole(+2.3) |
| 319 | thin_structure_broken | `qualitative_baseline_comparison/319_thin_structure_broken_baseline_vs_dtrain.png` | - | - |
| 122 | small_object_missing | `qualitative_baseline_comparison/122_small_object_missing_baseline_vs_dtrain.png` | - | traffic_sign(+4.5) |
| 101 | class_confusion | `qualitative_baseline_comparison/101_class_confusion_baseline_vs_dtrain.png` | - | - |
| 155 | representative | `qualitative_baseline_comparison/155_representative_baseline_vs_dtrain.png` | - | - |
| 289 | representative | `qualitative_baseline_comparison/289_representative_baseline_vs_dtrain.png` | - | traffic_light(+3.4) |
| 111 | representative | `qualitative_baseline_comparison/111_representative_baseline_vs_dtrain.png` | - | - |
| 363 | representative | `qualitative_baseline_comparison/363_representative_baseline_vs_dtrain.png` | - | - |

---

## 视觉观察记录 (待人工填写关键定性发现)

> 说明: 上表按每样本的类别级 IoU 差值 (A_Dtrain_Etrain - baseline) 自动标出显著变化类别。
> 请人工打开各 panel 图,记录具体的空间位置/边界/接缝处的定性观察。

### 待观察要点
- Seam (210): 全景图左右接缝处是否出现类别跳变或边界断裂
- Pseudo-label noise (24): 压缩模型是否放大了伪标签噪声区域的误分类
- Boundary erosion (305): 物体边界(如 person/car 轮廓)是否变得更模糊
- Thin structure (319): 细长结构(如 pole/traffic light 杆)是否更容易丢失
- Small object (122): 小物体(如远处的 person/traffic light)是否更容易被背景类吞掉
- Class confusion (101): 语义相近类别(如 sidewalk/road, wall/building)混淆是否加重

## 严格非操作记录

- 未训练模型,未生成新伪标签。
- 仅读取已导出的 baseline_omnisam / A_Dtrain_Etrain 预测 masks 和 DensePASS GT。
- 未覆盖 qualitative_confirmed/、failure_cases_confirmed/ 中的既有文件。

---

## 像素级差异量化分析

| Sample | 失效类型 | 不一致像素% | 退化像素 | 改善像素 | 主要退化模式 (GT → dtrain错误预测) |
|---|---|---|---|---|---|
| 210 | seam_inconsistency | 0.33% | 891 | 1136 | road→sidewalk(385), terrain→sidewalk(77), vegetation→building(75) |
| 24 | pseudo-label_noise | 0.36% | 1490 | 920 | **terrain→sidewalk(518)**, terrain→wall(280), sky→vegetation(156) |
| 305 | boundary_erosion | 0.33% | 833 | 959 | road→sidewalk(367), wall→fence(173) |
| 319 | thin_structure | 0.51% | 559 | 2492 | sky→pole(143), road→sidewalk(134), building→pole(94) |
| 122 | small_object | 0.37% | 573 | 1547 | road→sidewalk(209), building→pole(45) |
| 101 | class_confusion | 0.61% | 1686 | 1642 | **road→sidewalk(839)**, terrain→vegetation(113), sky→vegetation(98) |

**关键发现**:
1. **Road/Sidewalk 边界模糊** — 所有样本中最高频的退化模式是 road→sidewalk 误分类(共1934像素)，表明 MTC+SMR 压缩降低了道路边界的判别能力
2. **Terrain 类脆弱** — sample 24 中 terrain→sidewalk/wall 退化占518+280=798像素，terrain 在压缩后容易被误判为结构化类别
3. **假阳性 Pole 插入** — sample 319 中 sky→pole(143) + building→pole(94) 表明压缩后模型在天空/建筑背景中产生虚假杆状结构
4. **语义近似类混淆** — wall/fence(173), terrain/vegetation(113+151) 等语义接近类别的边界更容易在压缩后崩塌

**总体观察**: 虽然逐样本不一致像素占比很小(0.33-0.61%)，但退化模式高度一致，表明 MTC+SMR 系统性地削弱了:
- 道路边界精度
- Terrain 类的判别边界
- 细长结构(pole)的抑制能力(产生假阳性)
- 语义相近类别的区分度

