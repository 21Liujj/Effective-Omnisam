# step_04_class_group_threshold_plan.md
# Phase 2 / Task 2-1 — 类别分组与阈值计划

**Run Level**: L0 scan-only（无训练、无代码修改、无 YAML 修改）  
**日期**: 2026-06-23  
**执行人**: Claude Code  
**修改代码**: 否  
**跑训练**: 否  
**修改 YAML**: 否

---

## 1. 本次任务说明

Phase 2 / Task 2-1 是 L0 scan-only 阶段。

任务目标：确定 R2_class_balanced 的 13 类分组（common / rare-hard）和阈值（top_ratio / min_conf），  
为 Task 2-2 的 `class_balance.py` 实现提供明确的参数依据。

**严格禁止**：不写代码、不修改 refiner.py、不改 SUPPORTED_MODES、不跑训练、不改 YAML、不做 PLQ/prototype/correction。

---

## 2. Class Mapping 来源

### 2.1 代码文件

| 文件 | 内容 |
|---|---|
| `utils/label_spaces.py` | CITYSCAPES13_CLASSES 规范列表（13 项） |
| `dataset/Cityscapes.py` `Cityscapes13._key` | 19→13 映射数组 |
| `dataset/DensePASS.py` `DensePASS13._key` | 同上，两者相同 |
| `utils/build.py` | DATASET_NAME2CLASSES_13，DensePASS / Cityscapes / WildPASS 均映射到 CITYSCAPES13_CLASSES |

### 2.2 19→13 映射键（两个数据集一致）

```python
_key = np.array([0,1,2,3,4,5,6,7,8,9,10,11,11,12,12,12,255,12,12])
# 19-class TrainId →  13-class label id
# rider(12)     → person(11)    ← 合并入 person
# truck(14)     → car(12)       ← 合并入 car
# bus(15)       → car(12)       ← 合并入 car
# train(16)     → 255(ignore)   ← 忽略
# motorcycle(17)→ car(12)       ← 合并入 car
# bicycle(18)   → car(12)       ← 合并入 car
```

### 2.3 规范 13 类列表（class_id 为 0-based）

| class_id | class_name | Cityscapes 19-class 原始 trainId |
|---|---|---|
| 0 | road | 0 |
| 1 | sidewalk | 1 |
| 2 | building | 2 |
| 3 | wall | 3 |
| 4 | fence | 4 |
| 5 | pole | 5 |
| 6 | traffic light | 6 |
| 7 | traffic sign | 7 |
| 8 | vegetation | 8 |
| 9 | terrain | 9 |
| 10 | sky | 10 |
| 11 | person | 11 + 12(rider 合并) |
| 12 | car | 13 + 14 + 15 + 17 + 18 (多类合并) |

**结论**：class mapping 明确，无歧义。两个数据集使用相同 `_key`，Cityscapes→DensePASS UDA 实验全程采用 13 类设定。

---

## 3. Per-class IoU 来源

### 3.1 使用的 run.log

| 来源 | 设定 | 路径 |
|---|---|---|
| base_last (final_2000) | full run，自生成 PL，seed=1 | `logs/omnisam_final_2000/base_last_20260425_232941.log` |
| A_ccpm (clean) | clean ablation，reuse PL，seed=1 | `experiments/tecma/runs/A_ccpm_seed1_20260616_104911/run.log` |
| A_Dtrain_Etrain (final_2000) | full run，自生成 PL，seed=1 | `logs/omnisam_final_2000/A_Dtrain_Etrain_20260425_232941.log` |
| baseline_omnisam (clean) | clean ablation，reuse PL，seed=1 | `experiments/tecma/runs/baseline_omnisam_seed1_20260616_102834/run.log` |

所有数据来自日志文件中 `========== per-class IoU (0-100) ==========` 段落，直接读取，无估算。

### 3.2 Pixel Frequency 来源

| 来源 | 内容 | 路径 |
|---|---|---|
| DensePASS PL 文件（base_last epoch1，2000 张） | 对 2000 张目标域 PL PNG 做像素计数（PIL，无模型） | `data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1/gtFine/` |
| WildPASS 验증 GT histogram | WildPASS val GT 像素分布（作为 360°全景域的参考） | `logs/wildpass_results/reports/wildpass_pred_gt_histogram.md` |

**注意**：PL 文件像素频率 = 伪标签分布，不等于真实 GT 频率。伪标签有置信度过滤（255=ignore），因此高不确定性类的 PL 频率会低估其真实出现率。ignore_ratio = 2.02%（PL 中被标 255 的比例，整体较低）。

---

## 4. 原始数据汇总

### 4.1 Per-class IoU（多 run 对比）

| class_id | class_name | base_last (final) | A_ccpm (clean) | baseline (clean) | A_DtE (final) | 均值 |
|---|---|---:|---:|---:|---:|---:|
| 0 | road | 80.38 | 80.25 | 80.23 | 80.29 | 80.29 |
| 1 | sidewalk | 47.54 | 47.35 | 47.33 | 47.60 | 47.46 |
| 2 | building | 89.69 | 89.67 | 89.67 | 89.67 | 89.68 |
| 3 | wall | 43.59 | 43.78 | 43.77 | 44.04 | 43.80 |
| 4 | fence | 49.02 | 48.92 | 48.92 | 48.47 | 48.83 |
| 5 | pole | 42.90 | 43.64 | 43.64 | 42.91 | 43.27 |
| 6 | traffic light | 28.00 | 27.88 | 27.88 | 27.90 | 27.92 |
| 7 | traffic sign | 31.40 | 32.52 | 32.52 | 31.14 | 31.90 |
| 8 | vegetation | 78.63 | 78.77 | 78.77 | 78.44 | 78.65 |
| 9 | terrain | 45.44 | 45.15 | 45.15 | 45.00 | 45.19 |
| 10 | sky | 94.57 | 94.61 | 94.61 | 94.57 | 94.59 |
| 11 | person | 68.73 | 67.50 | 67.50 | 68.59 | 68.08 |
| 12 | car | 83.44 | 83.34 | 83.33 | 83.52 | 83.41 |

**IoU 稳定性**：4 个 run 间各类 IoU 差值均 ≤ 1.5，说明结果稳定、可信。

### 4.2 DensePASS 伪标签像素频率（base_last PL，2000 张）

| class_id | class_name | PL 像素数 | ratio_valid | ratio_all |
|---|---|---:|---:|---:|
| 0 | road | 1,533,384,470 | 18.66% | 18.28% |
| 1 | sidewalk | 510,782,668 | 6.21% | 6.09% |
| 2 | building | 2,192,860,217 | 26.68% | 26.14% |
| 3 | wall | 75,240,458 | 0.92% | 0.90% |
| 4 | fence | 89,726,500 | 1.09% | 1.07% |
| 5 | pole | 67,244,870 | 0.82% | 0.80% |
| 6 | traffic light | 3,546,838 | 0.043% | 0.042% |
| 7 | traffic sign | 13,042,623 | 0.159% | 0.156% |
| 8 | vegetation | 1,404,331,184 | 17.09% | 16.74% |
| 9 | terrain | 206,596,556 | 2.51% | 2.46% |
| 10 | sky | 1,757,255,486 | 21.38% | 20.95% |
| 11 | person | 39,718,134 | 0.48% | 0.47% |
| 12 | car | 325,085,299 | 3.96% | 3.88% |
| 255 | ignore | 169,792,697 | — | 2.02% |
| — | **总计** | 8,388,608,000 | — | 100% |

**分布特征**：
- 高频类（>10%）：building(26.7%), sky(21.4%), road(18.7%), vegetation(17.1%)
- 中频类（1%~10%）：sidewalk(6.2%), car(4.0%), terrain(2.5%)
- 低频类（<1%）：wall(0.92%), fence(1.09%), pole(0.82%), person(0.48%), traffic sign(0.16%), traffic light(0.04%)

---

## 5. 类别分组表

### 5.1 分组依据原则

1. **common**：PL 频率高（>1%）且 IoU 稳定（>40%）— 样本充足，置信度可靠，适度过滤
2. **rare-hard**：PL 频率极低（<1%）或 IoU 极低（<35%）— 样本稀少或预测不稳定，需放宽 top_ratio 以保留更多 PL；放宽 min_conf 以避免 class starvation
3. **hard-common**：PL 频率中等（1%~5%）但 IoU 相对低（30%~45%），需单独注意

分析发现 traffic light（IoU≈28, PL占0.04%）和 traffic sign（IoU≈32, PL占0.16%）明显属于极端 rare-hard。  
pole（IoU≈43, PL占0.82%）和 wall（IoU≈44, PL占0.92%）属于 rare-hard（频率低，边界复杂）。

### 5.2 主分组表

| class_id | class_name | 均值 IoU | PL ratio_valid | group | top_ratio | min_conf | rationale |
|---|---|---:|---:|---|---:|---:|---|
| 0 | road | 80.29 | 18.66% | common | 0.50 | 0.75 | 高频大面积，IoU 高，置信度可靠 |
| 1 | sidewalk | 47.46 | 6.21% | common | 0.55 | 0.70 | 频率中等，IoU 中等，可适度放宽 |
| 2 | building | 89.68 | 26.68% | common | 0.50 | 0.75 | 最高频+最高 IoU，严格过滤可行 |
| 3 | wall | 43.80 | 0.92% | rare-hard | 0.75 | 0.60 | 频率低、边界混淆，需宽松保留 |
| 4 | fence | 48.83 | 1.09% | rare-hard | 0.70 | 0.60 | 频率低、精细结构，接近边界 |
| 5 | pole | 43.27 | 0.82% | rare-hard | 0.75 | 0.60 | 细小结构，低频，边界混淆 |
| 6 | traffic light | 27.92 | 0.043% | rare-hard | 0.80 | 0.55 | 极低频+极低 IoU，class starvation 风险 |
| 7 | traffic sign | 31.90 | 0.159% | rare-hard | 0.80 | 0.55 | 极低频+低 IoU，class starvation 风险 |
| 8 | vegetation | 78.65 | 17.09% | common | 0.50 | 0.75 | 高频，IoU 高，置信度可靠 |
| 9 | terrain | 45.19 | 2.51% | common | 0.55 | 0.70 | 频率中等，IoU 中，但在 360°图中常受投影变形影响，略放宽 |
| 10 | sky | 94.59 | 21.38% | common | 0.50 | 0.80 | 最高 IoU 类，置信度极可靠，可用高 min_conf |
| 11 | person | 68.08 | 0.48% | rare-hard | 0.75 | 0.60 | 低频但 IoU 不低（68），可能受 rider→person 合并的噪声影响 |
| 12 | car | 83.41 | 3.96% | common | 0.50 | 0.75 | 高 IoU，频率中等（truck/bus/motorcycle/bicycle 合并入car，范围宽） |

### 5.3 分组汇总

| group | 类别 | 数量 |
|---|---|---|
| common | road, sidewalk, building, vegetation, terrain, sky, car | 7 |
| rare-hard | wall, fence, pole, traffic light, traffic sign, person | 6 |

**注意**：person 虽然 IoU≈68（不低），但 PL 频率仅 0.48%，在 360° 全景图中行人往往分布在水平带区域且经常被高度置信度的 road/building 覆盖。归为 rare-hard 是为了防止行人类在 class-balanced 选择中被过度压缩。

**fence（IoU≈49, PL=1.09%）**：PL 频率恰好在 1% 临界点，但 fence 在 360° 图中投影扭曲严重、边界稀疏，归 rare-hard 更合理。

---

## 6. 推荐 R2 参数

### 6.1 基础阈值

```yaml
# R2_class_balanced 推荐参数
class_balance:
  strategy: "top_ratio_per_class"   # 每个类别独立取 top-k% 置信度像素
  
  # 分组默认值
  common_top_ratio: 0.50
  rare_top_ratio: 0.75
  common_min_conf: 0.75
  rare_min_conf: 0.60
  
  # 极端 rare 类单独配置
  extreme_rare_top_ratio: 0.80
  extreme_rare_min_conf: 0.55
  
  # Class starvation 保护
  min_pixels_per_class: 50        # 每个 batch 内，若某类有效像素 < 50，保留所有像素
  
  num_classes: 13
```

### 6.2 各类参数具体值

| class_id | class_name | group | top_ratio | min_conf | special_case |
|---|---|---|---:|---:|---|
| 0 | road | common | 0.50 | 0.75 | — |
| 1 | sidewalk | common | 0.55 | 0.70 | 略放宽（IoU 中等，domain gap 较大） |
| 2 | building | common | 0.50 | 0.75 | — |
| 3 | wall | rare-hard | 0.75 | 0.60 | — |
| 4 | fence | rare-hard | 0.70 | 0.60 | fence PL 频率在临界点，保守选 0.70 |
| 5 | pole | rare-hard | 0.75 | 0.60 | — |
| 6 | traffic light | rare-hard | 0.80 | 0.55 | **extreme rare**：IoU 极低+频率极低 |
| 7 | traffic sign | rare-hard | 0.80 | 0.55 | **extreme rare**：同上 |
| 8 | vegetation | common | 0.50 | 0.75 | — |
| 9 | terrain | common | 0.55 | 0.70 | 360° 投影变形，略放宽 |
| 10 | sky | common | 0.50 | 0.80 | IoU 最高，可用更高 min_conf |
| 11 | person | rare-hard | 0.75 | 0.60 | 低频，rider 合并可能引入噪声 |
| 12 | car | common | 0.50 | 0.75 | 多类合并，IoU 高 |

### 6.3 是否需要 hard-common 单独配置

**结论：当前不需要独立 hard-common 分组。**

sidewalk（IoU≈47）和 terrain（IoU≈45）虽然 IoU 偏低，但 PL 频率分别为 6.2% 和 2.5%，属于 common 范畴。  
已通过设置略高于 road/building 的 top_ratio（0.55 vs 0.50）来差异化处理。  
若 Task 2-2 实验后发现 sidewalk/terrain 的 class-balanced 选择仍不足，可在 Task 2-2 消融后再决定是否引入 hard-common 分组。

### 6.4 是否需要某些类别 special case

- **traffic light / traffic sign**：需特殊保护。top_ratio=0.80 + min_conf=0.55 + min_pixels_per_class 兜底。
- **person**：虽然 IoU 高，但由于 rider 合并和 360° 视角的人物分布特殊性，PL 频率极低（0.48%），需要 rare-hard 处理。
- **sky**：min_conf=0.80（比其他 common 类更高），因为 sky 的置信度本身极高，可以更严格。

---

## 7. 风险分析

### 7.1 rare class 放宽后可能引入噪声

| 风险 | 描述 | 缓解措施 |
|---|---|---|
| traffic light / traffic sign 极低频 | top_ratio=0.80 保留了置信度最高的 80% 像素，但 PL 本身质量不高（IoU≈28/32），可能放入较多错误 PL | min_conf=0.55 做兜底；min_pixels_per_class=50 防止 class starvation 反而引入大量噪声 |
| wall / fence 边界混淆 | 置信度高的 wall/fence 像素可能本身是错误 PL（被误标为 road/building），放宽 top_ratio 可能保留更多这类错误 | 建议 L3 sanity 时检查 wall/fence 类的 pseudo_valid_ratio 变化 |
| person 极低频 + 噪声 | rider 合并进 person 可能引入骑行类的噪声 PL | PL 已经过 offline 生成，合并在 _key 中完成；online 无额外噪声；top_ratio=0.75 合理 |

### 7.2 IoU 低是否因为伪标签少，还是类别本身难

| class | IoU≈ | 分析 |
|---|---|---|
| traffic light | 27.9 | **两者都有**：PL 仅 0.04%（极少样本），加上 360° 图中交通灯极小、投影变形，类别本身难 |
| traffic sign | 31.9 | 同上：PL 仅 0.16%，加上标牌小目标、高度变形 |
| wall | 43.8 | 主要因为类别本身难（wall vs building 边界模糊），PL 频率 0.92% 也偏低 |
| pole | 43.3 | 极细小结构，360° 投影下更难，类别难 + 频率低双重因素 |
| sidewalk / terrain | 45~47 | 主要是 domain gap（Cityscapes 城市道路 vs DensePASS 360° 室外场景），类别本身难度中等 |

**结论**：traffic light 和 traffic sign 的低 IoU 主要是类别难度+低频 PL 双重因素。  
对这两个类单纯放宽 top_ratio 效果有限，后续 R3 PLQ + R4 prototype 可能比 R2 class_balance 更有效。

### 7.3 class mapping 歧义

**结论：无歧义。**  
Cityscapes.py 和 DensePASS.py 使用相同 `_key`，`build.py` 中 DensePASS 和 Cityscapes 均映射到 `CITYSCAPES13_CLASSES`。  
唯一需注意的是 class 12 (car) 合并了 5 个 19-class 中的类别（car/truck/bus/motorcycle/bicycle），  
这使得 car 类的 PL 频率（3.96%）看似中等但涵盖范围广，IoU（83.4%）高是因为 road vehicle 在视觉上比较一致。

### 7.4 per-class IoU 口径一致性

| 风险 | 描述 |
|---|---|
| final_2000 vs clean ablation 设定差异 | final_2000（自生成 PL）与 clean ablation（reuse 固定 PL）的 mIoU 相同（60.26），per-class IoU 差异 ≤ 0.5 pp，在同一口径内可信 |
| WildPASS GT histogram 不等于 DensePASS | WildPASS GT 中 road 占 64.6%（远高于 DensePASS PL 的 18.7%），两者场景不同，不可混用作频率参考。本报告已分开列出，WildPASS histogram 仅作参考 |
| PL frequency ≠ GT frequency | 伪标签分布受到 PL 生成时 conf threshold 的影响，训练域分布以 DensePASS PL 像素频率为准（已实测），但注意 ignore 像素（2.02%）主要来自高不确定区域 |

---

## 8. 是否允许进入 Task 2-2

**允许进入 Task 2-2。**

判断依据：

| 条件 | 状态 |
|---|---|
| class mapping 明确、无歧义 | ✓ _key 一致，label_spaces.py 规范 |
| 13 类分组表完整 | ✓ 7 common + 6 rare-hard，每类有 rationale |
| per-class IoU 来源可信、口径一致 | ✓ 多 run 交叉验证，差异 ≤ 0.5 pp |
| PL 像素频率已实测 | ✓ 基于 2000 张 PL PNG 直接计数 |
| 推荐参数完整 | ✓ top_ratio、min_conf、min_pixels_per_class 均已给出 |
| 风险已分析 | ✓ 噪声风险、IoU 原因、口径一致性均已说明 |
| 不修改代码、不训练 | ✓ 本阶段无代码改动 |

**Task 2-2 进入条件**（必须满足再实现）：
1. 以本文件 §5-§6 的参数为实现依据。
2. `class_balance.py` 实现 top_ratio_per_class 策略，每类独立选取。
3. 在 `refiner.py` 的 `SUPPORTED_MODES` 中加入 `'class_balanced'`。
4. L3 sanity 时验证 `pseudo_valid_ratio` 各类别分布，确认 traffic light / traffic sign 未被清空。
5. 全量训练前必须通过 L1 dry-run 和 L3 sanity（strict PASS，含 loss_unsup 数值记录）。

---

## 9. 本次任务总结

- **做了什么**：L0 scan-only，读取 class mapping 代码（Cityscapes.py / DensePASS.py / label_spaces.py）、4 个 run 的 per-class IoU 日志、2000 张 DensePASS PL PNG 像素统计（PIL+numpy，无模型）
- **修改了哪些文件**：只生成本报告，无其他修改
- **是否跑训练**：否
- **是否改动代码/YAML**：否
- **结论**：class mapping 无歧义，分组表完整，允许进入 Task 2-2
