# step_13_r4_proto_gate_filter_unit.md
# Phase 4 / Task 4-4a — R4 class-wise percentile proto_gate filtering 纯函数与单元测试

**Run Level**: L1 unit-test only  
**日期**: 2026-06-24  
**执行人**: Claude Code  
**是否训练**: No  
**是否运行 `run_adaptation.py`**: No  
**是否接入 `refiner.py`**: No  
**是否修改 `strategies.py`**: No  
**是否修改 loss**: No  
**是否修改 pseudo_label**: No  
**是否实现 correction / memory consistency**: No  
**是否保存 checkpoint**: No  
**是否 full-run / 看 mIoU**: No  
**是否进入 R4-4b**: No

---

## 1. 本次任务说明

本次是 **Phase 4 / Task 4-4a**：实现 R4 class-wise percentile proto_gate filtering 的纯函数和 synthetic 单元测试。

本阶段只实现 feature-resolution 的 keep/filter mask 计算，以及 high-resolution mask 对齐工具；不接入 `PseudoLabelRefiner`，不改变训练行为。

R4 当前约束保持不变：

```text
R4 只做 filtering / valid_mask refinement
不修改 pseudo_label
不改 loss
不做 correction
不做 memory consistency
```

---

## 2. 修改文件列表

| 文件 | 类型 | 内容 |
|---|---|---|
| `adaptation/pseudo_label/prototype_gate.py` | 修改 | 新增 class-wise percentile proto_gate filtering 纯函数 |
| `experiments/pl_refiner/scripts/test_proto_gate_filter_unit.py` | 新增 | R4-4a synthetic unit test |
| `experiments/pl_refiner/notes/step_13_r4_proto_gate_filter_unit.md` | 新增 | 本报告 |

未修改：

- `adaptation/pseudo_label/refiner.py`
- `adaptation/strategies.py`
- loss 相关代码
- `run_adaptation.py`
- checkpoint / metrics / mIoU 相关文件

---

## 3. 实现函数列表

文件：`adaptation/pseudo_label/prototype_gate.py`

### 3.1 `compute_classwise_percentile_keep_mask(...)`

输入：

- `sim_map: (B,Hf,Wf)`
- `sim_valid: (B,Hf,Wf)` bool
- `label_f: (B,Hf,Wf)` long
- `proto_valid: (num_classes,)` bool
- class groups / percentile / fallback 参数

输出：

- `keep_f: (B,Hf,Wf)` bool
- `filter_f: (B,Hf,Wf)` bool
- `gate_diag: dict`

核心语义：

- `sim_valid=True` 的位置默认 keep。
- `sim_valid=False` 的位置 fallback keep：`keep_f=True, filter_f=False`。
- `filter_f` 只表示 R4 额外过滤掉的位置。
- 每类只在 class-level loop 中处理，不做 Python 像素级循环。
- missing prototype / insufficient pixels / extreme rare 均 fallback keep。

`gate_diag` 包含：

- `proto_gate_filter_ratio_total`
- `proto_gate_keep_ratio_total`
- `proto_gate_filter_ratio_common`
- `proto_gate_filter_ratio_rare`
- `proto_gate_filter_ratio_extreme_rare`
- `proto_gate_filtered_pixels_by_class`
- `proto_gate_kept_pixels_by_class`
- `proto_gate_total_pixels_by_class`
- `proto_gate_threshold_by_class`
- `proto_gate_fallback_classes`
- `proto_gate_fallback_reasons`
- `proto_gate_common_filter_percentile`
- `proto_gate_rare_filter_percentile`
- `proto_gate_extreme_rare_filter_percentile`

### 3.2 `upsample_keep_mask_to_label_resolution(...)`

输入：

- `keep_f: (B,Hf,Wf)` bool
- `output_shape: (H,W)`

输出：

- `keep_highres: (B,H,W)` bool

实现：nearest upsample；不使用 bilinear。

### 3.3 `apply_proto_gate_to_valid_mask(...)`

输入：

- `r2_valid_mask: (B,H,W)` bool
- `keep_highres: (B,H,W)` bool

输出：

- `final_valid_mask = r2_valid_mask & keep_highres`

说明：该函数只组合 mask，不修改 pseudo_label，不改变 loss。

---

## 4. R4 filtering 策略

本次实现 R4-3c 选定的 class-wise percentile 策略：

```text
common classes       bottom 10% filtered
rare classes         bottom 5% filtered
extreme rare classes bottom 0%, fallback keep
```

当前 13 类分组：

```text
common_classes = [0,1,2,8,9,10,12]
rare_classes = [3,4,5,11]
extreme_rare_classes = [6,7]
```

fallback keep 规则：

1. `proto_valid[k] == False`：该类不过滤，fallback keep。
2. class 在 `sim_valid` 中没有有效像素：跳过并记录 fallback。
3. class valid pixel 数 `< min_filter_pixels_per_class`：不过滤，fallback keep。
4. class 属于 extreme rare：不过滤，fallback keep。
5. 如果 percentile 过滤后 keep pixels `< min_keep_pixels_per_class`：不过滤，fallback keep。
6. `sim_valid == False` 的位置不由 R4 决定过滤，fallback keep。

---

## 5. 为什么不用 absolute threshold

R4-3c threshold calibration 显示：

- `threshold=0.6` 过滤比例太小，几乎不起作用；
- absolute threshold `0.85` 过滤过强且 class 间波动大；
- class-wise percentile filtering 更稳定：
  - common bottom 10%
  - rare bottom 5%
  - extreme rare 0%
  - 总过滤比例约 10%
  - rare 过滤约 4%–5%

因此 R4-4a 不实现 absolute threshold filtering，而先实现更稳的 class-wise percentile proto_gate 纯函数。

---

## 6. 单元测试结果

测试脚本：

```text
experiments/pl_refiner/scripts/test_proto_gate_filter_unit.py
```

运行命令：

```bash
cd /root/autodl-tmp/OmniSAM-main
/root/miniconda3/envs/omnisam/bin/python \
  experiments/pl_refiner/scripts/test_proto_gate_filter_unit.py
```

输出：

```text
PASS classwise_filter_core
PASS min_keep_fallback_and_apply_highres
ALL_PROTO_GATE_FILTER_UNIT_TESTS_PASSED
```

覆盖项：

| 测试项 | 结果 | 说明 |
|---|---|---|
| shape 正常：`sim_map=(2,16,32)`, `sim_valid=(2,16,32)`, `label_f=(2,16,32)` | PASS | 输出 `keep_f/filter_f` shape 正确 |
| common class bottom 10% 过滤 | PASS | class 0 过滤约 10% |
| rare class bottom 5% 过滤 | PASS | class 3 过滤约 5% |
| extreme rare class 不过滤 | PASS | class 6 fallback keep |
| `proto_valid=False` class 不过滤 | PASS | class 4 fallback keep |
| valid pixels 少于 `min_filter_pixels_per_class` 不过滤 | PASS | class 5 fallback keep |
| 过滤后 keep pixels 少于 `min_keep_pixels_per_class` fallback keep | PASS | class 1 fallback keep |
| `sim_valid=False` 位置不由 R4 过滤 | PASS | `filter_f=False`, `keep_f=True` |
| `final_valid_mask` shape 为 `(B,H,W)` | PASS | high-res `(2,64,128)` |
| `final_valid_mask <= r2_valid_mask` | PASS | AND 关系成立 |
| 不产生 NaN、不报错 | PASS | synthetic 测试通过 |
| `gate_diag` 包含必须字段 | PASS | required keys 全部存在 |

---

## 7. 是否允许进入 R4-4b

结论：

```text
YES：过滤纯函数和单测通过，可以接入 PseudoLabelRefiner 做真实 5-step diagnostic。
```

注意：本次没有执行 R4-4b。下一步如果进入 R4-4b，应只做真实 5-step diagnostic，继续避免 full-run / mIoU / checkpoint 正式结果污染。
