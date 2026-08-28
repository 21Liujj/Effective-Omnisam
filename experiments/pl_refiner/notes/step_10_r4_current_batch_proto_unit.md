# step_10_r4_current_batch_proto_unit.md
# Phase 4 / Task 4-3a — current-batch prototype similarity 纯函数与单元测试

**Run Level**: L1 unit-test only
**日期**: 2026-06-24
**执行人**: Claude Code
**是否训练**: No
**是否运行 `run_adaptation.py`**: No
**是否接入 `refiner.py`**: No
**是否修改 `strategies.py`**: No
**是否修改 loss**: No
**是否实现 `proto_gate` mode**: No
**是否保存 checkpoint**: No
**是否 full-run**: No
**是否看 mIoU**: No
**是否进入 R4-3b / R4-4**: No

---

## 1. 本次任务说明

本次是 **Phase 4 / Task 4-3a**，目标是在不接入训练链路的前提下，实现 R4 v1 所需的 **current-batch prototype similarity** 纯函数，并用 synthetic tensor 做轻量单元测试。

本次只做：

```text
feat_t_de: (B, C, Hf, Wf)
pseudo_label / valid_mask: (B, H, W) or (B,1,H,W)
→ label/mask nearest downsample 到 feature resolution
→ current-batch class prototype
→ feature-location cosine similarity
→ summary diagnostics
```

本次不做训练、不运行 `run_adaptation.py`、不接入 `PseudoLabelRefiner`、不修改 `strategies.py`、不改 loss、不实现 `proto_gate` mode、不保存 checkpoint、不进入 R4-3b/R4-4。

---

## 2. 修改文件列表

| 文件 | 类型 | 内容 |
|---|---|---|
| `adaptation/pseudo_label/prototype_gate.py` | 新增 | R4 current-batch prototype similarity 纯函数 |
| `experiments/pl_refiner/scripts/test_current_batch_proto_gate_unit.py` | 新增 | synthetic tensor 单元测试脚本 |
| `experiments/pl_refiner/notes/step_10_r4_current_batch_proto_unit.md` | 新增 | 本报告 |

未修改：`adaptation/strategies.py`、`adaptation/pseudo_label/refiner.py`、loss、训练入口、checkpoint / metrics / mIoU 相关文件。

---

## 3. 实现函数列表

文件：`adaptation/pseudo_label/prototype_gate.py`

1. `downsample_label_and_mask_to_feature(pseudo_label, valid_mask, feature_shape, ignore_index=255)`
   - 输入 `(B,H,W)` 或 `(B,1,H,W)` label/mask。
   - 使用 nearest 将 label/mask downsample 到 `(Hf,Wf)`。
   - 输出 `label_f: (B,Hf,Wf) long`，`mask_f: (B,Hf,Wf) bool`。
   - `mask_f = mask_f & (label_f != ignore_index)`，保证 ignore pixel 不成为 valid。

2. `compute_current_batch_prototypes(feat, label_f, mask_f, num_classes=13, min_pixels_per_class=10, eps=1e-6)`
   - 输入 `feat: (B,C,Hf,Wf)`。
   - 每类使用 `label_f == k & mask_f` 的 feature locations。
   - `count >= min_pixels_per_class` 时生成 prototype。
   - `count < min_pixels_per_class` 时 `proto_valid[k]=False`，`prototype[k]=0`。
   - 输出 `prototypes: (num_classes,C)`、`proto_valid: (num_classes,) bool`、`proto_counts: (num_classes,) long`。

3. `compute_proto_similarity_map(feat, label_f, mask_f, prototypes, proto_valid, ignore_index=255, eps=1e-6)`
   - 对每个 feature location 取其 pseudo label 对应 prototype。
   - 计算 cosine similarity。
   - 仅 `mask_f=True` 且 `proto_valid[label]=True` 且 label 非 ignore 时 `sim_valid=True`。
   - missing prototype class 的 `sim_valid=False`，`sim_map=0`。
   - 输出 `sim_map: (B,Hf,Wf)`，`sim_valid: (B,Hf,Wf) bool`，无 NaN，范围 clamp 到 `[-1,1]`。

4. `summarize_proto_similarity(sim_map, sim_valid, label_f, num_classes=13, proto_valid=None, proto_counts=None)`
   - 输出 summary dict，包含 `sim_mean/sim_std/sim_min/sim_max/sim_valid_ratio/proto_valid_classes/proto_missing_classes/proto_counts_by_class/sim_mean_by_class/low_sim_ratio_at_0.2/0.4/0.6`。
   - 本阶段只 summary，不做 threshold filtering。

---

## 4. shape 对齐策略

采用 Task 4-2 推荐路线：

```text
feat_t_de 保持 feature resolution: (B,C,Hf,Wf)
pseudo_label / valid_mask nearest downsample 到 (B,Hf,Wf)
在 feature resolution 上计算 current-batch prototype 和 similarity_map
后续阶段再把 similarity_map upsample 回 high-res label/mask resolution
```

本次纯函数只实现到 feature-resolution similarity：`sim_map/sim_valid = (B,Hf,Wf)`。

原因：与现有 FPA `PrototypicalAdaptationEMA.update()` / `mse_pull_loss()` 的 label-to-feature 对齐方式一致；避免 upsample `(B,256,Hf,Wf)` feature 的显存开销；R4 当前只做 diagnosis/filtering 准备，不改变 CE loss。

---

## 5. prototype 计算策略

采用：

```text
先 L2 normalize 每个 feature location
→ 对同类 valid feature 求 mean
→ 再对 mean prototype 做 L2 normalize
```

实现上使用 `feat.detach()`，保证 prototype similarity 纯用于 mask/diagnosis，不引入梯度路径。

选择理由：先 normalize feature 再求 mean，更接近 cosine prototype 的方向平均；mean 后再次 normalize，保证 similarity 范围稳定。zero norm / NaN 使用 `torch.nan_to_num` 与 norm guard。

---

## 6. missing prototype fallback

函数层面：

```text
如果 class k 的 valid feature locations < min_pixels_per_class:
    proto_valid[k] = False
    prototypes[k] = 0
```

默认 `min_pixels_per_class = 10`。

similarity 层面：

```text
如果 proto_valid[label] == False:
    sim_valid = False
    sim_map = 0
```

后续接入 R4 filtering 时应采用 fallback keep R2：missing prototype class 不参与 prototype gate，final valid mask 对该类保持 R2 valid_mask。本阶段只输出 `sim_valid=False`，不做 threshold filtering。

---

## 7. 单元测试结果

运行命令：

```bash
cd /root/autodl-tmp/OmniSAM-main && \
/root/miniconda3/envs/omnisam/bin/python \
  experiments/pl_refiner/scripts/test_current_batch_proto_gate_unit.py
```

输出：

```text
PASS shape_normal
PASS b1hw_label
PASS missing_class_and_ignore
PASS all_false_valid_mask
ALL_CURRENT_BATCH_PROTO_GATE_UNIT_TESTS_PASSED
```

| 测试项 | 结果 | 说明 |
|---|---|---|
| shape 正常：`feat=(2,256,16,32)`, `pseudo_label=(2,64,128)`, `valid_mask=(2,64,128)` | PASS | downsample 后 shape 正确 |
| 支持 `pseudo_label=(B,1,H,W)` | PASS | label/mask squeeze 正常 |
| prototype 输出 shape | PASS | `prototypes=(13,256)`, `proto_valid=(13,)`, `proto_counts=(13,)` |
| similarity 输出 shape | PASS | `sim_map=(2,16,32)`, `sim_valid=(2,16,32)` |
| 无 NaN | PASS | normal / missing / all-false case 均无 NaN |
| similarity 范围 `[-1,1]` | PASS | `sim_map` clamp 并验证范围 |
| 少于 `min_proto_pixels` 的 class | PASS | `proto_valid[class]=False`，对应 `sim_valid=False` |
| `ignore_index=255` 不参与 prototype | PASS | ignore 位置 `sim_valid=False`，不计入 class count |
| `valid_mask` 全 False 极端情况 | PASS | 不崩溃，prototype 全 invalid，summary ratio 为 0 |

---

## 8. 是否允许进入 R4-3b

结论：

```text
YES：纯函数和单测通过，可以进入真实 batch similarity diagnosis。
```

说明：这里的 YES 指允许进入下一步真实 batch similarity diagnosis；不表示已经进入或应该立即执行 FPA proto_bank adapter。本次没有执行 R4-3b / R4-4。

---

## 9. 最终结论

**YES：纯函数和单测通过，可以进入真实 batch similarity diagnosis。**

建议下一步：仅做真实 batch similarity diagnosis；不要训练，不改 loss，不看 mIoU。
