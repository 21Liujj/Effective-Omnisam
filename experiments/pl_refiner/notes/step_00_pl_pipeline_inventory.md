# step_00_pl_pipeline_inventory.md
# Phase 0 / Task 0-1 — PL Pipeline 代码扫描报告

**Run Level**: L0 scan-only  
**日期**: 2026-06-23  
**执行人**: Claude Code  
**修改代码**: 否

---

## 1. 训练 / 评估 / PL 生成入口

| 功能 | 文件 | 入口函数 |
|---|---|---|
| UDA 训练主入口 | `run_adaptation.py` | `main()` → `Engine.setup()` + epoch loop |
| 训练逐 epoch | `adaptation/engine.py` | `Engine.train(epoch)` → `Trainer.train_one_epoch()` |
| 训练核心 forward+loss | `adaptation/strategies.py` | `WithMemStrategy.forward()` |
| PL 生成入口 | `adaptation/engine.py` | `Engine.generate_pseudo_labels(epoch)` |
| PL 生成核心 | `adaptation/pl_generator.py` | `PseudoLabelGenerator.run()` |
| 评估（UDA 内） | `adaptation/trainer.py` | `Trainer.validate()` |
| 评估独立入口 | `validation.py` | `main()` |

---

## 2. Pseudo Label 数据流

### 2.1 生成阶段（Offline）

`PseudoLabelGenerator.run()` 流程：

```
WildPASSforPL dataloader
  → 正向 pass: frame 0..N-1 → model(frame_idx, img) → logits_maps[:, 0..N-1]
  → 重置 memory
  → 反向 pass: frame 0..N-1 → logits_maps[:, N..2N-1]
  → merge_patch_sequence() / legacy merge
      → confidence threshold (uc_threshold=0.85) → uncertain → 255
  → 保存 PNG: {save_dir}/gtFine/{name}_labelTrainIds.png
  → [可选] save_sidecar → pseudo_meta/{name}.npz  (仅 saving.save_confidence=true)
```

PL 文件路径：`{pd_root}/DensePASS/{backbone}/{dataset}_{N}cls/epoch{E}/gtFine/`

### 2.2 训练时加载

- `WildPASSforAdaptation.__getitem__` 直接读 PL PNG（**不加载 sidecar**）
- 返回 `(image_patches, pseudo_label_patches, size, name, ori_label)`

### 2.3 训练时关键变量名（`strategies.py` `WithMemStrategy.forward`）

| 变量 | 含义 | Shape | 来源 |
|---|---|---|---|
| `frame_pl_t` | 从 PL 文件加载的伪标签 | `(B,H,W)` int64 | offline PL PNG |
| `pred_t` | 模型对目标域的 logits | `(B,C,H,W)` | online model forward |
| `conf_t` | `softmax(pred_t).max(1)` 置信度 | `(B,H,W)` float | online，每 iter 重算 |
| `pseudo_t` | `softmax(pred_t).max(1)` argmax | `(B,H,W)` | online（未用于 loss） |
| `valid_mask` | `frame_pl_t != 255` | `(B,H,W)` bool | offline PL |
| `visual_mask` | `(conf_t > conf_thr) & valid_mask` | `(B,H,W)` bool | online |
| `mask_t` | = `visual_mask`（无 text gate） | `(B,H,W)` bool | online |
| `frame_pl_t_f` | `frame_pl_t.clone(); [~mask_t]=255` | `(B,H,W)` | masked copy |
| `feat_t_de` | decoder 输出特征（含 memory attention） | `(B,256,H',W')` | online model forward |
| `feat_t` | backbone FPN features | list[3] | online |
| `ignore_index` | 255 | scalar | config |
| `conf_thr` | args.conf_thr（默认 0.85） | scalar | config |

**注意**：没有独立 teacher。`model_s` 仅用于 prototype 更新；`model`（目标模型）同时作为 student。

---

## 3. loss_unsup 数据流

```python
# strategies.py ~line 182-330
probs = torch.softmax(pred_t, 1)
conf_t, pseudo_t = probs.max(1)            # online confidence
valid_mask = (frame_pl_t != ignore_index)   # from offline PL file
visual_mask = (conf_t > conf_thr) & valid_mask
mask_t = visual_mask

frame_pl_t_f = frame_pl_t.clone()
frame_pl_t_f[~mask_t] = ignore_index
loss_unsup_raw = criteria['t'](pred_t, frame_pl_t_f)   # standard CE

if plugins.enabled("pseudo_conf_mask"):     # A 模块
    out = masked_unsup_ce(
        pred_logits=pred_t,
        pseudo_label=frame_pl_t,
        conf_map=conf_t,                    # online conf
        base_mask=mask_t,
        threshold=0.85,
    )
    loss_unsup = out.loss_masked            # sum(CE*keep) / valid_pixels
else:
    loss_unsup = loss_unsup_raw

loss_frame = (
    loss_sup_s
    + lambda_ssl * beta_scheduled * loss_unsup
    + lambda_fpa * alpha_scheduled * loss_proto
)
```

---

## 4. 可用 Feature 列表

| Feature | 变量 | Shape | 适合用途 |
|---|---|---|---|
| 最终 logits | `pred_t` | `(B,C,H,W)` | confidence、entropy、PLQ 计算 |
| Decoder 特征（memory-enhanced） | `feat_t_de` | `(B,256,H',W')` | prototype feature（最推荐） |
| 在线置信度 | `conf_t` | `(B,H,W)` | 直接使用 |
| Backbone FPN | `feat_t` | list[3] | 低层特征（用途较少） |
| 离线 sidecar 置信度 | N/A | `(H,W)` float16 | 当前训练 dataloader **未加载** |

`feat_t_de` 来自 `SegFormerHead_adaption.forward()` 返回值 `(x, _c)` 中的 `_c`，  
shape `(B, 256, H', W')`（H'≈H/4），已包含 memory attention，是 prototype feature 的最优选择。

---

## 5. Offline/Online PL 判断

**结论：混合模式**

| 组件 | 模式 |
|---|---|
| PL 文件（PNG）| Offline（epoch 前生成） |
| valid_mask | Online（基于离线 PL）|
| confidence `conf_t` | **Online**（每 iter 用当前模型重算） |
| pseudo_conf_mask 过滤 | Online（不修改离线文件） |
| sidecar confidence | Offline 可选（当前训练未使用） |

**PseudoLabelRefiner 应作为 online refiner 插入 strategies.py**，处理 `frame_pl_t`（离线加载）和 `conf_t`/`feat_t_de`（在线计算）。

---

## 6. R1-R7 是否需要重新生成 PL 文件

**结论：R1-R6 全部不需要重新生成离线 PL，R6 待确认。**

| 配置 | 需要重新生成 PL？ | 说明 |
|---|---|---|
| R0 baseline | 否 | 复用现有 |
| R1 conf_mask | 否 | 等价于现有 pseudo_conf_mask plugin（online） |
| R2 class_balanced | 否 | class-wise top-ratio 全部在线计算 |
| R3 plq_basic | 否 | PLQ = online conf × entropy × local_agreement |
| R4 proto_gate | 否 | prototype feature 在线计算，ignore 在线执行 |
| R5 rbpc | 否 | correction 对 frame_pl_t 副本在线执行 |
| R6 memory_plq | **待 Phase 6 扫描** | 需要 patch 坐标/overlap logits，当前不确定 |
| R7 full | 否 | R2-R5 组合，均 online |

---

## 7. 最小插入点

**位置**：`adaptation/strategies.py` → `WithMemStrategy.forward()`，  
在 `mask_t` 计算后、`loss_unsup` 计算前：

```python
# 现有约 line 183-199
probs = torch.softmax(pred_t, 1)
conf_t, pseudo_t = probs.max(1)
valid_mask = (frame_pl_t != ignore_index)
visual_mask = (conf_t > conf_thr) & valid_mask
mask_t = visual_mask

# ← PseudoLabelRefiner 插入点
# pl_result = pseudo_label_refiner.refine(
#     pseudo_label=frame_pl_t,
#     conf_map=conf_t,
#     pred_logits=pred_t,
#     feat=feat_t_de,          # 需在此之后提前计算（当前 feat_t_de 在 pred_t forward 时同时获得）
#     valid_mask=valid_mask,
#     global_step=global_step,
# )
# frame_pl_t = pl_result.refined_label
# mask_t     = pl_result.valid_mask
# plq_weight = pl_result.plq_weight

# 现有 frame_pl_t_f 处理继续...
frame_pl_t_f = frame_pl_t.clone()
frame_pl_t_f[~mask_t] = ignore_index
```

**注意**：`feat_t_de` 当前在 `pred_t, feat_t, feat_t_de = model(...)` 时已经获得（line ~177），  
在插入点时 `feat_t_de` 已经存在，**无需修改 model forward**。

---

## 8. 不确定项

| 项目 | 状态 |
|---|---|
| `psr_pld.train_filter` 是否在 strategies 中生效 | **未实现**：configs 有字段，strategies.py 不读取 |
| `WildPASSforAdaptation` 是否加载 sidecar | **当前不加载**（load_sidecar 未被调用） |
| R6 patch 坐标可用性 | 待 Phase 6 可行性扫描（Task 6-1）|
| `pseudo_t`（online）vs `frame_pl_t`（offline）差异量 | 未量化，epoch 内随模型更新有偏差 |

---

## 9. 本次任务总结

- **做了什么**：L0 代码扫描，阅读 engine.py / pl_generator.py / strategies.py / trainer.py / pseudo_conf_mask.py / model.py / pl_sidecar.py / pl_merge.py 等核心文件
- **修改了哪些文件**：无
- **生成了哪些输出**：本文件
- **是否跑训练**：否
- **是否改动 pseudo label**：否  
- **是否改动 loss**：否
- **当前风险**：无
- **下一步建议**：执行 Task 0-2（扫描已有实验结果），今日完成 Phase 0
