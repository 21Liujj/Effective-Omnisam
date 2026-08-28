# step_02_identity_refiner_report.md
# Phase 1 / Task 1-1 -- Identity PseudoLabelRefiner Report

**Run Level**: L1 dry-run (DONE) + L3 sanity (configs ready, user must run)
**Date**: 2026-06-23
**Modified code**: Yes (strategies.py, run_adaptation.py)
**Ran training**: No (L1 only; L3 sanity configs prepared for user)

---

## 1. 本次做了什么

建立了 `PseudoLabelRefiner` 插件骨架，实现 `mode=off` 和 `mode=identity` 两种模式。
在 `adaptation/strategies.py` 的 `WithMemStrategy.forward()` 中插入了单一入口。
完成了 L1 干跑（全部 assertions PASS）。

**注意**：施工方案 Task 1-1 写的新文件路径是 `omnisam/pseudo_label/`，但实际 repo 不存在 `omnisam/` 目录。
按代码结构一致性原则改为 `adaptation/pseudo_label/`，与现有所有 adaptation 代码放在同一包下。

---

## 2. 修改了哪些文件

| 文件 | 类型 | 变更内容 |
|---|---|---|
| `adaptation/pseudo_label/__init__.py` | **新增** | 模块入口，export PseudoLabelRefiner + PseudoLabelRefineResult |
| `adaptation/pseudo_label/refiner.py` | **新增** | 核心类，实现 mode=off/identity |
| `adaptation/pseudo_label/plq.py` | **新增** | 占位桩，R3/R4/R5/R6 用 |
| `adaptation/pseudo_label/class_balance.py` | **新增** | 占位桩，R2 用 |
| `adaptation/pseudo_label/prototype_bank.py` | **新增** | 占位桩，R4/R5 用 |
| `adaptation/pseudo_label/correction.py` | **新增** | 占位桩，R5 用 |
| `adaptation/pseudo_label/losses.py` | **新增** | 占位桩，R3 用 |
| `adaptation/pseudo_label/diagnostics.py` | **新增** | 占位桩，L2 诊断用 |
| `adaptation/strategies.py` | **修改** | 在 mask_t 计算后插入 refiner 调用（约 line 199） |
| `run_adaptation.py` | **修改** | 在 merged_cfg 读取块中加入 pseudo_label_refiner 字段（约 line 1072） |
| `experiments/pl_refiner/configs/R0_baseline.yaml` | **新增** | mode=off sanity 配置，train_max_iters=5 |
| `experiments/pl_refiner/configs/R1_identity_sanity.yaml` | **新增** | mode=identity sanity 配置，train_max_iters=5 |
| `experiments/pl_refiner/scripts/sanity_identity_refiner.py` | **新增** | L1 断言脚本 |

---

## 3. 训练 loop 插入点

位置：`adaptation/strategies.py` `WithMemStrategy.forward()` 中，
在 `mask_t = visual_mask` 之后、`proto_bank.snapshot()` 之前（torch.no_grad() 块内）。

```python
# 原有代码：
else:
    mask_t = visual_mask

# ---- PseudoLabelRefiner 插入点 --------------------------------
_pl_cfg = getattr(args, 'pseudo_label_refiner', None) or {}
if str(_pl_cfg.get('mode', 'off')).lower() != 'off':
    from adaptation.pseudo_label import PseudoLabelRefiner
    _refiner = PseudoLabelRefiner.from_args(
        args, num_classes=int(pred_t.shape[1]), device=device,
    )
    _pl_result = _refiner.refine(
        pseudo_label=frame_pl_t,
        confidence=conf_t,
        student_logits=pred_t,
        features=feat_t_de,
        valid_mask=valid_mask,
        ignore_index=ignore_index,
        step=global_step,
    )
    if _pl_result is not None:
        frame_pl_t = _pl_result.refined_label
        mask_t     = _pl_result.valid_mask
# ---------------------------------------------------------------

# 原有代码继续：
proto_bank.snapshot()
```

---

## 4. mode=off 如何关闭 refiner

`mode='off'` 时，`str(_pl_cfg.get('mode', 'off')).lower() != 'off'` 为 False，
整个 if 块**不执行**。没有 import，没有对象创建，没有函数调用。

默认配置（不传 YAML 中的 `pseudo_label_refiner` 字段）时，
`getattr(args, 'pseudo_label_refiner', None)` 返回 None，
`_pl_cfg.get('mode', 'off')` 得到 'off'，同样短路。

**结论**：mode=off 对所有已有 baseline 的 CPU/GPU overhead = 0，行为完全不变。

---

## 5. mode=identity 如何保证等价

`_refine_identity()` 直接返回：
- `refined_label = pseudo_label`（原对象，不复制）
- `valid_mask = valid_mask`（原对象，不复制）
- `pl_weight = None`（无权重）

外部 caller 将 `frame_pl_t = _pl_result.refined_label` 和 `mask_t = _pl_result.valid_mask`，
因为都是原对象，后续 `pseudo_conf_mask` plugin（若启用）、`frame_pl_t_f` masking、`loss_unsup` 计算
完全与原逻辑一致。

**注意**：mode=identity 不绕过 `pseudo_conf_mask` plugin。如果 R0/R1 sanity 都禁用了
`pseudo_conf_mask`（如 R0_baseline.yaml 所设），则两者的 loss path 完全相同。

---

## 6. 是否跑训练

**L1 干跑已执行**（sanity_identity_refiner.py），全部 assertions PASS：
```
[L1] Import OK
[L1] mode=off PASS: returns None
[L1] mode=identity PASS: label/mask unchanged, diagnostics correct
[L1] ValueError raised correctly: [PseudoLabelRefiner] Unknown mode 'class_balanced'. Supported: ['identity', 'off']
[L1] from_args factory PASS
ALL L1 ASSERTIONS PASSED
```

**L3 short-run 未执行**。配置已就绪，需用户手动跑（每个只跑 5 步）：
```bash
cd /root/autodl-tmp/OmniSAM-main
# R0 (mode=off):
python run_adaptation.py   --config experiments/pl_refiner/configs/R0_baseline.yaml   [原有必要参数: --dataset Cityscapes --src_ckpt ... --tgt_ckpt ... 等]

# R1 (mode=identity):
python run_adaptation.py   --config experiments/pl_refiner/configs/R1_identity_sanity.yaml   [同上参数]
```

对比验证目标：前 5 步的 `loss_unsup` 和 `pseudo_valid_ratio` 数值应一致。

---

## 7. 是否改动 pseudo label

**否**。mode=off 完全不触碰 pseudo label。
mode=identity 返回原始 `frame_pl_t` 的引用（not clone），不修改内容。

---

## 8. 是否改动 loss

**否**。插入点在 `proto_bank.snapshot()` 之前，在任何 loss 计算之前。
mode=off/identity 均不改变 `loss_unsup` 的计算逻辑，也不引入新的 loss 项。

---

## 9. Sanity 对比（L3 待用户运行）

L3 sanity 目标：

| 指标 | R0 mode=off | R1 mode=identity | 预期 |
|---|---|---|---|
| loss_unsup (iter 1-5) | X | X | **完全相同** |
| pseudo_valid_ratio | X | X | **完全相同** |
| pseudo_mean_confidence | X | X | **完全相同** |
| step_time | X | X | 几乎相同（identity 无额外 GPU 计算） |

用户运行后请将结果填入此处。

---

## 10. 当前风险

| 风险 | 严重度 | 说明 |
|---|---|---|
| feat_t_de 在 no_grad 块内可访问 | 低 | Python 作用域确认 feat_t_de 已在外部赋值，no_grad 块内可读 |
| mode=identity 与 pseudo_conf_mask 组合 | 低 | identity 不干预 pseudo_conf_mask plugin；sanity yaml 已禁用 pseudo_conf_mask |
| 路径改为 adaptation/pseudo_label/ | 低 | 与施工方案 omnisam/ 不一致，已说明原因；不影响功能 |
| L3 sanity 缺少完整参数 YAML | 中 | R0/R1 sanity YAML 只有 overlay 字段，run_adaptation.py 需要用户提供完整参数（--dataset, --src_ckpt 等）；建议参考 experiments/tecma/configs/ 的脚本 |
| train_max_iters 字段是否被正确读取 | 中 | run_adaptation.py 读取 merged_cfg 时仅 overlay 已有 dataclass 字段，需确认 TrainConfig 是否有 train_max_iters 字段 |

---

## 11. 下一步是否可以进入 Task 1-2

**可以，但建议先完成 L3 sanity 验证**。

Task 1-2 是生成 R0-R7 配置和 dry-run 检查脚本，不需要训练，是 L1 级别工作，风险低。

**L3 sanity 等价性验证是 Task 1-1 的验收标准**。建议用户先运行：
```bash
python run_adaptation.py --config experiments/pl_refiner/configs/R0_baseline.yaml [args]
python run_adaptation.py --config experiments/pl_refiner/configs/R1_identity_sanity.yaml [args]
```
确认 loss_unsup 数值一致后，再进入 Task 1-2。
