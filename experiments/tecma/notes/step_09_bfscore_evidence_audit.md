# Task 3-3 BFScore 证据审计 — step_09_bfscore_evidence_audit.md

> 因 `metrics.jsonl` 中 BFScore 为 None，审计训练日志中 BFScore 的真实来源。
> 仅只读检查，未跑评测。
> Generated: 2026-06-15

---

## 审计方法

在两个正式实验对应的训练日志中 grep：`[ValSummary]`、`BFScore`、`boundary_f_score`。

日志位置：`logs/omnisam_final_2000/`

---

## 审计结果

| Config | BFScore in registry | Source File | Line / Context | Evidence Status |
|---|---|---|---|---|
| baseline_omnisam | 39.97 | `logs/omnisam_final_2000/base_last_20260425_232941.log` | L48: `[ValSummary] BFScore=39.97`（紧邻 L49 `Epoch 1: val mIoU=60.26`） | **verified_from_train_log** |
| A_Dtrain_Etrain | 39.86 | `logs/omnisam_final_2000/A_Dtrain_Etrain_20260425_232941.log` | L208: `[ValSummary] BFScore=39.86`（紧邻 L209 `Epoch 1: val mIoU=60.16`） | **verified_from_train_log** |

---

## 日志原文证据

**baseline（base_last_20260425_232941.log, L46-49）：**
```
===============================================
[ValSummary] FWDice=88.42
[ValSummary] BFScore=39.97
Epoch 1: val mIoU=60.26, mAcc=71.32, PixAcc=88.69, FWIoU=80.76, mDice=72.80, best=60.26; update_best=True
```

**A_Dtrain_Etrain（A_Dtrain_Etrain_20260425_232941.log, L206-209）：**
```
===============================================
[ValSummary] FWDice=88.38
[ValSummary] BFScore=39.86
Epoch 1: val mIoU=60.16, mAcc=71.42, PixAcc=88.63, FWIoU=80.71, mDice=72.72, best=60.16; update_best=True
```

---

## 结论

- 两个 config 的 BFScore **均找到明确日志证据**，数值与 registry 完全一致（39.97 / 39.86）。
- Evidence Status = **verified_from_train_log**（两者）。
- BFScore 确实被计算并打印到训练日志，只是当时未写入 `metrics.jsonl`（该字段为 None）。因此 registry 中的 39.97/39.86 **可追溯、可信**，不需要重新评测。

### 备注（非阻塞）

- 根因：`[ValSummary] BFScore` 走的是 print 到日志，而 `metrics.jsonl` 的 BFScore 字段未被回填。若后续希望 BFScore 自动进 `metrics.jsonl`，需在评测落盘处补一行（属代码改动，本任务不做）。
- 当前论文表引用 BFScore 已有日志可溯源，满足证据要求。
