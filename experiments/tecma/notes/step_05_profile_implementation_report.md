# Task 2-2A 实现报告 — step_05

> 方案 A：把已有的 D/E token 统计追加写入 `train_timing.jsonl`。
> 不改训练逻辑、不改模型输出、不改 `sam/`、不实现方案 B。
> Generated: 2026-06-15

---

## 1. 修改了哪些文件

| 文件 | 改动 | 行数 |
|---|---|---|
| `adaptation/engine.py` | 在 `train_stats` 构建段（`memory_bank_size` 赋值之后、`avg_effective_patches_per_iter` 之前）插入 D/E 字段提取块 | +23 行（含注释），核心逻辑约 12 行 |

备份：`adaptation/engine.py.bak_task22a`（修改前原文件）。

**未改动**：`sam/`、`model/`、`adaptation/trainer.py`、`adaptation/strategies.py`、`adaptation/pl_generator.py`、`configs/`、`run_adaptation.py`。

插入位置（engine.py L545–567，紧跟原有 `memory_bank_size` 行）：

```python
self.last_train_stats["memory_bank_size"] = int(...)   # 原有行（未动）
# ── Task 2-2A: D/E token statistics (method A) ──
_mb = getattr(self.model_t, "memory_bank", None)
_cst = dict(getattr(_mb, "last_compression_stats", None) or {})
_d_active = bool(_cst) and not _cst.get("skipped", True)
self.last_train_stats["compression_tokens_before"] = float(_cst["original_num_tokens"])   if _d_active else None
self.last_train_stats["compression_tokens_after"]  = float(_cst["compressed_num_tokens"]) if _d_active else None
self.last_train_stats["compression_ratio"]         = float(_cst["compression_ratio"])     if _d_active else None
_e_active = (_mb is not None and bool(getattr(_mb, "sparse_enabled", False))
             and str(getattr(_mb, "sparse_mode", "none")).lower() not in ("none", ""))
self.last_train_stats["sparse_tokens_before"] = float(getattr(_mb, "last_sparse_tokens_before", 0.0)) if _e_active else None
self.last_train_stats["sparse_tokens_after"]  = float(getattr(_mb, "last_sparse_tokens_after",  0.0)) if _e_active else None
self.last_train_stats["sparse_keep_ratio"]    = float(getattr(_mb, "last_sparse_keep_ratio",    1.0)) if _e_active else None
self.last_train_stats["memory_tokens_scope"]        = "per_attention_call_all_chunks_concat"
self.last_train_stats["memory_tokens_before_final"] = float(self.last_train_stats.get("memory_tokens_before", 0.0))
self.last_train_stats["memory_tokens_after_final"]  = float(self.last_train_stats.get("memory_tokens_after",  0.0))
self.last_train_stats["avg_effective_patches_per_iter"] = float(...)  # 原有行（未动）
```

写入处 `json.dumps(self.last_train_stats)` → `train_timing.jsonl`（engine.py L598–600，未改动）。

---

## 2. 新增了哪些 train_timing 字段

共 9 个新字段（全部追加，无删除/改名）：

| 字段 | 类型 | 含义 |
|---|---|---|
| `compression_tokens_before` | float / null | D 压缩前单帧 token 数（最后一帧 per-chunk） |
| `compression_tokens_after` | float / null | D 压缩后单帧 token 数 |
| `compression_ratio` | float / null | D 压缩比 = after / before |
| `sparse_tokens_before` | float / null | E 检索前 token 数（最后一 iter 所有历史帧累加） |
| `sparse_tokens_after` | float / null | E 检索后 token 数（所有历史帧累加） |
| `sparse_keep_ratio` | float / null | E 实际保留比 = after / before |
| `memory_tokens_scope` | str | 固定字符串，标明 memory_tokens 口径 |
| `memory_tokens_before_final` | float | 复刻现有 `memory_tokens_before` 值（concat 全帧总数） |
| `memory_tokens_after_final` | float | 复刻现有 `memory_tokens_after` 值 |

`memory_tokens_scope` 取值：`"per_attention_call_all_chunks_concat"`
- 口径已确认：`memory_tokens_before/after` 来自 `memory_bank.py:445-446`，是 `torch.cat(to_cat_memory)` 之后送入单次 `memory_attention` 调用的全部 K/V token 总数（所有历史帧拼接），batch 维独立。**非 unconfirmed**。

---

## 3. 每个字段来自哪个对象/属性

| 字段 | 来源 |
|---|---|
| `compression_tokens_before` | `memory_bank.last_compression_stats["original_num_tokens"]` |
| `compression_tokens_after` | `memory_bank.last_compression_stats["compressed_num_tokens"]` |
| `compression_ratio` | `memory_bank.last_compression_stats["compression_ratio"]` |
| `sparse_tokens_before` | `memory_bank.last_sparse_tokens_before` |
| `sparse_tokens_after` | `memory_bank.last_sparse_tokens_after` |
| `sparse_keep_ratio` | `memory_bank.last_sparse_keep_ratio` |
| `memory_tokens_before_final` | 现有 `last_train_stats["memory_tokens_before"]`（trainer.py:276 → `last_memory_tokens_before`） |
| `memory_tokens_after_final` | 现有 `last_train_stats["memory_tokens_after"]`（trainer.py:277 → `last_memory_tokens_after`） |
| `memory_tokens_scope` | 硬编码字符串常量 |

**激活判定（避免误报）**：
- D（compression_*）：仅当 `last_compression_stats` 非空且 `skipped=False` 时写实值，否则 null。
  - D 关闭时，`last_compression_stats` 走 memory_bank.py:418 的 fallback，`skipped=True` → 写 null。
- E（sparse_*）：仅当 `sparse_enabled=True` 且 `sparse_mode != "none"` 时写实值，否则 null。
  - 用配置态判定（非 runtime token 数），避免 frame_idx=0 无历史帧时误判为关闭。

所有读取均为**只读** `getattr`，不修改任何 memory_bank 状态，不触发额外前向/同步。

---

## 4. baseline 时这些字段应是什么

配置 `baseline_omnisam`（A/D/E 全 off）：

```json
{
  "compression_tokens_before": null,
  "compression_tokens_after": null,
  "compression_ratio": null,
  "sparse_tokens_before": null,
  "sparse_tokens_after": null,
  "sparse_keep_ratio": null,
  "memory_tokens_scope": "per_attention_call_all_chunks_concat",
  "memory_tokens_before_final": 32768.0,
  "memory_tokens_after_final": 32768.0
}
```

D/E 字段全 null；final memory_tokens 保留（baseline 实测 32768，来自 `data_pd_exp_final_2000/base_last_*/train_timing.jsonl`）。

---

## 5. A_Dtrain_Etrain 时这些字段应是什么

配置 `A_Dtrain_Etrain`（A on, D train on, E train on）：

```json
{
  "compression_tokens_before": 4096.0,
  "compression_tokens_after": 1024.0,
  "compression_ratio": 0.25,
  "sparse_tokens_before": <实测>,
  "sparse_tokens_after": <实测>,
  "sparse_keep_ratio": ~0.5,
  "memory_tokens_scope": "per_attention_call_all_chunks_concat",
  "memory_tokens_before_final": 4096.0,
  "memory_tokens_after_final": 1024.0
}
```

说明：
- D 单帧 4096→1024（ratio 0.25），与 CLAUDE.md 记录的 pool kernel=2 stride=2 一致。
- `memory_tokens_after_final`（4096）已是 D 压缩后 concat 全帧的结果；现有最终实验 `A_Dtrain_Etrain_*/train_timing.jsonl` 中 `memory_tokens_after=4096`。
- E 的 `sparse_tokens_before/after` 具体数值**待实跑确认**（推测 keep_ratio≈0.5）。报告中标注"待实测"，不猜具体整数。
- 注意：`compression_*`（单帧 per-chunk）与 `memory_tokens_*_final`（全帧 concat）口径不同，论文中须分别说明。

---

## 6. 是否影响旧字段

**不影响**。
- 仅新增字段，未删除/改名任何旧字段。
- `memory_tokens_before` / `memory_tokens_after` / `memory_token_ratio` / `effective_memory_length` / `gpu_mem_peak` / `step_time` 等全部保留原值原名。
- 旧表格脚本读取旧字段不受影响（向后兼容）。
- `memory_tokens_before_final` / `after_final` 是旧字段的**副本**，不替代旧字段。

---

## 7. 是否跑了 sanity check

跑了，**未启动完整训练**（遵守限制）。三项检查：

1. **AST 解析**：`ast.parse(engine.py)` → `AST parse: OK`，无语法错误。
2. **字段提取逻辑单元测试**：抽取插入块逻辑，用 mock memory_bank 跑 3 种配置：
   - Case 1 baseline（D off, E off）：D/E 字段全 null，final=32768 ✅
   - Case 2 A_Dtrain_Etrain（D on, E on）：compression 4096→1024 ratio 0.25，sparse 4096→2048 ratio 0.5 ✅
   - Case 3 A_Dtrain（D on, E off）：compression 有值，sparse null ✅
   - JSON 序列化（含 null→`null`）round-trip 通过 ✅
   - 全部 assertions PASSED。
3. **未跑** 1-2 iter 真实训练：因真实训练需加载 SAM2 + LoRA + 数据集，启动成本高且按项目规则（feedback-no-script-execution）训练脚本由用户执行。单元级逻辑验证已覆盖字段取值与序列化正确性。

> 建议：用户下次跑任意一个 TeCMA train 实验时，自然会产生带新字段的 `train_timing.jsonl`，届时确认 `sparse_tokens_before/after` 实测值即可。

---

## 8. 是否建议进入 Task 2-3

**建议进入**，但有一条待确认项：

- ✅ 代码改动最小、向后兼容、逻辑经单元验证。
- ✅ 默认训练行为与模型输出不变（纯只读取属性 + dict 追加）。
- ⚠️ `sparse_tokens_before/after` 的真实整数值尚未由实跑确认（当前仅 mock 验证逻辑）。这不阻塞 Task 2-3，但 Phase 2 效率表最终定稿前应有一次真实 train 日志确认 E 的 token 数。
- ⚠️ `compression_*` 是"最后一帧 per-chunk"口径，若 Phase 5 token budget 曲线需要"逐帧平均压缩比"，可能需在后续 task 改为累计均值（当前不改，遵守最小改动）。

下一步（Task 2-3）建议：用户实跑一个短 train（或在已计划的 Phase 3 主实验中）验证新字段真实落盘，并据此校准效率表。
