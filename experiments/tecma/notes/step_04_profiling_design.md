# Phase 2 Profiling Design — step_04

> 设计文档，不改代码。描述最小侵入式插桩方案。
> Generated: 2026-06-15

---

## 1. 需要统计的指标

| # | 指标 | 含义 | 粒度 |
|---|------|------|------|
| M1 | `Nq`：query token 数 | memory_attention 的 curr 序列长度 = H×W | 常量（4096/patch），可从 shape 导出 |
| M2 | `Nm_before`：memory K/V token 数（D/E 之前） | 进入 D/E 前的原始 token 总数 | per-iter |
| M3 | `Nm_after`：memory K/V token 数（D/E 之后） | 实际送入 memory_attention 的 token 数 | per-iter |
| M4 | `D_tokens_before`：D 压缩前单帧 token 数 | SpatialMemoryCompressor 输入 L | per-chunk（每帧） |
| M5 | `D_tokens_after`：D 压缩后单帧 token 数 | SpatialMemoryCompressor 输出 L' | per-chunk |
| M6 | `D_ratio`：D 压缩比 = L'/L | 空间压缩比率 | per-chunk |
| M7 | `E_tokens_before`：E 检索前单帧 token 数 | SparseMemoryRetriever 输入 L | per-chunk |
| M8 | `E_tokens_after`：E 检索后单帧 token 数 | top-k 选取后 token 数 | per-chunk |
| M9 | `E_ratio`：E 实际 keep ratio | E_after / E_before | per-chunk |
| M10 | `memory_attention_time_ms`：memory attention 耗时 | CUDA 同步后的 wall time | per-iter |
| M11 | `iter_total_time_s`：每 iter 总耗时 | t_iter end - t_iter start | per-iter |
| M12 | `iter_compute_time_s`：每 iter 纯计算耗时 | strategy.forward() 耗时 | per-iter |
| M13 | `data_time_s`：每 iter 数据加载耗时 | dataloader next() 耗时 | per-iter |
| M14 | `gpu_mem_peak_gb`：GPU 显存峰值 | torch.cuda.max_memory_allocated() | per-epoch |
| M15 | `encoder_time_ms`：encoder 前向耗时 | SAM2 image encoder（可选） | per-iter（可选） |
| M16 | `decoder_time_ms`：decoder 前向耗时 | SegFormerHead（可选） | per-iter（可选） |
| M17 | `loss_time_ms`：loss 计算耗时（可选） | loss_sup + loss_unsup + loss_fpa | per-iter（可选） |

**必须统计**：M1–M14（论文主表和效率表）  
**可选统计**：M15–M17（需额外插桩，用于更细粒度的 profiling 附录）

---

## 2. 对应代码位置

### 2.1 已有统计（零代价可读）

代码中**已经计算并存储**以下值，无需新增 time 调用：

| 属性 / 变量 | 代码位置 | 对应指标 |
|---|---|---|
| `memory_bank.last_memory_tokens_before` | `memory_bank.py:445` | M2 (epoch last-iter 值) |
| `memory_bank.last_memory_tokens_after` | `memory_bank.py:446` | M3 (epoch last-iter 值) |
| `memory_bank.last_memory_token_ratio` | `memory_bank.py:447` | M3/M2 |
| `memory_bank.last_attention_time` | `memory_bank.py:460` | M10 (per-iter, in-memory) |
| `memory_bank.attention_time_accum` | `memory_bank.py:461` | M10 epoch 累计 |
| `memory_bank.last_compression_stats` | `memory_bank.py:365,475` | M4–M6（dict，但未入 JSONL） |
| `memory_bank.last_sparse_stats` | `memory_bank.py:404` | M7–M9（dict，但未入 JSONL） |
| `memory_bank.last_sparse_tokens_before` | `memory_bank.py:383` | M7 |
| `memory_bank.last_sparse_tokens_after` | `memory_bank.py:399` | M8 |
| `trainer.last_epoch_stats["step_time"]` | `trainer.py:264` | M11 epoch 均值 |
| `trainer.last_epoch_stats["iter_compute_time"]` | `trainer.py:265` | M12 epoch 均值 |
| `trainer.last_epoch_stats["data_time"]` | `trainer.py:265` | M13 epoch 均值 |
| `train_timing.jsonl["gpu_mem_peak"]` | `engine.py:532` | M14 per-epoch，已入 JSONL |
| `train_timing.jsonl["memory_attention_time"]` | `engine.py:540` | M10 epoch 累计，已入 JSONL |
| `train_timing.jsonl["memory_tokens_before/after"]` | `engine.py:276-278` | M2/M3 last-iter 值，已入 JSONL |
| `strategies.py extra_metrics["memory_attention_time_ms"]` | `strategies.py:571` | M10 per-iter，在 `last_iter_metrics` 中 |
| `strategies.py extra_metrics["sparse_tokens_before/after"]` | `strategies.py:576-578` | M7/M8 per-iter，在 `last_iter_metrics` 中 |

**结论**：M1–M3、M10–M14 在现有日志中已有 epoch 级别数据。`strategies.py` 的 `extra_metrics` 已经包含 per-iter 的 M7–M10，但 `trainer.py` 只提取了 text-fusion 相关字段，M7–M10 per-iter 数据**未写入任何 JSONL**。

### 2.2 缺口（需新增）

| 缺口 | 缺什么 | 插桩位置 |
|---|---|---|
| G1 | D/E 统计未入 epoch-level JSONL | `engine.py` train_stats 构建段（L527-545） |
| G2 | per-iter M7–M10 未写入 JSONL | `trainer.py` L207-214（提取 iter_metrics 的 for 循环） |
| G3 | per-iter 完整 profiling JSONL（按 log_interval 写入） | `trainer.py` + 新增 `profiling_step.jsonl` |
| G4 | Nq 未显式记录（可静态导出，不需要 runtime） | 一次性计算 = patch_size / encoder_stride^2 = 4096 |
| G5 | M15–M17（encoder/decoder/loss breakdown） | `strategies.py` forward() 内各子步前后加计时 |

---

## 3. 推荐插桩方案

### 原则

- **默认无开销**：所有新增统计都在 `args.profile_mode` 为 `True` 时才激活写入，或复用已有 attribute 读取
- **不影响梯度**：所有 `time.perf_counter()` 和 `torch.cuda.synchronize()` 调用必须在 `with torch.no_grad()` 外，且只在 profile 模式下触发 sync
- **不修改 `sam/`**：memory_attention 的内部不插桩，只在 `MemoryBank.forward()` 外侧计时（已有）
- **零 schema 变更**：新字段直接追加到现有 `train_timing.jsonl`，不新建表结构

### 方案 A（最小改动，推荐）：仅补全 epoch-level JSONL

在 `engine.py` 的 train_stats 构建段中追加读取 `last_compression_stats` 和 `last_sparse_stats`：

```python
# engine.py L527 附近（train_stats 构建）
mb = getattr(self.model_t, "memory_bank", None)
if mb is not None:
    cst = getattr(mb, "last_compression_stats", {}) or {}
    train_stats["d_tokens_before"]  = float(cst.get("original_num_tokens",   0))
    train_stats["d_tokens_after"]   = float(cst.get("compressed_num_tokens", 0))
    train_stats["d_compress_ratio"] = float(cst.get("compression_ratio",     1.0))
    sst = getattr(mb, "last_sparse_stats", {}) or {}
    train_stats["e_tokens_before"]  = float(mb.last_sparse_tokens_before)
    train_stats["e_tokens_after"]   = float(mb.last_sparse_tokens_after)
    train_stats["e_keep_ratio"]     = float(mb.last_sparse_keep_ratio)
```

**改动量**：`engine.py` 约 7 行，无逻辑变更。

### 方案 B（中等改动）：增加 per-iter profiling JSONL

在 `trainer.py` 的 iter_metrics 提取段追加 memory/timing 字段，并在 profile 模式下写入 `profiling_step.jsonl`：

```python
# trainer.py L207 附近，在 iter_metrics 提取 loop 后追加
if isinstance(iter_metrics, dict) and do_measure:
    prof_keys = (
        "memory_tokens_before", "memory_tokens_after", "memory_token_ratio",
        "memory_attention_time_ms", "memory_attention_time_avg",
        "sparse_tokens_before", "sparse_tokens_after", "sparse_keep_ratio",
        "effective_memory_length",
    )
    prof_rec = {
        "epoch": int(epoch + 1), "iter": int(it + 1),
        "global_step": int(global_step),
        "iter_total_s": float(time.perf_counter() - t_iter),  # ← t_iter 已有
        "iter_compute_s": float(comp_t),                      # ← comp_t 已有
        "data_s": float(data_durations[-1]) if data_durations else 0.0,
    }
    for k in prof_keys:
        v = iter_metrics.get(k)
        if v is not None:
            prof_rec[k] = float(v)
    self.last_step_profile_records.append(prof_rec)  # 写入 buffer
```

写入时机：epoch 结束时批量 flush 到 `profiling_step.jsonl`（避免每 iter IO）。

**改动量**：`trainer.py` 约 20 行（新增 buffer 属性 + 追加逻辑 + flush）。

### 方案 C（可选，仅用于 profiling 论文附录）：M15–M17 encoder/decoder/loss 拆分

在 `strategies.py` 的 `WithMemStrategy.forward()` 内：

```python
# model forward 前后加 CUDA sync + perf_counter
# 插桩位置：strategies.py L165-185（model(frame_img_t, ...) 调用）
if do_profile:
    torch.cuda.synchronize()
    t_enc0 = time.perf_counter()
pred_t, aux_t = model(frame_img_t, frame_idx, ...)
if do_profile:
    torch.cuda.synchronize()
    enc_time_ms = (time.perf_counter() - t_enc0) * 1000.0
```

注意：此处 `model.forward()` 包含 encoder + memory_bank + decoder，无法直接分拆，除非在 `model/model.py` 的 `OmniSAM_adapt.forward()` 内部添加更细粒度的计时。由于 `model/model.py` 不在禁改范围，可选添加。

**改动量**：`model/model.py` ~15 行，`strategies.py` 传参 ~5 行。建议延后到确认需要才实现。

---

## 4. 输出日志格式

### 4.1 现有 `train_timing.jsonl`（epoch 级，方案 A 后字段）

```json
{
  "epoch": 1,
  "step_time": 4.08,
  "iter_compute_time": 2.99,
  "data_time": 0.17,
  "epoch_train_time": 1496.8,
  "gpu_mem_peak": 25.34,
  "memory_attention_time": 93.6,
  "memory_attention_time_per_call": 0.0117,
  "memory_tokens_before": 4096.0,
  "memory_tokens_after": 4096.0,
  "memory_token_ratio": 1.0,
  "d_tokens_before": 4096.0,
  "d_tokens_after": 1024.0,
  "d_compress_ratio": 0.25,
  "e_tokens_before": 4096.0,
  "e_tokens_after": 2048.0,
  "e_keep_ratio": 0.5,
  "compressor_mode": "pool",
  "compress_location": "before_attention",
  "effective_memory_length": 4096.0,
  "memory_bank_size": 9,
  "dataset": "Cityscapes",
  "seed": 1
}
```

注意：当前 `memory_tokens_before/after` 反映的是**所有历史帧累积后 concat 的总 token 数**；`d_tokens_before/after` 反映的是**最后一帧的单帧 D 压缩结果**（`last_compression_stats`）。两者含义不同，文档/论文中必须区分。

### 4.2 新增 `profiling_step.jsonl`（per-iter，方案 B 后）

```json
{"epoch":1,"iter":1,"global_step":1,"iter_total_s":4.12,"iter_compute_s":3.01,"data_s":0.16,
 "memory_tokens_before":4096.0,"memory_tokens_after":1024.0,"memory_token_ratio":0.25,
 "memory_attention_time_ms":11.7,"sparse_tokens_before":1024.0,"sparse_tokens_after":512.0,
 "sparse_keep_ratio":0.5,"effective_memory_length":1024.0}
{"epoch":1,"iter":2,"global_step":2,...}
```

每 iter 一行，便于后处理绘制随 iter 变化的曲线。

### 4.3 现有 `pl_timing.jsonl`（PL 阶段，无需改动）

已包含：`t_forward`, `t_backward`, `mem_attn_time`, `raw_mem_tokens`, `compressed_mem_tokens`, `mem_token_ratio_acc`。

---

## 5. 风险与规避

| 风险 | 原因 | 规避 |
|---|---|---|
| CUDA sync 拖慢训练 | 每 iter 多一次 `torch.cuda.synchronize()` | memory_attention 外的 sync 已存在（`_sync_cuda`），不额外增加；方案 B 仅 buffer，不 sync |
| IO 开销 | 每 iter 写磁盘 | 方案 B 使用 buffer，epoch 结束后 batch flush |
| `last_compression_stats` 是最后一 chunk 的 stats | 若有多帧，只保留最后一帧 | 文档注明，论文中用"per-chunk 平均"而非"epoch 末值"；或改为累计均值（下个 task 决定） |
| `--profile_mode` 不在 AdaptConfig 中 | 需要添加新字段 | 只用 `getattr(args, "profile_mode", False)`，无需 dataclass 注册也能工作；或直接用现有 `verbose` 参数 |
| 在 PL 阶段 D/E 关闭时 stats 为空 | `last_compression_stats` 可能是 `{}` | 已有 fallback 默认值（memory_bank.py L464-474），安全 |
| 修改 `strategies.py` 可能影响所有实验 | 该文件被所有配置共用 | 方案 A/B 不改 strategies.py，方案 C 延后 |

---

## 6. 最小改动文件列表

### 方案 A（推荐先实施，约 7 行）

| 文件 | 位置 | 改动内容 |
|---|---|---|
| `adaptation/engine.py` | L527–545（train_stats 构建） | 追加 D/E stats 字段到 train_stats dict |

### 方案 B（推荐随后实施，约 20 行）

| 文件 | 位置 | 改动内容 |
|---|---|---|
| `adaptation/trainer.py` | `__init__` + `train_one_epoch` L207–214 + epoch 末 | 新增 `last_step_profile_records` buffer，追加 prof_rec，epoch 末 flush 到 `profiling_step.jsonl` |

### 方案 C（可选，约 20 行，延后）

| 文件 | 位置 | 改动内容 |
|---|---|---|
| `model/model.py` | `OmniSAM_adapt.forward()` | 在 encoder / memory_bank / decoder 调用前后加 `time.perf_counter()`，存入 `self.last_forward_timing` |
| `adaptation/strategies.py` | `WithMemStrategy.forward()` forward 调用后 | 读取 `model.last_forward_timing`，追加到 extra_metrics |

**完全不改动的文件**：
- `sam/` 目录（SAM2 原始代码）
- `configs/` 目录
- `adaptation/pl_generator.py`（PL 插桩已完整）
- `adaptation/engine.py` PL 段
- `dataset/`
- `run_adaptation.py`

---

## 附：当前已可读取的 profiling 数据

无需新增任何代码，直接从**已有 JSONL 日志**中可得到以下论文所需数据：

```
train_timing.jsonl:
  step_time              ← M11（epoch 均值）
  iter_compute_time      ← M12（epoch 均值）
  data_time              ← M13（epoch 均值）
  memory_attention_time  ← M10（epoch 累计）
  memory_attention_time_per_call ← M10（epoch 均值/call）
  gpu_mem_peak           ← M14（epoch peak）
  memory_tokens_before   ← M2（last-iter concat total）
  memory_tokens_after    ← M3（last-iter concat total）
  effective_memory_length ← Nm（last-iter）
  compressor_mode / compress_location ← D 模块状态

pl_timing.jsonl:
  mem_attn_time          ← M10 in PL phase
  raw_mem_tokens / compressed_mem_tokens ← M2/M3 in PL
```

**论文主表和效率表的所有必须数据都可从已有 JSONL 中读取**，无需等待新插桩跑实验。

方案 A/B 的新增插桩用于：per-iter 曲线图、D 单帧压缩比精确数据、E keep_ratio 精确数据。
