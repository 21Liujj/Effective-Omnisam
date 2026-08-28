# Task 2-3 实现报告 — step_06 真实短程 sanity check

> 用极短真实训练（8 张子集、2 train iter、1 epoch）确认 Task 2-2A 新增字段在真实 forward 中正确落盘。
> 不跑完整实验、不覆盖主结果、不改模型逻辑、不做 per-iter profiling。
> Generated: 2026-06-15

---

## 0. 前置：config check

运行 `/root/miniconda3/envs/omnisam/bin/python experiments/tecma/scripts/check_configs.py`：
- 9 configs / 0 validation errors
- 26 acceptance checks / 0 failed
- `[PASS] All checks passed.`

---

## 1. 实际运行命令

两个配置各跑一次 `run_adaptation.py`，参数极小：`--num_epochs 1 --batch_size 4 --warmup_pl 8 --iters_pl 8 --train_max_iters 2`，PL 子集 8 张（`sanity_runs/subset8.txt`，取自 `dataset/densepass_list/train.txt` 前 8 行）。

**baseline_omnisam：**
```bash
CUDA_VISIBLE_DEVICES=0 python run_adaptation.py \
  --config experiments/tecma/configs/baseline_omnisam.yaml \
  --seed 1 --dataset Cityscapes --class_set 13 --backbone sam2_s \
  --data_root data --pd_root experiments/tecma/sanity_runs/task23_baseline_20260615_sanity \
  --src_ckpt /root/autodl-tmp/omnisam_s.pth --tgt_ckpt /root/autodl-tmp/omnisam_s.pth \
  --amp --tf32 --num_epochs 1 --batch_size 4 \
  --pl_batch_size 1 --pl_num_workers 2 --no-pl_shuffle \
  --pl_subset_file experiments/tecma/sanity_runs/subset8.txt \
  --warmup_pl 8 --iters_pl 8 --train_max_iters 2 \
  --no-use_text_proto --lambda_txt 0.0 --weight_txt_align 0.0 \
  --no-use_text_gate_ssl --no-use_wandb --no-use_tensorboard
```

**A_Dtrain_Etrain：** 同上，仅替换
`--config experiments/tecma/configs/A_Dtrain_Etrain.yaml`
`--pd_root experiments/tecma/sanity_runs/task23_A_Dtrain_Etrain_20260615_sanity`

> 说明：`train_timing.jsonl` 在 `train_one_epoch` 末尾、`validate()` 之前写出。拿到该文件后，A_Dtrain_Etrain 的 100-图 validation 被提前终止（数据已落盘，无需等完整 val）。这是为省时，不影响被检字段。

---

## 2. 输出目录

| 配置 | pd_root（隔离的 sanity 目录） |
|---|---|
| baseline | `experiments/tecma/sanity_runs/task23_baseline_20260615_sanity/` |
| A_Dtrain_Etrain | `experiments/tecma/sanity_runs/task23_A_Dtrain_Etrain_20260615_sanity/` |

train_timing 路径：`<pd_root>/checkpoints/Cityscapes_13cls/sam2_s/metrics/train_timing.jsonl`

**未触碰** `data_pd_exp_final_2000/`（主结果）及任何既有完整实验。

---

## 3. baseline 的 train_timing 字段样例

```
step_time                  = 3.5827564019709826
iter_compute_time          = 2.092226443812251
memory_attention_time      = 0.4853552710264921
memory_tokens_before       = 32768.0
memory_tokens_after        = 32768.0
gpu_mem_peak               = 19.540457248687744
--- 新增字段 ---
compression_tokens_before  = null
compression_tokens_after   = null
compression_ratio          = null
sparse_tokens_before       = null
sparse_tokens_after        = null
sparse_keep_ratio          = null
memory_tokens_scope        = "per_attention_call_all_chunks_concat"
memory_tokens_before_final = 32768.0
memory_tokens_after_final  = 32768.0
```

验收：
- D 字段全 null ✅
- E 字段全 null ✅
- `memory_tokens_before/after_final` 有值（32768） ✅
- `memory_tokens_scope` 明确字符串 ✅
- 旧字段（step_time / iter_compute_time / memory_attention_time / memory_tokens_* / gpu_mem_peak）齐全 ✅

---

## 4. A_Dtrain_Etrain 的 train_timing 字段样例

```
step_time                  = 3.5493572717532516
iter_compute_time          = 1.8963799411430955
memory_attention_time      = 0.21699152700603008
memory_tokens_before       = 4096.0
memory_tokens_after        = 4096.0
gpu_mem_peak               = 18.77684450149536
--- 新增字段 ---
compression_tokens_before  = 4096.0
compression_tokens_after   = 1024.0
compression_ratio          = 0.25
sparse_tokens_before       = 8192.0
sparse_tokens_after        = 4096.0
sparse_keep_ratio          = 0.5
memory_tokens_scope        = "per_attention_call_all_chunks_concat"
memory_tokens_before_final = 4096.0
memory_tokens_after_final  = 4096.0
```

验收：
- D：compression 4096→1024，ratio 0.25（pool kernel=2 stride=2，与设计一致）✅
- E：sparse 8192→4096，keep_ratio 0.5 ✅ —— **真实值已确认**（之前 Task 2-2A 仅 mock 验证，本次实跑落实）
- `memory_tokens_before/after_final` 有值（4096，已是 D 压缩后 concat 结果）✅
- `memory_tokens_scope` = `per_attention_call_all_chunks_concat` ✅

口径说明（重要）：
- `compression_*`：单帧 per-chunk（最后一帧），4096→1024。
- `sparse_tokens_before` = 8192：E 在 forward 内对所有历史 memory chunk **累加**（`+=`），本帧有 2 个有效历史 chunk × 4096 = 8192；E 取 top-k 后 4096，keep_ratio 0.5。
- `memory_tokens_*_final` = 4096：现有口径，单次 memory_attention 调用 concat 全部 chunk 的 K/V 总数。
- 三者口径不同，论文效率表中须分别标注，不可混用。

---

## 5. PL 阶段是否确认 D/E 关闭

检查两配置的 `pl_timing.jsonl`：

| 字段 | baseline | A_Dtrain_Etrain |
|---|---|---|
| `compressor_mode` | none | pool（**仅配置态**） |
| `memory_compression_ratio` | 1.0 | **1.0** |
| `raw_memory_tokens` | 19797.3 | 19797.3 |
| `compressed_memory_tokens` | 19797.3 | **19797.3（= raw，无压缩）** |
| `sparse_patch_count` | 0 | **0** |

结论：**PL 阶段 D/E 未生效**。
- A_Dtrain_Etrain 的 `compressor_mode=pool` 只是反映模块被配置，但 `memory_compression_ratio=1.0` 且 `raw==compressed`（19797.3=19797.3），证明 PL forward 中**未实际压缩**；`sparse_patch_count=0` 证明 E 未选择。
- config 确认：`A_Dtrain_Etrain.yaml` 中 D 与 E 均 `apply_in_pl: false` / `apply_in_train: true`。
- 与 Task 1 设计（D/E 仅训练开、PL 不开）完全一致 ✅

> 注：`pl_timing.jsonl` 使用自有字段名（`raw_memory_tokens` / `compressed_memory_tokens` / `memory_compression_ratio`），**没有** Task 2-2A 新增的 `compression_tokens_*` / `sparse_tokens_*` 字段。这是预期的——Task 2-2A 只改了 `train_timing.jsonl`，未动 PL 日志。PL 阶段的 token 信息已由其原生字段覆盖，无需补充。

---

## 6. 新增字段是否全部可用

全部 9 个新字段在两个配置下均正确写入并通过 JSON 序列化：

| 字段 | baseline | A_Dtrain_Etrain |
|---|---|---|
| compression_tokens_before | null ✅ | 4096.0 ✅ |
| compression_tokens_after | null ✅ | 1024.0 ✅ |
| compression_ratio | null ✅ | 0.25 ✅ |
| sparse_tokens_before | null ✅ | 8192.0 ✅ |
| sparse_tokens_after | null ✅ | 4096.0 ✅ |
| sparse_keep_ratio | null ✅ | 0.5 ✅ |
| memory_tokens_scope | str ✅ | str ✅ |
| memory_tokens_before_final | 32768.0 ✅ | 4096.0 ✅ |
| memory_tokens_after_final | 32768.0 ✅ | 4096.0 ✅ |

旧字段全部保留、值合理，无破坏。

---

## 7. 是否需要进一步修复

**不需要。**
- 字段全部正确落盘，null/实值逻辑符合预期。
- 旧字段未受影响，向后兼容。
- PL 阶段 D/E 关闭已确认。
- 未发现 logging 字段写不出的情况，因此**未做任何额外代码修改**（遵守"除非字段写不出否则不改代码"的限制）。

---

## 8. 是否建议进入 Phase 3

**建议进入 Phase 3。**

理由：
- Task 2-2A 新增字段在真实 forward 下全部验证通过（baseline null / A_Dtrain_Etrain 实值）。
- D/E 的 train/PL 行为与 Phase 1 设计一致。
- 效率统计插桩（Phase 2）目标达成：论文 memory token / attention cost 表所需字段已可从 `train_timing.jsonl` 直接读取。

遗留备注（不阻塞 Phase 3）：
- `compression_*` 为"最后一帧 per-chunk"口径；若 Phase 5 token budget 曲线需要"逐帧平均压缩比"，届时再评估是否改累计均值（当前不改）。
- 本次为 8 张 / 2 iter 的功能性 sanity，数值不代表真实训练规模；Phase 3 主实验会产出可用于论文的完整 epoch 级数据。

---

## 附：sanity 产物清单（可随时删除，不影响主结果）

```
experiments/tecma/sanity_runs/
├── subset8.txt
├── task23_baseline_20260615_sanity/            (含 metrics/train_timing.jsonl, pl_timing.jsonl)
├── task23_baseline_20260615_sanity.log
├── task23_A_Dtrain_Etrain_20260615_sanity/     (含 metrics/train_timing.jsonl, pl_timing.jsonl)
└── task23_A_Dtrain_Etrain_20260615_sanity.log
```
