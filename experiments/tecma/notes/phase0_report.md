# TeCMA Phase 0 报告：代码与实验现状盘点

> 论文主线：Token-Efficient Cross-FoV Memory Adaptation for SAM2-Based Panoramic Semantic Segmentation
> 本报告为 **Phase 0 共用报告**，按子任务分节追加。只读盘点，未修改任何代码。
> 仓库：`/root/autodl-tmp/OmniSAM-main`

---

## Task 0-1：代码仓库扫描（只读，建立代码地图）

**执行日期**：2026-06-15
**约束遵守**：未修改任何代码；不确定项标注「未确认」；未重构。

### 1. 训练 / 评估 / PL 入口

| 功能 | 入口文件 | 关键位置 | 说明 |
|---|---|---|---|
| UDA 训练 | run_adaptation.py | main() @ :1052；argparse @ :1039；YAML @ :895 | argparse + --config YAML；支持 --plugins 点号 override（:850） |
| 源域预训练 | train.py | main() @ :124 | OmniSAM（非 adapt） |
| 单模型验证 | validation.py | main() @ :662；指标聚合 @ :444-485 | 输出 mIoU/mAcc/PixAcc/FWIoU/mDice |
| 多模型对比 | validation_muti.py | — | compare_omnisam_vs_60_36 |
| PL 生成 | engine.py → PseudoLabelGenerator.run() | adaptation/pl_generator.py | Engine 自动触发，不单独跑 |

训练命令：
```
RUN_IN_BACKGROUND=1 SRC_CKPT=/root/autodl-tmp/omnisam_s.pth TGT_CKPT=/root/autodl-tmp/omnisam_s.pth bash scripts/run_plugin_final_2000_base_vs_ADEtrain.sh
```
评估命令：
```
CKPT=/path/to/best.pth bash scripts/validate_densepass.sh
CKPT=/path/to/best.pth bash scripts/validate_wildpass.sh
```

### 2. A / D / E 模块位置

| 模块 | 代码名 | 定义文件 | 调用/注入点 |
|---|---|---|---|
| A pseudo_conf_mask | pseudo_conf_mask | adaptation/plugins/pseudo_conf_mask.py（masked_unsup_ce） | strategies.py:9 import；strategies.py:294-326 启用 |
| D memory压缩 | efficient_tam_memory | model/modules/memory_compressor.py（SpatialMemoryCompressor） | engine.py:239-262 → memory_bank.configure_compression()@131；forward 内 memory_compressor.compress()（concat 前 before_attention） |
| E 稀疏检索 | efficient_sam2_sparse | model/modules/sparse_memory_retriever.py（SparseMemoryRetriever） | engine.py:265-297 → memory_bank.configure_sparse_retrieval()@189；forward 内 sparse_retriever.retrieve()（concat 前 逐 chunk） |

插件管理：adaptation/plugins/manager.py（PluginManager，default_cfg@38，enabled@107）。另有 depth_boundary_prior/depth_confidence（非 TeCMA 范围，默认 disabled）。

### 3. 配置开关表（manager.py default_cfg，全部默认 enabled:false）

A — pseudo_conf_mask: enabled=false, threshold=0.85, min_valid_ratio=0.01, ignore_index=255, fallback=skip

D — efficient_tam_memory: enabled=false, mode=pool, compress_ratio=0.5, pool_stride=2, **apply_in_pl=true(默认)**, apply_in_train=true

E — efficient_sam2_sparse: enabled=false, mode=attn_topk, keep_ratio=0.5, min_keep_tokens=64, saliency_source=feature_norm, cache_saliency=true, fallback_full_when_uncertain=true, **apply_in_pl=true(默认)**, apply_in_train=true

⚠️ default_cfg 里 D/E 的 apply_in_pl 默认 true；较优配置 A_Dtrain_Etrain 必须在 YAML 显式设 apply_in_pl:false（禁令#10）。

D/E 区分 train vs PL：
- 配置层：YAML apply_in_train/apply_in_pl → configure_compression(enable_in_train=, enable_in_pseudo_label=) / configure_sparse_retrieval(...)
- 运行层：set_runtime_context(stage=)@memory_bank.py:185 设 stage；_compression_enabled_now()@231 与 _sparse_enabled_now()@216 据此决定本次 forward 是否生效。

### 4. memory attention 位置（TeCMA 核心）

| 项 | 位置 |
|---|---|
| 调用点 | model/memory_bank.py:407 self.sam_model.memory_attention(curr=, curr_pos=, memory=, memory_pos=, num_obj_ptr_tokens=) |
| forward | model/memory_bank.py:274 |
| Q (query) | vision_feats（当前 patch 特征 (L,B,C)）→ curr |
| Q pos | vision_pos_embeds → curr_pos |
| K/V (memory) | memory=torch.cat(to_cat_memory)（历史 memory tokens，FIFO num_maskmem=6）→ memory |
| K/V pos | memory_pos_embed=torch.cat(to_cat_memory_pos_embed) → memory_pos |
| 模块定义 | sam/sam/modeling/sam2_base.py:120；装配 sam/training/model/sam2.py:29（sam/ 目录，禁改） |
| D 注入 | forward 内 concat 前 memory_compressor.compress(feats, maskmem_enc, mem_hw) |
| E 注入 | forward 内 concat 前 逐 chunk sparse_retriever.retrieve(memory_tokens=feats, ...) |

已存在的效率插桩（影响 Phase 2 工作量）：memory_bank forward 内已用 time.perf_counter() 包裹 memory_attention，维护：last_attention_time/attention_time_accum/attention_time_count；last_memory_tokens_before/after/last_memory_token_ratio；last_compression_stats（original/compressed_num_tokens, compression_ratio）；last_sparse_tokens_before/after, last_sparse_keep_ratio。

### 5. 结果文件位置

- 最终对比实验：data_pd_exp_final_2000/{base_last_20260425_232941, A_Dtrain_Etrain_20260425_232941}/
- checkpoint：…/checkpoints/Cityscapes_13cls/sam2_s/{best_iou<XX.XX>.pth, last_sam2_s_13cls.pth, resume_sam2_s.pth}
- 每 epoch 指标：…/metrics.jsonl（val_mIoU/mAcc/PixAcc/FWIoU/mDice, best_ckpt_path, avg_patch_count_*）
- 训练耗时：…/metrics/train_timing.jsonl（step_time/epoch_time/iter_compute_time/train_patches_per_sec/memory_tokens_*）
- PL 耗时：…/metrics/pl_timing.jsonl
- 训练日志：logs/omnisam_final_2000/{base_last_*, A_Dtrain_Etrain_*}.log、*.master.log
- WildPASS 结果：logs/wildpass_results/{reports, summaries, val_ckpts, train_time_reuse_pl}
- 官方源域 ckpt：/root/autodl-tmp/omnisam_s.pth

timing 写入位置：PL→engine.py:486 写 pl_timing.jsonl + 聚合 pl_generator.py:349-353；Train→engine.py:598 写 train_timing.jsonl + engine.py:721-722（memory_tokens_before/after）+ strategies.py:565-573（每 iter memory 统计采集）。

### 6. 指标计算位置

utils/metrics.py：fast_hist@9、per_class_iu@14、per_class_dice@18、freq_weighted_dice@26、boundary_f_score@61、scores_from_hist@97（返回 mIoU/mAcc/PixAcc/FWIoU/mDice，0–100）。
- mIoU / per-class IoU：per_class_iu + scores_from_hist；validation.py:444 与 trainer.py 验证循环均调用。
- BFScore：boundary_f_score。**仅在 adaptation/trainer.py:313/344 训练期验证循环计算**并写入 metrics["BFScore"]（打印 [ValSummary] BFScore= @ trainer.py:352）。

### 7. 目前可能存在的风险

1. BFScore 与独立 validation.py 脱钩：validation.py 标准输出不含 BFScore，BFScore 只在训练内验证（trainer.py）产生。Phase 3 若用 validation.py 单独复评拿不到 BFScore。未确认 validation_muti.py 是否单独算 BFScore。
2. D/E 的 apply_in_pl 默认 true，与 A_Dtrain_Etrain（PL 关 D/E）相反。复现漏设 YAML 会违反禁令#10。
3. E 的 saliency 缓存：cache_saliency=true + memory_id 含 runtime_iter；跨 run 是否正确重置未确认，可能影响速度可比性。
4. train_timing.jsonl 完整字段未逐一确认：已知含 memory_tokens_*，是否含显存峰值、D/E 各自 token 拆分未确认（Phase 2 需补）。
5. 顶层无 experiments/ 目录：现有产物分散在 data_pd_exp_final_2000/、logs/、exp2/。本报告新建 experiments/tecma/notes/，未来 TeCMA 产物建议统一归此。

### 8. 后续建议的最小改动点（仅建议，未实施）

- Phase 1：D/E 的 apply_in_pl/apply_in_train 每份 YAML 显式写明；统一命名到 configs/tecma/（新增，不动 configs/ablations/）。
- Phase 2：核心量（attention 耗时/memory token before-after/压缩比/keep_ratio）已在 memory_bank.py 暴露且已写入 train/pl timing。最小改动=在 engine.py/strategies.py 聚合处补：(a) 显存峰值 torch.cuda.max_memory_allocated，(b) D vs E 各自 token 削减拆分。memory attention 本体在 sam/，禁改——所有计时统计在 memory_bank.py 包裹层完成，无需碰 sam/。
- Phase 3：直接用 scripts/run_plugin_final_2000_base_vs_ADEtrain.sh，核对 YAML 中 D/E PL 关闭。

验收自检：训练命令✅ / 评估命令✅ / A/D/E 文件位置✅ / config 开关名✅ / memory attention 函数位置✅（memory_bank.py:407）/ 结果输出格式✅。D/E 与 memory attention 均已定位 → Task 0-1 通过。

---

## Task 0-2：已有实验结果资产盘点（只读，不重跑）

**执行日期**：2026-06-15 ｜ **约束**：未改代码；未删/覆盖结果；未合并；缺失项写「缺失」不编造。
**数据来源**：直接读取 metrics.jsonl / train_timing.jsonl / pl_timing.jsonl / 训练 .log（tr 去 tqdm 进度条后 grep）/ WildPASS reports & val_ckpts JSON。

### 资产表（已有结果）

| 实验名 | 配置 | 数据集 | 是否完整 | mIoU | BFScore | 训练耗时 | 端到端耗时 | 显存(GB) | 日志路径 | 可用于论文 |
|---|---|---|---|---:|---:|---|---|---:|---|---|
| **base_last** (最终对比) | baseline（A/D/E 全关） | Cityscapes→DensePASS val 100, PL 2000, seed=1 | ✅完整 | **60.26** | **39.97** | step 5.27s/it；epoch_train 2100s(35m) | PL 7002s + train ≈ 2h40m | train峰 26.82 / PL峰 8.12 | `data_pd_exp_final_2000/base_last_20260425_232941/`；`logs/omnisam_final_2000/base_last_*.log` | ✅主表 baseline |
| **A_Dtrain_Etrain** (最终对比) | A on；D/E 仅 train（PL 关） | 同上 | ✅完整 | **60.16** | **39.86** | step 4.08s/it；epoch_train 1497s(25m) | PL 7082s + train ≈ 2h32m | train峰 25.34 / PL峰 8.12 | `data_pd_exp_final_2000/A_Dtrain_Etrain_20260425_232941/`；`logs/omnisam_final_2000/A_Dtrain_Etrain_*.log` | ✅主表 method |
| triplet/base | baseline | DensePASS val（样本量**未确认**，疑同最终） | ⚠️部分 | 60.35 | 39.38 | 缺失(stdout 无 timing) | 缺失 | 缺失 | `logs/omnisam_time_ablation_seed1_triplet/...base_last_*.log` | △辅助/需重核 |
| triplet/A+D (plus1_plus4) | A+D（train） | 同上 | ⚠️部分 | 60.20 | 39.00 | 缺失 | 缺失 | 缺失 | `..._plus1_plus4_*.log` | △消融辅助 |
| triplet/A+D+E (plus1_plus4_plus5_trainonly) | A+D+E（train） | 同上 | ⚠️部分 | 60.09 | 38.96 | 缺失 | 缺失 | 缺失 | `..._plus1_plus4_plus5_trainonly_*.log` | △消融辅助 |
| ta/A only (plus1_confmask) | 仅 A | DensePASS val（样本量未确认） | ⚠️部分 | 60.16 | 缺失(未提取) | 缺失 | 缺失 | 缺失 | `logs/omnisam_time_ablation/plugin_ablation_base_plus1_confmask_*.log` | △单模块消融 |
| ta/D only (plus4_efficienttam) | 仅 D | 同上 | ⚠️部分 | 59.98 | 缺失 | 缺失 | 缺失 | 缺失 | `..._plus4_efficienttam_*.log` | △单模块消融 |
| ta/E only (plus5_sparse) | 仅 E | 同上 | ⚠️部分 | 59.94 | 缺失 | 缺失 | 缺失 | 缺失 | `..._plus5_efficientsam2_sparse_*.log` | △单模块消融 |
| WildPASS eval (debug) | base ckpt 直接评 | WildPASS13, 500 张 | ❌异常 | **0.1014** | 缺失 | — | eval 1349s | — | `logs/wildpass_results/reports/wildpass_debug_eval_metrics.json` | ❌不可用(崩溃) |
| WildPASS val_ckpts ×3 run | final vs paper ckpt | WildPASS13, 500 张 | ❌异常 | 0.0975–0.1027 | 缺失 | — | eval ~1320–1400s | — | `logs/wildpass_results/val_ckpts/{160538,180208,190132}/` | ❌不可用(崩溃) |
| DensePASS sanity | base ckpt 单独复评 | DensePASS（验证脚本） | ✅完整 | **60.34** | 缺失(validation.py 不算 BFScore) | — | eval | — | `logs/wildpass_results/reports/densepass_sanity_metrics.json` | ✅可作 DensePASS 复现核对 |

### 关键数值（最终对比实验，已确认可信）

| 量 | base | A_Dtrain_Etrain | 差值 |
|---|--:|--:|--:|
| mIoU | 60.26 | 60.16 | -0.10 |
| BFScore | 39.97 | 39.86 | -0.11 |
| mAcc | 71.32 | 71.42 | +0.10 |
| step_time | 5.27 s/it | 4.08 s/it | **-22.5%** |
| epoch_train_time | 2100s (35m) | 1497s (25m) | -10m |
| memory_attention_time (train, 1 epoch) | 263.0s | 93.6s | **-64.4%** |
| memory_tokens (train, per attn) | 32768 | 4096 | **8× 压缩(0.125)** |
| train 显存峰值 | 26.82 GB | 25.34 GB | -1.48 GB |
| PL total | 7002s | 7082s | +80s（D/E PL 关，基本持平）✅符合禁令#10 |
| PL 显存峰值 | 8.12 GB | 8.12 GB | 0 |

> 注：train memory_tokens 32768 vs 4096 是「单次 attention 全 batch 累计 token」口径（memory_bank_size=9）；CLAUDE.md 记的 4096→1024 per-frame 是另一口径，两者不矛盾但**口径需在 Phase 2 统一**。

### 判定：哪些可直接用 / 临时 / 需重跑

**A. 可直接用于论文主结果表**
- base_last（60.26 / BF 39.97）与 A_Dtrain_Etrain（60.16 / BF 39.86）：metrics + timing + 显存 + token 全齐，seed=1，2000 张 PL，DensePASS val 100。→ **主表 + 效率表的核心两行已就绪**。

**B. 临时 / 辅助（需补齐才能进表）**
- triplet（base / A+D / A+D+E）与单模块消融（A / D / E）：**有 mIoU、部分有 BFScore，但缺 timing/显存/样本量元数据**。可作消融趋势佐证，进正式消融表前需：(1) 确认样本量与 val 协议一致；(2) 补 step_time / memory_attention_time / 显存 / token。→ Phase 4 处理。

**C. 必须重跑**
- **WildPASS 全部结果**：mIoU≈0.10、PixAcc>1.0，明显是 **label 映射/转换 bug**（仓库已有 `wildpass_label_conversion_fix_report` 但结果未更新）。当前完全不可用。→ 若 Phase 8 要 WildPASS，须先修 label 转换再重评。

### 缺失清单（明确标注，未编造）
1. triplet / 单模块消融的 timing（step_time、memory_attention_time）、显存峰值、token before/after — **缺失**。
2. 单模块消融（A/D/E）的 BFScore — 日志中存在但本次未逐一提取 — **未提取**（非缺失，可补）。
3. 消融实验的样本量（PL 张数 / val 张数）是否与最终实验一致 — **未确认**。
4. WildPASS / DensePASS 的 BFScore（走 validation.py 路径，不算 BFScore）— **结构性缺失**。
5. `wildpass_summary.md` 与 `wildpass_train_time.csv` / `wildpass_val_ckpts.csv` — **只有表头，正文空**（汇总脚本未填）。

### 验收：已得到「已有结果资产表」，后续只补缺不重复浪费算力。
- 主表两行（base / A_Dtrain_Etrain）✅就绪
- 消融多行 △需补 timing/元数据
- WildPASS ❌须修 bug 重跑

---

## Task 0-2b：主结果口径复核（只读，不重跑）

**执行日期**：2026-06-15 ｜ **约束**：未改代码；未重跑；无证据写「未确认」。
**证据来源**：启动脚本 `scripts/run_plugin_final_2000_base_vs_ADEtrain.sh`、`configs/ablations/{omnisam_base.yaml, omnisam_base_plus5_efficientsam2_sparse.yaml}`、两实验 `metrics.jsonl` / `metrics/train_timing.jsonl` / `metrics/pl_timing.jsonl`、`logs/omnisam_final_2000/*.master.log`、`dataset/densepass_list/{train,val,label}.txt`、`run_adaptation.py`、`adaptation/{trainer.py,engine.py,strategies.py}`、`utils/metrics.py`。

### 复核表

| Item | base_last | A_Dtrain_Etrain | Evidence Path | Risk |
|---|---|---|---|---|
| source dataset | Cityscapes（13cls） | Cityscapes（13cls） | 启动脚本 DATASET=Cityscapes CLASS_SET=13 | 低 |
| target dataset (PL) | DensePASS train | DensePASS train | `--pl_subset_file …DensePASS…`；`INPUT_LIST=dataset/densepass_list/train.txt` | 低 |
| validation dataset | DensePASS val | DensePASS val | `dataset/densepass_list/val.txt` | 低 |
| **validation 样本数** | **100** | **100** | `val.txt`=100 行（= DensePASS 官方标注 val 全集） | **低（是全集，非 subset）** |
| **PL 样本数** | **2000** | **2000** | QUICK_IMAGES=2000；`train.txt`=2000 行 → **PL=DensePASS train 全量** | 低 |
| train 样本数 | = PL 2000（用 PL 训练） | 同 | TRAIN_MAX_ITERS=0（不截断，跑满 1 epoch） | 低 |
| **epoch 数** | **1** | **1** | master.log `NUM_EPOCHS=1` | **中（单 epoch）** |
| seed | 1 | 1 | 启动脚本 SEED=1 | **中（单 seed）** |
| batch size | 4（PL=1） | 4（PL=1） | BATCH_SIZE=4 PL_BATCH_SIZE=1 | 低 |
| patch / crop size | 1024×1024 | 1024×1024 | `run_adaptation.py:130` image_size default=1024 | 低 |
| A (pseudo_conf_mask) | **off** | **on** | base: YAML+opts enabled=false；ADE: opts `pseudo_conf_mask.enabled=true` | 低 |
| D (efficient_tam_memory) | off | **on (train)** | ADE opts `efficient_tam_memory.enabled=true` | 低 |
| E (efficient_sam2_sparse) | off | **on (train)** | base off；ADE YAML enabled=true + opts confirm | 低 |
| D apply_in_train | n/a | **true** | ADE opts `efficient_tam_memory.apply_in_train=true` | 低 |
| D apply_in_pl | n/a | **false** | ADE opts `efficient_tam_memory.apply_in_pl=false` | 低（符合禁令#10） |
| E apply_in_train | n/a | **true** | ADE opts `efficient_sam2_sparse.apply_in_train=true` | 低 |
| E apply_in_pl | n/a | **false** | ADE opts `efficient_sam2_sparse.apply_in_pl=false` | 低（符合禁令#10） |

> ⚠️ **配置真值 = YAML + `--opts` override**。实验目录内**未保存 config 快照**（`data_pd_exp_final_2000/*/` 下只有 DensePASS/ 与 checkpoints/，无 yaml/command.txt）。当前配置只能从「启动脚本固定写死的 opts 字符串 + master.log 回显」还原，**未确认运行时是否被环境变量改写**（脚本里多数值有默认，master.log 已回显 NUM_EPOCHS/QUICK_IMAGES/SEED/BATCH 等关键项，与上表一致）。→ Phase 1 必须落实 config 快照保存。

### 2. metrics.jsonl 复核

| 问题 | 答案 | 证据 |
|---|---|---|
| mIoU=60.26 / 60.16 来自哪个 epoch | **epoch 1（唯一 epoch）** | metrics.jsonl `"epoch":1`；master.log NUM_EPOCHS=1 |
| best 还是 last checkpoint | **best==last**（仅 1 epoch，二者同一权重） | `best_ckpt_path` = `best_iou60.26.pth` / `best_iou60.16.pth`；engine.py:668 best 命名；同时存 `last_sam2_s_13cls.pth` |
| val 样本数是否记录在 metrics | **未记录**（metrics.jsonl 无 num_images 字段） | KEYS 中无 image-count 字段；样本数由 `val.txt`=100 旁证 |
| 是否有 per-class IoU | **metrics.jsonl 无**；但 `validation.py` 路径有（per_class_iou_pct，见 densepass_sanity_metrics.json） | metrics.jsonl KEYS 无 per_class；sanity json 有 13 类 IoU |
| BFScore 39.97 / 39.86 来自哪个字段 | **不在 metrics.jsonl**；来自训练日志 `[ValSummary] BFScore=` 行（`trainer.py:352` 打印，`:344` 由 `boundary_f_score` tolerance=2 计算） | `logs/omnisam_final_2000/{base_last,A_Dtrain_Etrain}_*.log` grep |

### 3. timing 口径复核（代码级确认）

| 量 | base | ADE | 口径（代码） |
|---|--:|--:|---|
| step_time | 5.27 s/it | 4.08 s/it | `trainer.py:264` = **一个 epoch 内全部 iter 时长的平均**（sum(iter_durations)/n_iters）；含 data+compute |
| epoch_train_time | 2100s | 1497s | `trainer.py:274` = **sum(compute_durations)**（纯计算，不含 data load）；另有 epoch_total_time（含全部）base 2633s / ADE 2043s |
| memory_attention_time | 263.0s | 93.6s | train_timing.jsonl 字段；= **epoch 内 memory_attention 累计耗时**（per_call=0.0329/0.0117s × ~8000 calls）；底层 `memory_bank.py` 用 perf_counter 包裹，`_sync_cuda` 前后同步 → **GPU 同步计时，口径可靠** |
| memory_tokens | 32768 | 4096 | train_timing.jsonl `memory_tokens_before/after`；= **单次 attention 输入 K/V token 数**（before==after，因 D 在该口径已生效后的稳定值）；**与 CLAUDE.md 4096→1024 per-frame 口径不同**（此为全 batch 累计），Phase 2 须统一 |
| train 显存峰值 | 26.82 GB | 25.34 GB | `engine.py:529` = `torch.cuda.max_memory_allocated()/1024^3` → **进程级峰值，口径可靠** |
| PL total | 7002s | 7082s | pl_timing.jsonl `pl_total_time`；= forward+backward+merge+confidence 全链路（merge 占大头 ~1300s） |

### 4. 主表资格判断

**结论：完整正式实验（full official protocol），但是单 seed / 单 epoch 的「最小正式实验」。**

- ✅ **DensePASS val=100 是官方标准全集**（DensePASS 标注验证集就是 100 张全景图），**不是 debug/subset**。
- ✅ **PL=2000 = DensePASS train 全量**，非抽样子集。
- ✅ patch 1024、batch 4、AMP+TF32、seed=1、源/目标 ckpt = `/root/autodl-tmp/omnisam_s.pth`（脚本默认 `exp2/...best_sam2_s_10.pth`，本次由环境变量覆盖为 omnisam_s.pth — **此点未在 master.log 回显，未确认实际用的哪个**；两 run 一致即不影响对比公平性）。
- ⚠️ **单 epoch + 单 seed**：符合该项目既有 UDA 复现惯例，可作主表，但**论文需明确标注 "1 epoch on full DensePASS-train PL, seed=1"**，并建议 Phase 3 至少补 1–2 个额外 seed 以给出 ±std（否则审稿易质疑 ±0.1 的差异在噪声内）。

定位：**完整正式实验** ✓（非 subset/debug/preliminary），但属「单次运行」，统计显著性需补 seed。

### 5. val100 说明

- val100 = **DensePASS 标准评估全集**（官方 100 张），**符合 DensePASS 标准评估，无需补"完整 DensePASS validation"**（100 已是完整 val）。
- **不需要重训**：mIoU/BFScore 已由训练内验证产生且口径正确。
- **建议轻量重评（非必须）**：若主表要 **per-class IoU**（论文常配）或要让 BFScore 进 metrics 文件，可用 `validation.py` 对已存 `best_iou*.pth` **重评 checkpoint**（只评不训，~分钟级）。注意 `validation.py` 当前**不输出 BFScore**（结构性缺口，见 Task 0-1 风险#1）；BFScore 仍只能从训练日志取，或需在 validation.py 接 `boundary_f_score`。

### 6. 是否可进入 Phase 1？

- 是否可直接作论文主结果：**可（作为 full-protocol 单次运行的主表两行）**，但需在论文标注单 seed/单 epoch；强烈建议 Phase 3 补 seed。
- 是否需要重评：**非必须**；若要 per-class IoU / 让 BFScore 入文件，做一次只读重评即可（不重训）。
- 是否需要重训：**否**。
- **是否可进入 Phase 1：可以。** 主结果口径已查清，无阻塞性问题。Phase 1 第一件事应是「config 快照保存 + BFScore 接入 validation.py + metrics 增补 per-class IoU/val 样本数」，正好补齐本次发现的三个缺口。

### 缺口（带去 Phase 1）
1. 实验目录无 config 快照 → Phase 1 落实自动保存 YAML+opts+seed+commit。
2. metrics.jsonl 无 per-class IoU、无 val 样本数 → Phase 1 增补。
3. BFScore 未入 validation.py / metrics 文件，仅在训练日志 → Phase 1 接入。
4. src/tgt ckpt 实际路径未在 master.log 回显 → 未确认（两 run 一致，不影响对比）。
5. memory_tokens 口径（全 batch 累计 vs per-frame）与 CLAUDE.md 不一致 → Phase 2 统一。

---
