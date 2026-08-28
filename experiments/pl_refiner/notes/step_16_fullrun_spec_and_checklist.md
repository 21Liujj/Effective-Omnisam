# step_16_fullrun_spec_and_checklist.md
# Phase 4 / 后续 — R2 & R2+R4 Full-run 配置 / 脚本 / 输出目录规范 / 日志字段检查清单

**Run Level**: Spec / config-authoring only
**日期**: 2026-06-25
**执行人**: Claude Code
**是否训练**: No（本步骤只产出配置与脚本，由用户自行在终端执行）
**是否运行 run_adaptation.py**: No
**是否 full-run**: No
**是否改算法 / loss / pseudo_label**: No

---

## 0. 本步骤产出物

| 文件 | 作用 |
|------|------|
| `experiments/pl_refiner/configs/R0_identity_fullrun.yaml` | 匹配 baseline（identity refiner，相同 PL） |
| `experiments/pl_refiner/configs/R2_class_balanced_fullrun.yaml` | R2 full-run 配置 |
| `experiments/pl_refiner/configs/R2R4_proto_gate_fullrun.yaml` | R2+R4 full-run 配置 |
| `experiments/pl_refiner/scripts/run_pl_refiner_fullrun.sh` | 三个 full-run 的统一启动脚本 |
| 本文件 step_16 | 规范与检查清单 |

---

## 1. 设计依据（已实地核对远端代码）

1. **refiner 作用点是训练阶段的 unsup-loss `valid_mask`，不是 PL 生成**（`adaptation/pseudo_label/refiner.py`：`mode` 派发 → `_refine_class_balanced` / `_refine_proto_gate`，只改 `valid_mask`，不改 `pseudo_label` 张量，不改 loss 公式）。
   - **推论**：三个 full-run 必须 **reuse 同一份 base PL**，保证 PL 像素完全相同，唯一变量是 valid_mask 精炼。`reuse_pl_dir` 指向 `data_pd_exp_final_2000/base_last_20260425_232941/.../epoch1`（只读复用，不写入）。

2. **配置 schema 已核对一致**：`class_balance.py` 的 `ClassBalancedSelector` 读取 **group-based** 字段（`common_top_ratio` / `rare_top_ratio` / `extreme_rare_top_ratio` / `*_min_conf` / `min_pixels_per_class`），**不是**旧 `R2_class_balanced.yaml` 里的 `top_ratio: 0.67`。
   - 因此 full-run 的 `class_balance` 块直接沿用 **已验证通过的 `R4_proto_gate_5step_diag.yaml`** 同款 group-based 块，避免使用旧版 schema。
   - ⚠ 旧文件 `configs/R2_class_balanced.yaml` 用的是过时 schema（`top_ratio`/`min_conf` 顶层），**不要用于 full-run**；本步骤新建 `R2_class_balanced_fullrun.yaml`。

3. **proto_gate 块**与 `R4_proto_gate_5step_diag.yaml` 逐字段一致（已通过 R4-4b 5-step 诊断）。refiner 内含硬不变量 `final_valid_mask <= r2_valid_mask`（refiner.py 行 ~487 `raise RuntimeError`），full-run 自动保护。

4. **run-control 差异（唯一区别）**：
   - 诊断：`train_max_iters: 5`（5 步，val 是 artifact，不可用）
   - **full-run：`train_max_iters: 0`（跑满整个 epoch，epoch-end val mIoU 是合法结果）**

5. **输出路径**（核对 `engine.py`）：
   - metrics：`<pd_root>/checkpoints/<dataset>_<class_set>cls/<backbone>/metrics.jsonl`
   - 计时：`.../metrics/train_timing.jsonl`
   - checkpoint：`.../best_iou<XX.XX>.pth`（仅 val mIoU 创新高才存）
   - 这些都落在新的 `pd_root` 下，**不会触碰 `data_pd_exp_final_2000/`**。

---

## 2. Full-run 矩阵与执行顺序

| 顺序 | tag | config | 角色 | 对比对象 |
|------|-----|--------|------|---------|
| 1（必跑，先跑） | `R0_identity` | `R0_identity_fullrun.yaml` | 匹配 baseline（同 PL，identity refiner） | — |
| 2（必跑） | `R2_class_balanced` | `R2_class_balanced_fullrun.yaml` | R2 主结果 | vs R0 |
| 3（建议） | `R2R4_proto_gate` | `R2R4_proto_gate_fullrun.yaml` | R2+R4 主结果 | vs R2 |

**为什么需要 R0_identity 作为 baseline**：现有 `data_pd_exp_final_2000/base_last` 是「不同代码路径 + 无 refiner」的旧 baseline。要严格 claim「R2/R4 带来的增量」，必须在**同一 refiner 框架、相同 PL、identity 模式**下取得对照，排除框架差异。R0_identity 就是这个对照。
（若用户只想要相对 final_2000 baseline 的对比，也可跳过 R0，但论文级结论建议补 R0。）

不在本批次执行：R3 PLQ full-run（已饱和，不建议）、R5 correction、R6 full（仅 lite scan）。

---

## 3. 完整执行命令（用户自行在终端执行）

### 3.0 前置（一次性）

```bash
# 登录后进入仓库与 conda 环境
cd /root/autodl-tmp/OmniSAM-main
conda activate omnisam

# 确认 base PL 存在（只读复用，不可改写）
ls data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1 | head

# 确认源 ckpt 存在
ls -la /root/autodl-tmp/omnisam_s.pth

# 赋予脚本可执行权限
chmod +x experiments/pl_refiner/scripts/run_pl_refiner_fullrun.sh
```

### 3.1 一次性顺序跑全部三个（前台，带 tee 日志）

```bash
cd /root/autodl-tmp/OmniSAM-main
GPU=0 SEED=1 NUM_EPOCHS=1 RUNS="R0 R2 R2R4" \
  bash experiments/pl_refiner/scripts/run_pl_refiner_fullrun.sh
```

### 3.2 推荐：分步执行（先 R0+R2，确认稳定后再 R2R4）

```bash
# 第一批：baseline + R2
cd /root/autodl-tmp/OmniSAM-main
GPU=0 SEED=1 NUM_EPOCHS=1 RUNS="R0 R2" \
  bash experiments/pl_refiner/scripts/run_pl_refiner_fullrun.sh

# 确认 R2 mIoU 合理、无 NaN 后，再跑 R2R4
GPU=0 SEED=1 NUM_EPOCHS=1 RUNS="R2R4" \
  bash experiments/pl_refiner/scripts/run_pl_refiner_fullrun.sh
```

### 3.3 后台长跑（nohup，避免断连中断）

```bash
cd /root/autodl-tmp/OmniSAM-main
mkdir -p experiments/pl_refiner/notes
STAMP=$(date +%Y%m%d_%H%M%S)
nohup env GPU=0 SEED=1 NUM_EPOCHS=1 RUNS="R0 R2 R2R4" STAMP="$STAMP" \
  bash experiments/pl_refiner/scripts/run_pl_refiner_fullrun.sh \
  > experiments/pl_refiner/notes/fullrun_master_${STAMP}.log 2>&1 &
echo "PID=$! STAMP=$STAMP"
# 跟踪：
tail -f experiments/pl_refiner/notes/fullrun_master_${STAMP}.log
```

### 3.4 单独只跑某一个（手动调用 run_adaptation.py，等价单跑）

```bash
cd /root/autodl-tmp/OmniSAM-main
STAMP=$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/omnisam/bin/python3 run_adaptation.py \
  --config experiments/pl_refiner/configs/R2_class_balanced_fullrun.yaml \
  --data_root data \
  --pd_root experiments/pl_refiner/fullruns/R2_class_balanced_${STAMP} \
  --dataset Cityscapes --backbone sam2_s --class_set 13 \
  --src_ckpt /root/autodl-tmp/omnisam_s.pth \
  --tgt_ckpt /root/autodl-tmp/omnisam_s.pth \
  --reuse_pl_dir data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1 \
  --num_epochs 1 --batch_size 4 --seed 1 \
  2>&1 | tee experiments/pl_refiner/notes/fullrun_R2_class_balanced_${STAMP}.log
```

---

## 4. 输出目录规范

每个 run 写入独立、带时间戳的目录（绝不复用旧目录，符合 experiments 规则）：

```
experiments/pl_refiner/fullruns/
├── R0_identity_<STAMP>/
│   ├── run_meta.txt                         # seed/gpu/commit/config/cmd/起止时间
│   ├── DensePASS/sam2_s/Cityscapes_13cls/epoch1/   # （reuse 模式下 save_dir 指向复用 PL，不重生成）
│   └── checkpoints/Cityscapes_13cls/sam2_s/
│       ├── metrics.jsonl                    # epoch-end mIoU/mAcc/PixAcc/FWIoU/mDice/BFScore
│       ├── metrics/train_timing.jsonl       # 每 step 计时
│       └── best_iou<XX.XX>.pth              # 仅创新高才存
├── R2_class_balanced_<STAMP>/  （同结构）
└── R2R4_proto_gate_<STAMP>/    （同结构）

experiments/pl_refiner/notes/
├── fullrun_R0_identity_<STAMP>.log
├── fullrun_R2_class_balanced_<STAMP>.log
└── fullrun_R2R4_proto_gate_<STAMP>.log

experiments/pl_refiner/results/
├── r2_class_balanced_fullrun_diag.jsonl     # R2 每 step refiner 诊断
└── r2r4_proto_gate_fullrun_diag.jsonl       # R2R4 每 step refiner 诊断
```

**目录命名规则**：`<tag>_<STAMP>`，`STAMP=YYYYmmdd_HHMMSS`。三个 run 共用同一 STAMP（脚本默认）便于成组对比。

**禁止**：
- 不写入 `data_pd_exp_final_2000/`（只读 reuse）
- 不复用上一次 full-run 的 `pd_root`
- 多次实验请用不同 STAMP，避免覆盖

---

## 5. 日志字段检查清单（每个 full-run 跑完后逐项核对）

### 5.1 启动/环境核对（开跑前看 run_meta.txt 与 log 头部）

- [ ] `seed=1`，`gpu`、`commit`、`config` 路径、`reuse_pl_dir` 已记录到 `run_meta.txt`
- [ ] log 头部 `Reuse fixed pseudo-label dir for epoch 1: data_pd_exp_final_2000/base_last_...`（确认确实复用，未重生成 PL）
- [ ] `[plugins] enabled=['pseudo_conf_mask']`（D/E 关闭，符合 R2/R4 设定）
- [ ] `train_max_iters` **未出现 5**（应跑满 epoch；diag artifact 的 5-step 不应再现）

### 5.2 refiner 行为核对（diag jsonl + refiner_verbose log）

通用：
- [ ] 每条 record `refiner_mode` 正确：R2→`class_balanced`，R2R4→`proto_gate`
- [ ] `nan_count == 0` 全程
- [ ] `proto_sim_nan_count == 0` 全程（R2R4 / 含 proto 诊断时）
- [ ] `loss_sup`、`loss_unsup` 全程有限、无爆炸（无 inf/NaN，无突然归零）

R2 专项（`r2_class_balanced_fullrun_diag.jsonl`）：
- [ ] `r2_total_valid_ratio` 稳定在合理区间（诊断期约 0.46–0.47，full-run 可漂移但不应塌缩到 0 或跳到 1）
- [ ] `r2_common_valid_ratio` / `r2_rare_valid_ratio` / `r2_extreme_rare_valid_ratio` 均有值
- [ ] starvation guard（`skipped` 中 `starvation_kept_all_*` / `all_below_min_conf_*`）只在极稀类偶发，不应在 common 类频繁触发

R2R4 专项（`r2r4_proto_gate_fullrun_diag.jsonl`）：
- [ ] **硬不变量**：`proto_gate_final_valid_ratio <= r2_total_valid_ratio` 每条都成立（refiner 内已 assert，若违反会直接抛错终止——log 中不应有 `proto_gate invariant failed`）
- [ ] `proto_gate_filter_ratio_total` 约 0.08–0.12（设计区间）
- [ ] `proto_gate_filter_ratio_common` 约 0.10
- [ ] `proto_gate_filter_ratio_rare` 约 0.05（< common，rare 保护）
- [ ] `proto_gate_filter_ratio_extreme_rare == 0` 或 `null`（fallback keep）
- [ ] `proto_gate_fallback_reasons` 仅在 wall/traffic light/traffic sign/像素过少类出现（`no_valid_pixels`/`extreme_rare_keep`/`below_min_filter_pixels_*`），属设计内行为

### 5.3 字段命名（沿用 step_15 审计结论，写报告时严格执行）

- [ ] R2 前置比例只引用：`r2_total_valid_ratio` / `r2_common_valid_ratio` / `r2_rare_valid_ratio` / `r2_extreme_rare_valid_ratio`
- [ ] R4 后置比例只引用：`proto_gate_final_valid_ratio` / `proto_gate_final_common_valid_ratio` / `proto_gate_final_rare_valid_ratio` / `proto_gate_final_extreme_rare_valid_ratio`
- [ ] **不引用** `total_valid_ratio_after_R2` / `rare_valid_ratio_after_R2` / `extreme_rare_valid_ratio_after_R2` / `refiner_total_valid_ratio`（命名歧义，见 step_15 §7）

### 5.4 训练完成性核对

- [ ] log 出现 `Epoch 1 [Train]: 100%|...| <N>/<N>`，N 远大于 5（跑满 epoch）
- [ ] log 末尾 `Training Finished.`
- [ ] `metrics.jsonl` 存在且含 `mIoU`/`mAcc`/`PixAcc`/`FWIoU`/`mDice`/`BFScore`
- [ ] `train_timing.jsonl` 存在
- [ ] 若 mIoU 创新高，`best_iou<XX.XX>.pth` 已生成

### 5.5 结果合法性（与 diag artifact 区分）

- [ ] **本次 full-run 的 epoch-end val mIoU 是合法结果**（跑满 epoch，可写入报告/ledger）
- [ ] 区别于 R4-4b 的 `val mIoU=60.22`（那是 5-step diag artifact，不可用，见 step_15 §9）
- [ ] 结果写入 `docs/claude/experiment-ledger.md`（含 seed/gpu/commit/config/STAMP）

---

## 6. 对比与判读建议

跑完三个 run 后，按以下对比得结论（每项需 metrics.jsonl 支撑，推测须标注）：

1. **R2 vs R0_identity**：mIoU/BFScore 差值 → R2 class-balanced selection 的净增益
2. **R2R4 vs R2**：mIoU/BFScore 差值 → R4 proto_gate 在 R2 之上的增量
3. **valid_ratio 轨迹**：R0 → R2 → R2R4 的 `*_valid_ratio` 递减是否符合「过滤越来越严」的预期
4. 若 R2R4 mIoU ≥ R2 且 BFScore 不降 → proto_gate 可保留进入 R7_full
5. 若 R2R4 < R2 → 需回看 `proto_gate` 阈值是否过激（调 `common_filter_percentile`），属 R4-tuning，不在本批次

---

## 7. 严格禁止（沿用项目规则）

- 不覆盖 `data_pd_exp_final_2000/` 的 checkpoint 与 PL（只读复用）
- 不复用旧 `pd_root` 目录
- 不在 full-run 中开启 D/E（保持 refiner 净效应；D/E 是另一条已收口的加速线）
- 不进入 R5 correction / R6 full implementation
- 结论必须由 `metrics.jsonl` / `*_diag.jsonl` / `.log` 支撑

