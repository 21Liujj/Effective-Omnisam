# TeCMA Phase 1 报告：实验配置标准化

> 论文主线：Token-Efficient Cross-FoV Memory Adaptation for SAM2-Based Panoramic Semantic Segmentation
> 本报告为 **Phase 1 共用报告**，按子任务分节。只新增 config 文件，未改模型核心代码、未改默认 baseline、未覆盖旧 config。

---

## Task 1-1：TeCMA 实验配置命名规范 + 9 配置清单

**执行日期**：2026-06-15
**约束遵守**：未改模型核心代码；未改默认 baseline；未覆盖任何旧 config（旧 `configs/ablations/*` 原样保留）；新文件全部放 `experiments/tecma/configs/`。

### 配置系统机制（已确认）
- YAML 经 OmegaConf 加载 → 合并 → 提取 `plugins` 子树 → `args.plugins_cfg`（`run_adaptation.py:_load_experiment_overrides` @ :882, setattr @ :1081）。
- `--opts k=v,...` 支持点号覆盖任意键（:897-904）。
- **结论**：YAML 的 `plugins` block 可完全自包含表达 A/D/E 全部开关，无需依赖 `--opts`。本任务 9 个 config 因此**全部自包含**（每个开关显式写明），直接修复 Task 0-2b 缺口#1（旧实验依赖 opts override、目录无快照）。

### 命名规范
- 模块缩写：**A = pseudo_conf_mask / CCPM**；**D = efficient_tam_memory / FSMC**；**E = efficient_sam2_sparse / SG-SMR**。
- 后缀语义：`*train` = 该模块仅训练阶段开启（`apply_in_train=true, apply_in_pl=false`）；`PLon` = D/E 在 PL 生成阶段**也**开启（`apply_in_pl=true`），仅用于反向消融证明 PL 阶段不该开。
- A 无 train/PL 后缀：A 本身只作用于训练期 unsup-loss masking（不涉及 PL 生成），见 Task 0-1。

### 9 个配置清单（均位于 `experiments/tecma/configs/<name>.yaml`，各 29 行，格式统一）

| # | 配置名 | A | D | E | D.train | D.pl | E.train | E.pl | compress_ratio / pool_stride | E keep_ratio / min_keep |
|---|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|---|
| 1 | baseline_omnisam | ✗ | ✗ | ✗ | (true) | false | (true) | false | 0.5 / 2 | 0.5 / 64 |
| 2 | A_ccpm | ✓ | ✗ | ✗ | (true) | false | (true) | false | 0.5 / 2 | 0.5 / 64 |
| 3 | Dtrain_fsmc | ✗ | ✓ | ✗ | true | false | (true) | false | 0.5 / 2 | 0.5 / 64 |
| 4 | Etrain_sgsmr | ✗ | ✗ | ✓ | (true) | false | true | false | 0.5 / 2 | 0.5 / 64 |
| 5 | A_Dtrain | ✓ | ✓ | ✗ | true | false | (true) | false | 0.5 / 2 | 0.5 / 64 |
| 6 | A_Etrain | ✓ | ✗ | ✓ | (true) | false | true | false | 0.5 / 2 | 0.5 / 64 |
| 7 | Dtrain_Etrain | ✗ | ✓ | ✓ | true | false | true | false | 0.5 / 2 | 0.5 / 64 |
| 8 | **A_Dtrain_Etrain** (PRIMARY) | ✓ | ✓ | ✓ | true | **false** | true | **false** | 0.5 / 2 | 0.5 / 64 |
| 9 | **A_Dtrain_Etrain_PLon** (NEG) | ✓ | ✓ | ✓ | true | **true** | true | **true** | 0.5 / 2 | 0.5 / 64 |

注：括号内 `(true)` 表示该模块 enabled=false 时 apply_in_train 标志为惰性（不生效），仍显式写 true 以保持模板一致；所有 config 的 `apply_in_pl` 在模块关闭时均写 false 以杜绝歧义。

### 关键验收：#8 vs #9 唯一差异
`diff A_Dtrain_Etrain.yaml A_Dtrain_Etrain_PLon.yaml` 仅 2 处 `apply_in_pl`（D 和 E）：#8=false，#9=true。其余完全相同。**两者不会混淆。** ✓

### 每个 config 明确的字段（任务要求逐项核对）
- 是否启用 A / D / E：见上表 enabled 列 ✓
- D/E 是否用于 train：`apply_in_train`（全部 true）✓
- D/E 是否用于 PL：`apply_in_pl`（仅 #9 为 true）✓
- compression ratio：`efficient_tam_memory.compress_ratio=0.5`、`pool_stride=2` ✓
- top-k / sparse ratio：`efficient_sam2_sparse.keep_ratio=0.5`、`min_keep_tokens=64`、`mode=attn_topk` ✓
- 输出目录：**不写入 YAML**（YAML 只含 plugins block）；由 CLI `--pd_root` 决定，建议规范见下。

### 输出目录规范（建议，未改脚本）
YAML 不含路径，故沿用既有 CLI 传参。建议 TeCMA 各 config 的 PD_ROOT/日志按统一前缀：
```
PD_ROOT=experiments/tecma/raw_results/<config_name>_<STAMP>
LOG_DIR=experiments/tecma/logs
```
（Phase 3 起实际跑实验时落实；本任务不跑实验、不改启动脚本。）

### 标准运行命令（供 Phase 3 使用，单 config 示例）
```bash
cd /root/autodl-tmp/OmniSAM-main && conda activate omnisam
CUDA_VISIBLE_DEVICES=0 python run_adaptation.py \
  --config experiments/tecma/configs/A_Dtrain_Etrain.yaml \
  --seed 1 --dataset Cityscapes --class_set 13 --backbone sam2_s \
  --data_root data --pd_root experiments/tecma/raw_results/A_Dtrain_Etrain_<STAMP> \
  --src_ckpt /root/autodl-tmp/omnisam_s.pth --tgt_ckpt /root/autodl-tmp/omnisam_s.pth \
  --amp --tf32 --num_epochs 1 --batch_size 4 \
  --pl_batch_size 1 --pl_subset_file <DensePASS train 2000 list> --no-pl_shuffle
```
> 注：因 config 已自包含 A/D/E，**无需再传 `--opts`**（这是相对旧脚本的改进）。其余非插件超参（seed/epoch/batch/patch=image_size 1024）仍由 CLI/默认值决定，与最终实验一致。

### 未决/需说明项（不猜测）
1. **输出目录不进 YAML**：当前框架 YAML 仅承载 plugins，路径走 CLI。若希望 config 完全自描述（含路径/seed/epoch），需在 `_load_experiment_overrides` 增加顶层字段解析——**属改代码，未做**，留待你决定是否要。
2. **config 快照保存**：本任务让 config 自包含已大幅缓解，但「运行时把实际生效配置 dump 到结果目录」仍需改 engine（Task 0-2b 缺口#1）——**未做**，建议作 Task 1-2。
3. 9 个 config 的 compress_ratio/keep_ratio 暂统一用既有默认 0.5；Phase 5 token-budget 曲线会扫这两个值，届时再派生带后缀的变体 config（如 `Etrain_sgsmr_k0.25`），不改这 9 个基准。

### 验收
9 个配置均可明确区分（enabled + apply 矩阵已逐行校验）；**A_Dtrain_Etrain 与 A_Dtrain_Etrain_PLon 仅 apply_in_pl 不同，不会混** → Task 1-1 通过。

---
