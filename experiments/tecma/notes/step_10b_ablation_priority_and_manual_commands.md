# Task 4-0b 消融优先级 + 手动运行命令 — step_10b

> 依 ChatGPT 建议修正重跑优先级，并给出 P0 五个配置的手动运行命令。
> Claude 不运行命令；以下命令由用户手动执行。未跑训练、未启动评估、未改代码、未造数据。
> Generated: 2026-06-15

---

## 1. 优先级总览

### 复用（不重跑）
| Config | 状态 |
|---|---|
| baseline_omnisam | existing_verified_main_result |
| A_Dtrain_Etrain | existing_verified_main_result |

### P0 最小正式消融（5 个，优先）
| # | Config | 论证目的 |
|---|---|---|
| 1 | A_ccpm | A 是精度保护模块 |
| 2 | A_Dtrain | A 保护下 D 的贡献 |
| 3 | A_Etrain | A 保护下 E 的贡献 |
| 4 | Dtrain_Etrain | 无 A 时 D/E 是否伤精度 |
| 5 | A_Dtrain_Etrain_PLon | PL 阶段关闭 D/E 是否合理 |

### P1（时间充足再跑）
| # | Config | 目的 |
|---|---|---|
| 1 | Dtrain_fsmc | D-only 纯隔离 |
| 2 | Etrain_sgsmr | E-only 纯隔离 |

---

## 2. 论证理由

- **A_ccpm**：仅启用 A（pseudo_conf_mask，D/E off）。相对 baseline，证明 **A 作为精度保护模块**对精度的独立作用。
- **A_Dtrain**：A+D。在 A 保护下隔离 **D 的贡献**（token 压缩带来的效率收益，精度是否守住）。
- **A_Etrain**：A+E。在 A 保护下隔离 **E 的贡献**（稀疏检索的效率收益与精度影响）。
- **Dtrain_Etrain**：D+E 无 A。对比 A_Dtrain_Etrain，证明 **没有 A 时 D/E 是否会伤精度** —— 反证 A 保护的必要性。
- **A_Dtrain_Etrain_PLon**：PL 阶段也开 D/E。对比 A_Dtrain_Etrain（PL 关 D/E），证明 **当前"PL 阶段关闭 D/E"是合理决策**（精度/稳定性不因 PL 开 D/E 而更好，反而可能更差或更慢）。
- **D-only / E-only（P1）**：最纯粹的单模块隔离，有价值，但主论证已由 P0 覆盖，故**不列入最小正式消融第一优先级**。

---

## 3. P0 手动运行命令（用户手动执行，Claude 不运行）

说明：
- 使用现有 `run_main_experiments.sh`（`run_ablation_experiments.sh` 尚不存在）。
- 每条一次只跑一个 config，seed=1，full DensePASS protocol（脚本默认 `QUICK_IMAGES=2000`、`NUM_EPOCHS=1`、`TRAIN_MAX_ITERS=0`=跑满 epoch）。
- **不含** `QUICK_IMAGES` / `TRAIN_MAX_ITERS` 等短程参数。
- 输出目录由脚本自动带 `config + seed + 时间戳`，已存在则拒绝写入，**不覆盖旧结果**。
- 建议一个跑完确认 `train_rc=0` 再跑下一个（单卡串行）。

```bash
cd /root/autodl-tmp/OmniSAM-main
```

**P0-1  A_ccpm（A 单独 / 精度保护）**
```bash
SEED=1 bash experiments/tecma/scripts/run_main_experiments.sh A_ccpm
```

**P0-2  A_Dtrain（A+D）**
```bash
SEED=1 bash experiments/tecma/scripts/run_main_experiments.sh A_Dtrain
```

**P0-3  A_Etrain（A+E）**
```bash
SEED=1 bash experiments/tecma/scripts/run_main_experiments.sh A_Etrain
```

**P0-4  Dtrain_Etrain（D+E 无 A）**
```bash
SEED=1 bash experiments/tecma/scripts/run_main_experiments.sh Dtrain_Etrain
```

**P0-5  A_Dtrain_Etrain_PLon（PL 也开 D/E 对照）**
```bash
SEED=1 bash experiments/tecma/scripts/run_main_experiments.sh A_Dtrain_Etrain_PLon
```

> 预计每个约 2.5h（PL 生成 ~2h 为大头），P0 五个串行 ≈ 12–13h。
> 可选 dry-run 先验证命令（不训练、不建目录）：在任意一条前加 `DRY_RUN=1`。

---

## 4. P1 命令（时间充足再跑）

```bash
SEED=1 bash experiments/tecma/scripts/run_main_experiments.sh Dtrain_fsmc
SEED=1 bash experiments/tecma/scripts/run_main_experiments.sh Etrain_sgsmr
```

---

## 5. 跑完后只需贴回（每个 config）

1. `run_summary.json`（输出目录根，全量贴）—— 含 mIoU + 全部 profile 字段。
2. `.../checkpoints/Cityscapes_13cls/sam2_s/metrics/train_timing.jsonl` 末行 —— D/E token、attn time、gpu_mem。
3. `run.log` 中 `[ValSummary] BFScore=` 行（BFScore 证据）。

Claude 收到后：解析 → 追加进 `main_results_registry.md`（Status: pending→verified）→ 更新 `main_results_draft.md` 消融行。

---

## 6. 验收对照

- ✅ 最小正式消融改为 5 个 P0：A_ccpm / A_Dtrain / A_Etrain / Dtrain_Etrain / A_Dtrain_Etrain_PLon。
- ✅ 复用两行：baseline_omnisam / A_Dtrain_Etrain。
- ✅ P1 两个：Dtrain_fsmc / Etrain_sgsmr。
- ✅ 命令一次一个 config、seed=1、full protocol、无短程参数、不覆盖旧结果。
- ✅ Claude 未运行任何命令。
