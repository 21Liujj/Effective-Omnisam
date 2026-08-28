# Task 4-0 旧消融复核与重跑清单 — step_10_old_ablation_reuse_audit.md

> 只读审计：判断哪些旧 A/D/E 消融可复用、哪些必须为正式消融表重跑。
> 未跑训练、未改代码、未生成假数据。
> Generated: 2026-06-15

---

## 0. 命名映射（plugin → A/D/E）

从日志 `[plugins] enabled=...` 行确认：

| 旧日志命名 | plugin | TeCMA 模块 |
|---|---|---|
| plus1 / confmask | `pseudo_conf_mask` | **A**（伪标签置信掩码） |
| plus4 / efficienttam | `efficient_tam_memory` | **D**（空间记忆压缩） |
| plus5 / efficientsam2_sparse | `efficient_sam2_sparse` | **E**（稀疏记忆检索） |
| base | enabled=[] | baseline |

---

## 1. 关键发现（决定复用/重跑的根因）

1. **只有 2 个实验带完整插桩**：全仓库仅 `data_pd_exp_final_2000/base_last_*` 与 `A_Dtrain_Etrain_*` 两个目录有 `train_timing.jsonl`（timing / gpu_mem_peak / memory_tokens / D/E token 拆分）。
2. **所有旧消融日志均无 timing/memory/token 字段**：`grep -rl 'memory_tokens|gpu_mem_peak|compression_tokens' logs/omnisam_time_ablation*` 返回空。旧消融跑在 Task 2-2A 插桩之前，**只有 mIoU + BFScore**。
3. **旧消融协议与主结果不一致**：
   - `omnisam_time_ablation/` 系列：PL 子集 **50 张**。
   - `omnisam_time_ablation_seed1_triplet/` 系列：PL 子集 **200 张**。
   - 正式主结果：PL 子集 **2000 张**。
   - 样本量差 10–40×，mIoU 不可直接与主表并列。

---

## 2. 复核表

| Config | Old mIoU | Old BFScore | Has timing? | Has peak memory? | Has memory tokens? | Has compression stats? | Has sparse stats? | Protocol confirmed? | Can reuse? | Need rerun? | Priority |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_omnisam (base_last, final_2000) | 60.26 | 39.97 | ✅ | ✅ 26.82GB | ✅ 32768 | ✅ null(off) | ✅ null(off) | ✅ 2000/seed1/ep1 | ✅ 复用 | ❌ | — (已是主结果) |
| A_Dtrain_Etrain (final_2000) | 60.16 | 39.86 | ✅ | ✅ 25.34GB | ✅ 4096 | ✅ 0.25 | ✅ 0.5 | ✅ 2000/seed1/ep1 | ✅ 复用 | ❌ | — (已是主结果) |
| A_ccpm (A-only) | 60.16¹ | 39.33¹ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠ 50张/旧协议 | ❌ | ✅ | **P1** |
| Dtrain_fsmc (D-only) | 59.98¹ | 38.98¹ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠ 50张/旧协议 | ❌ | ✅ | **P1** |
| Etrain_sgsmr (E-only) | 59.94¹ | 38.90¹ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠ 50张/旧协议 | ❌ | ✅ | **P1** |
| A_Dtrain (A+D) | 60.20² | 39.00² | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠ 200张/旧协议 | ❌ | ✅ | **P2** |
| Dtrain_Etrain (D+E) | —³ | —³ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ 无干净旧跑 | ❌ | ✅ | **P2** |
| A_Etrain (A+E) | —³ | —³ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ 无干净旧跑 | ❌ | ✅ | P3 |
| A_Dtrain_Etrain_PLon (PL也开D/E) | —³ | —³ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ 无旧跑 | ❌ | ✅ | P3 |

注：
- ¹ 来自 `logs/omnisam_time_ablation/plugin_ablation_base_plus{1,4,5}_*_20260425_153453.log`（50 张子集）。plus5_train_only(59.94/38.91) 与 plus5(59.94/38.90) 一致。
- ² 来自 `logs/omnisam_time_ablation_seed1_triplet/plugin_ablation_base_plus1_plus4_20260425_185447.log`（200 张）；同目录 plus1_plus4_plus5_trainonly=60.09/38.96。
- ³ 无与该 config 一一对应的干净旧实验日志（或仅在 trainonly 组合中出现，无法单独拆出）。
- 所有"Old mIoU/BFScore"仅供参考，**不进入正式消融表**（协议不一致 + 无效率字段）。

---

## 3. 判断结论（按规则）

- **规则1**：baseline_omnisam、A_Dtrain_Etrain 已有正式主结果 → **复用**，不重跑。
- **规则2**：其余 7 个消融 config 旧日志**只有 mIoU/BFScore，无 timing/memory/token** → 正式消融表**必须重跑**。
- **规则3**：旧消融协议（50/200 张）与主结果（2000 张）不一致、且无统一 seed/epoch 确认 → 重跑。
- **规则4**：本批不存在"仅缺 BFScore、其他字段齐全"的情况（旧消融连 timing 都没有），故无"可复用但需补 BFScore"项。注：BFScore 旧日志里**有**（`[ValSummary]`），但因缺效率字段仍需重跑。

---

## 4. 重跑优先级（2026-06-15 修正，依 ChatGPT 建议）

> 修正要点：最小正式消融不再以"D-only/E-only 单模块"为第一优先，而是以**"A 是否为精度保护模块"+"D/E 在有/无 A 时的行为"+"PL 阶段关闭 D/E 是否合理"**这条论证主线为第一优先。D-only / E-only 仍有价值，但降为 P1。

### 已有两行复用（不重跑）

| Config | 状态 |
|---|---|
| baseline_omnisam | existing_verified_main_result（对照基线） |
| A_Dtrain_Etrain | existing_verified_main_result（完整方法） |

### P0 最小正式消融（5 个，优先重跑）

| # | Config | 论证目的 |
|---|---|---|
| 1 | A_ccpm | 证明 **A 是精度保护模块**（仅 loss 掩码，D/E off）：相对 baseline 看 A 单独对精度的作用 |
| 2 | A_Dtrain | 证明 **D 在 A 保护下的贡献**：A+D vs A_Dtrain_Etrain / vs A_ccpm，隔离 D 的效率收益与精度影响 |
| 3 | A_Etrain | 证明 **E 在 A 保护下的贡献**：A+E 同理隔离 E |
| 4 | Dtrain_Etrain | 证明 **没有 A 时 D/E 是否伤精度**：D+E 无 A，对比 A_Dtrain_Etrain，验证 A 的保护必要性 |
| 5 | A_Dtrain_Etrain_PLon | 证明 **PL 阶段关闭 D/E 是合理的**：PL 也开 D/E 的对照，验证"PL 不开 D/E"决策（精度/稳定性是否更差） |

论证主线：
- **A_ccpm** → A 是精度保护模块（A 单独对精度的影响）。
- **A_Dtrain / A_Etrain** → 在 A 保护下，D、E 各自的贡献（效率收益是否以精度为代价）。
- **Dtrain_Etrain** → 去掉 A 后 D/E 是否伤精度（反证 A 的必要性）。
- **A_Dtrain_Etrain_PLon** → PL 阶段开 D/E 的对照，证明当前"PL 关闭 D/E"是更优决策。

这 5 个 + 复用的 baseline + A_Dtrain_Etrain = **7 行**，构成最小正式消融表，完整支撑"A 保护精度、D/E 提效、PL 不开 D/E"的核心论证。

### P1 时间充足再跑（2 个）

| # | Config | 目的 |
|---|---|---|
| 1 | Dtrain_fsmc | D-only：D 完全独立（无 A 无 E）的效率/精度 |
| 2 | Etrain_sgsmr | E-only：E 完全独立的效率/精度 |

D-only / E-only 有价值（最纯粹的单模块隔离），但**不是最小正式消融的第一优先级** —— 因为主论证已由 P0 的"A 保护 + 有 A 下的 D/E + 无 A 的 D+E"覆盖；单模块纯隔离属于补强。

---

## 5. 可暂时不跑

- **baseline_omnisam、A_Dtrain_Etrain**：已是 verified 主结果，复用，不跑。
- **Dtrain_fsmc、Etrain_sgsmr（P1）**：时间充足再跑，可延后。

（注：原 §1–§3 的复用/重跑判定不变——除两个主结果外，其余 config 旧日志均缺 timing/memory/token，正式消融表一律需带插桩重跑；此处仅调整**优先级排序**。）

---

## 6. 建议执行方式（遵守分工）

- 重跑用 `run_main_experiments.sh <config>`（已验证可复现、带插桩、独立输出目录、防覆盖）。
- 训练由用户手动跑；每个约 2.5h（2000 张，PL 生成为大头）。P0 五个 ≈ 12–13h。
- 跑完贴回各自 `run_summary.json`，Claude 解析并追加进 `main_results_registry.md`（Status: pending → verified）。
- BFScore 从 `[ValSummary]` 日志行取，证据可溯。
- 手动运行命令见 `step_10b_ablation_priority_and_manual_commands.md`。

> 建议先跑 P0 五个拿到最小正式消融，再按需 P1。
