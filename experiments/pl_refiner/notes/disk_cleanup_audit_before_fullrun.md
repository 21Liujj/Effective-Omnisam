# disk_cleanup_audit_before_fullrun.md
# OmniSAM 实验服务器磁盘清理审计 — 第一阶段（仅扫描，未删除任何文件）

**扫描时间**: 2026-06-25
**执行人**: Claude Code
**项目路径**: `/root/autodl-tmp/OmniSAM-main`（已确认存在）
**阶段**: Phase 1 — scan only
**是否执行删除**: **否**（未执行任何 rm / 清理）
**最终结论**: `AUDIT_DONE_NO_DELETE`

> 本阶段只运行了 `df` / `du` / `find` / `ls` 等只读命令，没有删除、移动、压缩任何文件。
> 下方所有"proposed action"均为**建议**，需用户在第二阶段逐项确认后才执行。

---

## 1. 当前磁盘占用

### 1.1 关键挂载点

| Filesystem | 挂载点 | Size | Used | Avail | Use% | 说明 |
|---|---|---:|---:|---:|---:|---|
| `/dev/md0` | `/root/autodl-tmp` | 100G | 80G | **21G** | **80%** | **数据盘 — full-run 写入这里，是本次清理目标** |
| `overlay` | `/` | 30G | 24G | 6.7G | 78% | 系统盘（项目不在此，但偏紧） |
| `AutoFS:fs1` | `/autodl-pub/data` | 10T | 4.4T | 5.7T | 44% | 公共只读数据源 |
| `/dev/sda2` | (nvidia) | 879G | 417G | 418G | 50% | — |

**结论**：数据盘 `/dev/md0` 仅剩 **21G**。即将跑的 R0/R2/R2R4 三个 full-run，每个会生成 ckpt（best+last+resume ≈ 0.7G）+ metrics，若再叠加 reuse PL（复用，不重生成）则增量可控；但叠加多 seed / 多 epoch 时风险上升。建议清理后再开跑。

### 1.2 `/root/autodl-tmp` 一级占用

```
2.2M    .autodl
64M     Depth-Anything-V2-main
4.8G    venvs
74G     OmniSAM-main      ← 项目本体
80G     总计
```

---

## 2. 最大目录（项目内）

### 2.1 项目一级目录

```
74G   OmniSAM-main 总计
├── 32G   experiments          ← 最大头
├── 14G   aux_outputs          （depth_anything_v2 预生成深度先验）
├── 13G   data                 （cityscapes 12G + DensePASS 144M + WildPASS 791M）★禁删
├── 6.7G  data_pd_exp_final_2000  ★禁删（最终确认证据）
├── 5.8G  data_extra           （WildPASS / Stanford360 raw + processed）
├── 1.6G  sam                  （SAM2 backbone + 官方 ckpt zoo）
├── 1.3G  .git
├── 90M   wandb
├── 35M   logs
├── 16M   runs                 （TensorBoard events）
└── 其余 <20M
```

### 2.2 experiments 内部（32G）

```
32G   experiments
├── 26G   tecma                ← memory-efficient 线（已完成消融）
│   ├── 24G   tecma/runs       ← 17+ 个完成 run（每个 ckpt 0.7G + 再生 PL 2.7G）
│   │         token_budget 1.4G
│   ├── 1.5G  tecma/sanity_runs
│   ├── 135M  tecma/figures    ★论文图，禁删
│   └── notes/tables/configs/eval_predictions_confirmed  ★证据，禁删
└── 6.0G   pl_refiner          ← 当前主线
    ├── 6.0G  pl_refiner/sanity_runs  ← 9 个 5-step 诊断 run 的 ckpt（全是 diagnostic artifact）
    └── notes/tables/configs/scripts/results  ★证据，禁删（合计 <1M）
```

---

## 3. 最大文件分类

### 3.1 大文件类型与总量

| 文件类型 | 单个大小 | 数量(experiments内) | 合计 | 性质 |
|---|---:|---:|---:|---|
| `resume_sam2_s.pth` | ~358M | 26 | **9.08 GB** | 优化器 resume 状态，完成 run 后无论文价值 |
| `last_sam2_s_13cls.pth` | ~178M | 25 | 4.34 GB | 末轮权重 |
| `best_iou*.pth` | ~178M | 25 | 4.34 GB | 最优权重（部分有论文价值） |
| tecma `DensePASS/` 再生 PL | ~2.7G/run | 7 | ~16-17 GB | 每 run 重生成的伪标签（可再生） |
| `sam/checkpoints/*.pt` | 309M–857M | 3 | ~1.5 GB | SAM2 官方 backbone（base/large 未必用到） |

### 3.2 SAM2 官方 ckpt（sam/checkpoints）

```
309M  sam2.1_hiera_base_plus.pt
857M  sam2.1_hiera_large.pt      ← 项目用 sam2_s(small)，base/large 可能未使用
```

### 3.3 pl_refiner sanity_runs 逐 run（全部为 5-step 诊断）

| run | 含 ckpt | 性质 |
|---|---|---|
| R0_off | best60.22 + last + resume | 诊断 artifact |
| R1_conf_mask_l3 | best60.22 + last + resume | 诊断 artifact |
| R1_identity | best60.19 + last + resume | 诊断 artifact |
| R1_identity.interrupted_20260623 | 仅 resume | **中断残留** |
| R2_class_balanced_l3 | best60.22 + last + resume | 诊断 artifact |
| R3_plq_basic_5step | best60.22 + last + resume | 诊断 artifact |
| R4_proto_gate_5step | best60.22 + last + resume | **R4-4b 诊断 artifact（step_15 已明确 60.22 不可引用）** |
| R4_proto_similarity_5step | best60.22 + last + resume | 诊断 artifact |
| R4_proto_threshold_calibration_5step | best60.22 + last + resume | 诊断 artifact |

> 这 9 个 run 的 `.pth` 全是 5-step 诊断产物。论文证据是各自的 `metrics.jsonl` + `experiments/pl_refiner/results/*.jsonl` + `notes/`，**这些 ckpt 本身无论文价值**。合计约 6.0G。

---

## 4. 文件类别标记

### A. 绝对禁止删除

| 路径 | 大小 | 原因 |
|---|---:|---|
| `data/cityscapes` | 12G | 源域原始数据集 |
| `data/DensePASS`、`data/WildPASS` | 935M | 目标域/额外数据集 |
| `data_pd_exp_final_2000/base_last_20260425_232941/.../epoch1` | (内含) | **fixed PL dir，full-run 依赖** |
| `data_pd_exp_final_2000/`（整目录） | 6.7G | 最终确认实验证据 + 最终 ckpt |
| `/root/autodl-tmp/omnisam_s.pth`（base ckpt，autodl-tmp 根） | — | source/target base checkpoint |
| `experiments/pl_refiner/{notes,tables,configs,scripts,results}` | <1M | R0–R4 全部证据链 |
| `experiments/tecma/{notes,tables,configs,figures,eval_predictions_confirmed}` | ~140M | memory-efficient 线论文证据 + 图 |
| 所有 run 的 `metrics.jsonl` / `metrics/*.jsonl` / `*.log` | 小 | 实验指标与日志证据 |
| `aux_outputs/depth_anything_v2` | 14G | 深度先验（depth_adapter 实验依赖，用途明确则保留）|

### B. 可以安全删除（低风险，仍建议确认）

| 路径 | 大小 | 原因 |
|---|---:|---|
| `__pycache__`（20 个目录） | ~2M | Python 字节码缓存，自动重建 |
| `wandb/`（34 个 offline-run 缓存） | 90M | wandb 离线缓存，指标已另存 metrics.jsonl |

### C. 建议压缩或迁移后删除（需确认）

| 路径 | 大小 | 原因 |
|---|---:|---|
| `experiments/*/**/resume_sam2_s.pth`（26 个，**不含 data_pd_exp_final_2000**） | 9.08G | 优化器 resume 状态；对应 run 已完成，不再续训则无价值 |
| `experiments/pl_refiner/sanity_runs/*/checkpoints/**/*.pth` | ~6.0G | 5-step 诊断 artifact，证据已在 metrics/results/notes |
| `experiments/tecma/runs/*/DensePASS/`（再生 PL，7 处） | ~16G | 每 run 重生成的伪标签，可再生；非 fixed PL dir |
| `experiments/tecma/runs/*/checkpoints/**/last_sam2_s_13cls.pth` | ~3G | 末轮权重；保留 best 即可时可删 |

### D. 需要用户确认（用途/价值不明确）

| 路径 | 大小 | 原因 |
|---|---:|---|
| `experiments/tecma/runs/*/checkpoints/**/best_iou*.pth`（17+ 个） | ~3G | memory-efficient 消融最优权重；是否仍需复算/出图未知 |
| `sam/checkpoints/sam2.1_hiera_large.pt` | 857M | 项目主用 sam2_s，large 是否被某实验引用待确认 |
| `sam/checkpoints/sam2.1_hiera_base_plus.pt` | 309M | 同上 |
| `data_extra/`（WildPASS/Stanford360 raw+processed） | 5.8G | 补充实验数据，是否仍需跑 WildPASS/Stanford 待确认 |
| `aux_outputs/depth_anything_v2` | 14G | 若深度先验线已弃用可迁移；当前 CLAUDE.md 标注 depth 默认 disabled、无稳定收益 |
| `.git`（1.3G，objects 1.3G） | 1.3G | `git gc --aggressive` 可能压缩，但不删历史；需确认 |

---

## 5. 清理建议表

| category | path | size | reason | proposed action | safe? |
|---|---|---:|---|---|---|
| A | `data/cityscapes` | 12G | 源域数据集 | keep | ✅ keep |
| A | `data_pd_exp_final_2000/` | 6.7G | 最终证据 + fixed PL dir | keep | ✅ keep |
| A | `/root/autodl-tmp/omnisam_s.pth` | — | base ckpt | keep | ✅ keep |
| A | `experiments/*/{notes,tables,configs,scripts,results}` | <1M+140M | 证据链/论文图 | keep | ✅ keep |
| A | 各 run `metrics.jsonl` / `*.log` | 小 | 指标证据 | keep | ✅ keep |
| A | `aux_outputs/depth_anything_v2` | 14G | 深度先验依赖（若仍用） | unknown_keep_for_now | ⚠ |
| B | `__pycache__`（20 dirs） | ~2M | 字节码缓存 | delete_after_user_confirm | ✅ safe |
| B | `wandb/`（34 缓存） | 90M | 离线缓存，指标另存 | delete_after_user_confirm | ✅ safe |
| C | `experiments/**/resume_sam2_s.pth`（26，排除 final_2000） | 9.08G | 完成 run 的优化器状态 | delete_after_user_confirm | ⚠ confirm |
| C | `experiments/pl_refiner/sanity_runs/**/*.pth` | ~6.0G | 5-step 诊断 artifact | delete_after_user_confirm | ⚠ confirm |
| C | `experiments/tecma/runs/*/DensePASS/`（再生 PL） | ~16G | 可再生伪标签 | compress_then_delete_after_user_confirm | ⚠ confirm |
| C | `experiments/tecma/runs/*/**/last_sam2_s_13cls.pth` | ~3G | 末轮权重（保留 best 时可删） | delete_after_user_confirm | ⚠ confirm |
| D | `experiments/tecma/runs/*/**/best_iou*.pth` | ~3G | 消融最优权重 | move_to_backup_after_user_confirm | ⚠ confirm |
| D | `sam/checkpoints/sam2.1_hiera_large.pt` | 857M | 主用 small，large 用途待查 | unknown_keep_for_now | ⚠ |
| D | `sam/checkpoints/sam2.1_hiera_base_plus.pt` | 309M | 同上 | unknown_keep_for_now | ⚠ |
| D | `data_extra/` | 5.8G | 补充实验数据 | unknown_keep_for_now | ⚠ |
| D | `.git` | 1.3G | git gc 可压缩，不删历史 | unknown_keep_for_now | ⚠ |

---

## 6. 可释放空间估计

| 层级 | 内容 | 估计可释放 |
|---|---|---:|
| **直接安全删除（B）** | `__pycache__` + `wandb` 缓存 | **~92 MB** |
| **用户确认后可删（C，强建议）** | resume 优化器状态 9.08G + pl_refiner 诊断 ckpt 6.0G + tecma last 权重 3G | **~18 GB** |
| **压缩/迁移后可删（C，tecma 再生 PL）** | tecma/runs/*/DensePASS 再生伪标签 | **~16 GB** |
| **进一步确认（D）** | tecma best ckpt 3G + sam large/base 1.2G + data_extra 5.8G + git gc | **~10+ GB** |

**保守可立即回收（B + C-resume + C-sanity 诊断 ckpt）**：约 **15 GB**（数据盘从 21G→~36G 可用），且**不触碰任何论文证据**（metrics/log/results/notes 全保留）。

**激进上限（含 tecma 再生 PL + D 类确认）**：可回收 **40+ GB**。

---

## 7. Full-run 前推荐保留的证据链（绝对不动）

下列构成完整论文证据，**必须原样保留**：

1. **数据与依赖**
   - `data/cityscapes`、`data/DensePASS`、`data/WildPASS`（数据集）
   - `data_pd_exp_final_2000/base_last_.../epoch1`（fixed PL，full-run 依赖）
   - `/root/autodl-tmp/omnisam_s.pth`（base ckpt）
2. **memory-efficient 线（tecma）**
   - `experiments/tecma/{notes,tables,configs,figures,eval_predictions_confirmed}`
   - 各 run 的 `metrics.jsonl` + `metrics/train_timing.jsonl` + `*.log`
   - `data_pd_exp_final_2000/A_Dtrain_Etrain_...` 与 `base_last_...`（最终对比 ckpt + 指标）
3. **pl_refiner 线（R0–R4）**
   - `experiments/pl_refiner/{notes,tables,configs,scripts,results}` 全部
   - 各 sanity_run 的 `metrics.jsonl`（ckpt 本身是 artifact，可删；指标须留）
   - `experiments/pl_refiner/results/*.jsonl`（R3/R4 诊断 jsonl）
4. **本次新建 full-run 资产**
   - `configs/R0_identity_fullrun.yaml`、`R2_class_balanced_fullrun.yaml`、`R2R4_proto_gate_fullrun.yaml`
   - `scripts/run_pl_refiner_fullrun.sh`、`notes/step_16_fullrun_spec_and_checklist.md`

---

## 8. 下一步建议

1. **第二阶段（需用户逐项确认）**优先回收顺序：
   - 先删 B 类（pycache + wandb，~92M，零风险）
   - 再删 C-resume（9.08G 优化器状态，完成 run 不续训即无价值）
   - 再删 C-sanity（6.0G pl_refiner 5-step 诊断 ckpt，保留各 metrics.jsonl）
   - → 此三步即可回收 ~15G，安全不碰证据
2. **保留 best ckpt 的策略建议**：每个 tecma run 至少保留 `best_iou*.pth` + `metrics.jsonl`，删除 `resume` + `last`，可兼顾"可复算/出图"与"省空间"。
3. **tecma 再生 PL（~16G）**：若 memory-efficient 线消融已定稿（CLAUDE.md 标注已完成），其 per-run `DensePASS/` 再生伪标签可压缩归档后删除——但建议先和 final_2000 证据交叉确认无遗漏再动。
4. **删除前务必**：对每个待删 `.pth`，确认同目录 `metrics.jsonl` 与对应 `notes/*.md` 仍在（证据完整性校验），再执行。
5. **删除操作应记录**：第二阶段建议生成一份"已删除清单"（路径+大小+时间），写入本 notes 目录，便于追溯。

---

## 9. 最终结论

```
AUDIT_DONE_NO_DELETE
```

- 已完成全盘只读扫描（df / du / find / ls），**未删除、移动、压缩任何文件**。
- 数据盘 `/dev/md0` 现 21G 可用；保守清理（B + C-resume + C-sanity）可安全回收约 **15G**，激进上限 **40+G**。
- 所有论文证据链（metrics.jsonl / *.log / results jsonl / notes / tables / configs / figures / fixed PL / final_2000 / base ckpt）均标记为禁删。
- 待第二阶段用户逐项确认后再执行实际删除。
