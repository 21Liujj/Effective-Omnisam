# Phase 4-clean Ablation Conclusion

Date: 2026-06-16  
Project: OmniSAM / TeCMA  
Status: completed, official ablation candidate

## 1. Phase 4 objective

Phase 4 aims to verify the contribution of three TeCMA modules in the OmniSAM unsupervised domain adaptation setting from Cityscapes to DensePASS:

- **A = CCPM / pseudo_conf_mask**: Confidence-Calibrated Pseudo-label Masking
- **D = FSMC / efficient_tam_memory**: FoV-aware Structured Memory Compression
- **E = SG-SMR / efficient_sam2_sparse**: Saliency-Guided Sparse Memory Retrieval

The key question is whether TeCMA can reduce memory-token volume and memory-attention training cost while preserving segmentation accuracy and boundary consistency.

## 2. Phase 4-v0 issue

Phase 4-v0 had already completed five ablation configurations, but it was later found that `REUSE_PL_DIR` was not actually propagated to the launched runs. As a result, each configuration regenerated pseudo labels independently.

Therefore:

- Phase 4-v0 is useful as preliminary trend / sanity analysis.
- Phase 4-v0 should **not** be used as the final official ablation table.
- Phase 4-v0 and Phase 4-clean should not be mixed in one formal paper table.

## 3. Clean ablation setting

Phase 4-clean fixes the above issue by explicitly reusing one canonical pseudo-label directory for all six configurations:

```text
data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1
```

Each run log contains the evidence line:

```text
Reuse fixed pseudo-label dir for epoch 1: data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1
```

Checked files per run:

- `run_summary.json`
- `run.log`
- `checkpoints/Cityscapes_13cls/sam2_s/metrics/train_timing.jsonl`

`run_summary.json` has `BFScore: null`, so BFScore was extracted from `[ValSummary] BFScore=...` in `run.log`. `pl_timing_jsonl: null` is expected because the clean ablation reused existing pseudo labels and did not regenerate PL.

## 4. Official result table

| Config | CCPM | FSMC | SG-SMR | mIoU | BFScore | Step Time | Mem. Attn. Time | Peak Mem. | Mem. Tokens |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|
| baseline_omnisam |  |  |  | 60.26 | 39.59 | 2.3505 | 117.4825 | 19.5405 | 32768 |
| A_ccpm | ✓ |  |  | 60.26 | 39.59 | 2.4089 | 117.5635 | 19.9780 | 32768 |
| A_Dtrain | ✓ | ✓ |  | 60.12 | 39.23 | 2.2938 | 60.9970 | 18.9858 | 8192 |
| A_Etrain | ✓ |  | ✓ | 60.09 | 39.23 | 2.3798 | 79.6928 | 19.2829 | 16384 |
| Dtrain_Etrain |  | ✓ | ✓ | 60.02 | 39.23 | 2.3237 | 52.3798 | 18.3393 | 4096 |
| A_Dtrain_Etrain | ✓ | ✓ | ✓ | 60.02 | 39.23 | 2.3153 | 51.9189 | 18.7768 | 4096 |

Relative to baseline:

| Config | ΔmIoU | ΔBFScore | Mem. Attn. Reduction | Token Reduction | Peak Mem. Saving |
|---|---:|---:|---:|---:|---:|
| A_ccpm | +0.00 | +0.00 | -0.1% | 1× | -0.44 GB |
| A_Dtrain | -0.14 | -0.36 | 48.1% | 4× | 0.55 GB |
| A_Etrain | -0.17 | -0.36 | 32.2% | 2× | 0.26 GB |
| Dtrain_Etrain | -0.24 | -0.36 | 55.4% | 8× | 1.20 GB |
| A_Dtrain_Etrain | -0.24 | -0.36 | 55.8% | 8× | 0.76 GB |

## 5. Main conclusion

Phase 4-clean verifies the core TeCMA hypothesis under a fair fixed-pseudo-label setting:

- D/E significantly reduce memory tokens and memory-attention time.
- The final `A_Dtrain_Etrain` configuration achieves 8× memory-token reduction.
- Memory-attention time decreases from 117.48s to 51.92s, a 55.8% reduction.
- mIoU only decreases from 60.26 to 60.02, a 0.24 drop.
- BFScore only decreases from 39.59 to 39.23, a 0.36 drop.
- This indicates that TeCMA can substantially reduce memory-attention training cost while largely preserving segmentation accuracy and boundary consistency.

## 6. Module interpretation

### A / CCPM

Under fixed high-fidelity pseudo labels, CCPM nearly preserves the baseline result:

- mIoU: 60.26 → 60.26
- BFScore: 39.59 → 39.59
- memory tokens: unchanged at 32768
- memory attention time: essentially unchanged

Therefore, CCPM should be described conservatively. It is better framed as a **conservative confidence filtering / pseudo-label stability guard**, not as a module that significantly improves mIoU in the fixed-PL clean setting.

Recommended wording:

> CCPM behaves as a conservative confidence filter under fixed high-quality pseudo labels, preserving accuracy without adding memory tokens.

Avoid wording such as “CCPM significantly improves mIoU.”

### D / FSMC

FSMC is the primary efficiency contributor. With `A_Dtrain`, memory tokens decrease from 32768 to 8192, a 4× reduction. Memory-attention time decreases from 117.48s to 61.00s, a 48.1% reduction. The mIoU drop is only 0.14.

Recommended wording:

> FSMC contributes the major efficiency gain by structurally compressing memory tokens before attention.

### E / SG-SMR

SG-SMR also reduces memory-attention cost, but the standalone gain is weaker than FSMC. With `A_Etrain`, memory tokens decrease from 32768 to 16384, a 2× reduction, and memory-attention time decreases by 32.2%.

Recommended wording:

> SG-SMR further removes redundant memory tokens and complements FSMC when combined.

### D + E

The combination of FSMC and SG-SMR reaches the strongest token compression:

- `Dtrain_Etrain`: 4096 tokens, 8× reduction, 55.4% memory-attention time reduction
- `A_Dtrain_Etrain`: 4096 tokens, 8× reduction, 55.8% memory-attention time reduction

Adding A to D+E does not noticeably change mIoU in the clean table. Therefore, the final framework can include A for pseudo-label confidence control and completeness, but the main measured efficiency gain should be attributed to D/E.

## 7. Reviewer-facing analysis

A reviewer may ask whether the accuracy is improved. The honest answer is no: under fixed pseudo labels, the final configuration slightly decreases mIoU and BFScore. The contribution should instead be positioned as an efficiency-preserving adaptation design:

- The clean table controls pseudo-label quality by using one canonical PL set.
- Under this controlled setting, D/E reduce memory-attention cost substantially.
- The accuracy and boundary metrics remain very close to baseline, with small degradation.
- This is a favorable accuracy-efficiency trade-off, not a pure accuracy improvement.

A reviewer may also ask whether A is necessary. The clean table suggests that A does not improve the fixed-PL result. Its role should be explained as stabilizing pseudo-label supervision and making the method robust in the full training pipeline, not as the dominant source of quantitative gain in this particular clean ablation.

## 8. Paper-ready conclusion

Suggested English paragraph for Experiments / Ablation:

> Table X reports the ablation study under fixed pseudo labels on Cityscapes→DensePASS. We use the same canonical pseudo-label set for all variants to isolate the effect of each TeCMA component. Under this controlled setting, CCPM preserves the baseline accuracy and boundary score, suggesting that it acts as a conservative confidence filter when high-quality pseudo labels are fixed. FSMC provides the major efficiency gain, reducing memory tokens by 4× and memory-attention time by 48.1% with only a 0.14 mIoU drop. SG-SMR also reduces redundant memory tokens, giving a 2× token reduction and a 32.2% memory-attention time reduction. Combining FSMC and SG-SMR yields the strongest compression. The full TeCMA configuration reduces memory tokens by 8× and memory-attention time by 55.8%, while incurring only 0.24 mIoU and 0.36 BFScore degradation. These results support the main claim that TeCMA substantially lowers memory-attention training cost while largely preserving segmentation accuracy and boundary consistency.

Short paper sentence:

> Under fixed pseudo labels, TeCMA reduces memory tokens by 8× and memory-attention time by 55.8%, while incurring only 0.24 mIoU and 0.36 BFScore degradation.

## 9. Mentor-report conclusion

Phase 4-clean 已完成，并且可以作为正式 ablation table 的基础。相比 baseline，最终 `A_Dtrain_Etrain` 将 memory tokens 从 32768 降到 4096，实现 8× reduction；memory attention time 从 117.48s 降到 51.92s，下降约 55.8%；mIoU 从 60.26 降到 60.02，仅下降 0.24；BFScore 从 39.59 降到 39.23，仅下降 0.36。说明 TeCMA 的核心主线成立：在基本保持语义分割精度和边界一致性的同时，可以显著降低 memory attention 训练开销。

需要注意的是，A/CCPM 在 fixed high-fidelity PL 下没有体现明显涨点，因此不应向导师或论文中表述为显著提升 mIoU。更合适的说法是：A 是 conservative confidence filtering / pseudo-label stability guard；主要效率贡献来自 D/FSMC，E/SG-SMR 提供进一步稀疏化，D+E 达到最强 token 压缩。

## 10. PLon note

`A_Dtrain_Etrain_PLon` 已经在 v0 中跑过。它本来就是测试 PL 阶段开启 D/E 的 negative ablation，因此不能复用 canonical PL。Phase 4-clean 不包含 PLon 是合理的。后续如果论文需要，可以将 PLon 单独作为 supplementary / negative ablation 报告。目前不建议把 PLon 和 fixed-PL clean table 混在同一张主表里。

## 11. Risks and limitations

1. The clean table is seed=1 only. It is suitable as the current official candidate, but multi-seed statistics would strengthen the claim if time permits.
2. The final method is an efficiency-accuracy trade-off; it does not outperform baseline in mIoU.
3. BFScore is extracted from `run.log` because `run_summary.json` currently stores `BFScore: null`.
4. `pl_timing_jsonl` is null by design in the clean setting, since pseudo labels are reused.
5. Command output / printing occasionally has formatting artifacts, but the actual clean ablation execution succeeded and the required logs exist.

## 12. Next-stage recommendations

Phase 5 should focus on complexity analysis and paper experiment writing:

1. Formalize token-complexity analysis for baseline, FSMC, SG-SMR, and FSMC+SG-SMR.
2. Convert memory-attention time and token reduction into a clear paper figure/table.
3. Write the Experiments/Ablation subsection using Phase 4-clean as the main table.
4. Keep Phase 4-v0 only as internal preliminary trend evidence.
5. If needed, place PLon as supplementary negative ablation rather than in the main fixed-PL table.
