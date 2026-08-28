# Paper Ablation Table Draft

**Table X. Ablation study under fixed pseudo labels on Cityscapes→DensePASS.**

All variants reuse the same canonical pseudo-label set:

```text
data_pd_exp_final_2000/base_last_20260425_232941/DensePASS/sam2_s/Cityscapes_13cls/epoch1
```

BFScore is extracted from `[ValSummary] BFScore=...` in `run.log`, because `run_summary.json` currently records `BFScore: null`.

| Method | CCPM | FSMC | SG-SMR | mIoU ↑ | BFScore ↑ | Mem. Tokens ↓ | Mem. Attn. Time ↓ | Peak Mem. ↓ |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|
| OmniSAM baseline |  |  |  | 60.26 | 39.59 | 32768 | 117.48s | 19.54GB |
| + CCPM | ✓ |  |  | 60.26 | 39.59 | 32768 | 117.56s | 19.98GB |
| + CCPM + FSMC | ✓ | ✓ |  | 60.12 | 39.23 | 8192 | 61.00s | 18.99GB |
| + CCPM + SG-SMR | ✓ |  | ✓ | 60.09 | 39.23 | 16384 | 79.69s | 19.28GB |
| + FSMC + SG-SMR |  | ✓ | ✓ | 60.02 | 39.23 | 4096 | 52.38s | 18.34GB |
| TeCMA full | ✓ | ✓ | ✓ | 60.02 | 39.23 | 4096 | 51.92s | 18.78GB |

**Draft explanation.** Table X reports the ablation study under fixed pseudo labels on Cityscapes→DensePASS. We use the same canonical pseudo-label set for all variants to isolate the effect of each TeCMA component. Under this controlled setting, CCPM preserves the baseline mIoU and BFScore, suggesting that it behaves as a conservative confidence filter when high-quality pseudo labels are fixed. FSMC provides the major efficiency gain, reducing memory tokens by 4× and memory-attention time by 48.1% with only a 0.14 mIoU drop. SG-SMR also reduces redundant memory tokens, giving a 2× token reduction and a 32.2% memory-attention time reduction. Combining FSMC and SG-SMR yields the strongest token compression. The full TeCMA configuration reduces memory tokens by 8× and memory-attention time by 55.8%, while incurring only 0.24 mIoU and 0.36 BFScore degradation. These results support the main claim that TeCMA substantially lowers memory-attention training cost while largely preserving segmentation accuracy and boundary consistency.

**Recommended short wording.** Under fixed pseudo labels, TeCMA reduces memory tokens by 8× and memory-attention time by 55.8%, while incurring only 0.24 mIoU and 0.36 BFScore degradation.

**Do not claim.** This table should not be used to claim that CCPM significantly improves mIoU or that the full TeCMA configuration outperforms the baseline in segmentation accuracy. The correct interpretation is an accuracy-preserving efficiency trade-off.
