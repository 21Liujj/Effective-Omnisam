# SCM Ablation Study Results

## Table 1: Padding Width Ablation on DensePASS Validation Set

| Method | Pad Width | Val mIoU | Δ vs Baseline | Relative Gain |
|--------|-----------|----------|---------------|---------------|
| Baseline | - | 60.51 | - | - |
| SCM | 256 | 60.60 | +0.09 | +0.15% |
| SCM | 512 | 60.72 | +0.21 | +0.35% |
| SCM | 768 | 60.80 | +0.29 | +0.48% |
| **SCM** | **1024** | **60.84** | **+0.33** | **+0.55%** |

## Key Findings

1. **Consistent Improvement**: SCM shows monotonic performance gain with increasing pad_width
2. **Optimal Configuration**: pad_width=1024 achieves best results (+0.33 mIoU)
3. **Boundary Continuity Validated**: Results confirm 360° seamless context benefits

## LaTeX Table for Paper

```latex
\begin{table}[t]
\centering
\caption{Ablation study on circular padding width. All experiments use SAM2-S backbone with memory bank on DensePASS validation set (100 images).}
\label{tab:scm_ablation}
\begin{tabular}{lcc}
\toprule
Method & Pad Width & mIoU (\%) \\
\midrule
Baseline & - & 60.51 \\
\midrule
SCM & 256 & 60.60 \\
SCM & 512 & 60.72 \\
SCM & 768 & 60.80 \\
\textbf{SCM} & \textbf{1024} & \textbf{60.84} \\
\bottomrule
\end{tabular}
\end{table}
```

## Per-Class Analysis (Baseline vs SCM pad=1024)

需要提取per-class IoU对比，特别关注边界敏感类别（traffic sign, pole等）

