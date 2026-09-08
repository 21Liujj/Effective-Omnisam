# EI Conference Paper v2 (English)

**Title:** Efficient Panoramic Semantic Segmentation with Memory Optimization and Boundary-Aware Inference

**Target:** EI Conference (6-8 pages English)

**Status:** ✅ Complete LaTeX structure ready

---

## Paper Structure

### Sections (sections/)
- `00_abstract.tex` - Abstract (150-200 words)
- `01_introduction.tex` - Introduction (1 page)
- `02_related_work.tex` - Related Work (0.5 page)
- `03_method.tex` - Method (2 pages)
  - MTC: Memory Token Compression
  - SMR: Sparse Memory Retrieval
  - PQR: Pseudo-label Quality Refinement (auxiliary)
- `04_experiments.tex` - Experiments (1 page)
- `05_ablation.tex` - Ablation Study (1.5 pages)
- `06_discussion.tex` - Discussion (0.5 page)
- `07_conclusion.tex` - Conclusion (0.3 page)

### Tables (tables/)
- `main_results.tex` - Main comparison (Baseline vs. Ours)
- `memory_ablation.tex` - Memory module ablation
- `token_budget_tradeoff.tex` - Token budget analysis
- `pseudo_label_ablation.tex` - PQR ablation

---

## Core Results

### Main Results (Single Seed, Cityscapes → DensePASS)

| Method | mIoU | BFScore | Train Time | Iter Time | Reduction |
|--------|------|---------|------------|-----------|-----------|
| OmniSAM | 60.26 | 39.97 | 43m53s | 5.27s | - |
| Ours (MTC+SMR) | 60.16 | 39.86 | 34m03s | 4.09s | **22.4%** |

### Efficiency Gains
- **Memory compression:** 8× (32768 → 4096 tokens)
- **Attention time reduction:** -55.8% (117.5s → 51.9s)
- **Training time reduction:** -22.4%
- **Performance loss:** Only -0.10% mIoU, -0.11 BFScore

### Key Findings
1. ✅ **MTC+SMR** achieves significant training speedup with minimal accuracy loss
2. ✅ **Token budget** directly correlates with attention overhead
3. ✅ **Qualitative results** show maintained major scene structures
4. ⚠️ **PQR** provides only marginal gains (+0.05 mIoU)

---

## Building the Paper

### Requirements
- LaTeX distribution (TeX Live, MiKTeX, or Overleaf)
- IEEEtran document class
- Standard LaTeX packages (booktabs, graphicx, amsmath, etc.)

### Local Build
```bash
# Using latexmk (recommended)
latexmk -pdf main.tex

# Or manually
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Clean Build
```bash
latexmk -c  # Clean auxiliary files
latexmkrc -C  # Clean all generated files including PDF
```

---

## File Organization

```
EI_Conference_Paper_v2_EN/
├── main.tex                    # Main LaTeX document
├── refs.bib                    # BibTeX references
├── README.md                   # This file
├── sections/                   # Section files
│   ├── 00_abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_related_work.tex
│   ├── 03_method.tex
│   ├── 04_experiments.tex
│   ├── 05_ablation.tex
│   ├── 06_discussion.tex
│   └── 07_conclusion.tex
├── tables/                     # Table files
│   ├── main_results.tex
│   ├── memory_ablation.tex
│   ├── token_budget_tradeoff.tex
│   └── pseudo_label_ablation.tex
├── figures/                    # Figure files (placeholder)
│   ├── fig_qualitative_full_protocol.png
│   ├── fig_speed_accuracy_tradeoff_miou.png
│   └── fig_token_budget_attention_time.png
└── build/                      # Build output directory
```

---

## Key Contributions

1. **Memory Token Redundancy Analysis**
   - Identified and systematically analyzed redundancy in OmniSAM panoramic UDA training
   - Clarified relationship between token count and attention overhead

2. **MTC+SMR Method**
   - Memory Token Compression (MTC): Spatial aggregation to reduce structural redundancy
   - Sparse Memory Retrieval (SMR): Importance-based selection to reduce content redundancy
   - Achieved 22.4% training time reduction with minimal quality loss

3. **Comprehensive Evaluation**
   - Module ablation validates efficiency sources
   - Token budget analysis reveals compression limits
   - Qualitative comparison confirms structural preservation
   - PQR exploration establishes supervision reliability boundaries

---

## Limitations & Future Work

### Current Limitations
- Single random seed (statistical validation needed)
- Single target dataset (cross-dataset generalization untested)
- Training-only efficiency (inference acceleration not verified)
- No memory footprint analysis

### Future Directions
1. Multi-seed statistical validation
2. Cross-dataset evaluation (WildPASS, etc.)
3. Inference-time efficiency analysis
4. Boundary and small object detailed analysis
5. Adaptive token budget strategies
6. Extension to other memory-based architectures

---

## Experiment Reference

**Data Source:** `/root/autodl-tmp/OmniSAM-main/experiments/tecma/notes/full_experiment_inventory_20260903.md`

**Key Experiments:**
- Baseline multi-seed: `experiments/tecma/runs/baseline_omnisam_seed*/`
- CAMS multi-seed: `experiments/tecma/runs/cams_seed*/`
- A+D+E (MTC+SMR): `experiments/tecma/runs/A_Dtrain_Etrain_seed*/`
- Token budget ablation: `experiments/tecma/runs/token_budget/`

---

## Citation

```bibtex
@inproceedings{anonymous2026efficient,
  title={Efficient Panoramic Semantic Segmentation with Memory Optimization and Boundary-Aware Inference},
  author={Anonymous Authors},
  booktitle={Proceedings of [Conference Name]},
  year={2026}
}
```

---

## Contact

For questions about this paper, please contact: [email@example.com]

---

**Last Updated:** 2026-09-03  
**Version:** 2.0 (English Translation)  
**Status:** Ready for submission after figures are added
