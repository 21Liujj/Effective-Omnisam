# OmniSAM: Efficient Panoramic Segmentation with Memory-Optimized SAM2

[![License](https://img.shields.io/badge/License-Research-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

**A memory-efficient adaptation framework for panoramic semantic segmentation using SAM2**

> 🎯 **Core Achievement**: Maintaining 60.3+ mIoU and 39.8+ BFScore with **8× memory compression** (4096 tokens vs 32768 baseline)

## 📋 Table of Contents
- [Overview](#overview)
- [Key Innovations](#key-innovations)
- [Architecture](#architecture)
- [Results](#results)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Reproducing Experiments](#reproducing-experiments)
- [Project Structure](#project-structure)
- [Citation](#citation)

## 🌟 Overview

OmniSAM is a research project that extends SAM2 (Segment Anything Model 2) for efficient panoramic semantic segmentation through:

1. **Memory Token Compression (MTC)**: 4× spatial compression with minimal performance loss
2. **Panoramic Semantic Memory Budget (Pano-SMB)**: Adaptive token allocation based on semantic density
3. **Boundary-Guarded Token Selection**: Protecting boundary regions for better segmentation quality
4. **SG-DRF Integration**: Self-Guided Dense Refinement Framework for domain adaptation
5. **Class-Aware Token Protection**: Protecting rare classes (traffic lights, poles, etc.)

### Key Results

| Method | mIoU (%) | BFScore | Memory Tokens | Compression |
|--------|----------|---------|---------------|-------------|
| Baseline | 60.22 | 39.71 | 32768 | 1× |
| **OmniSAM** | **60.31** | **39.82** | **4096** | **8×** |
| **OmniSAM + SG-DRF** | **74.59** | - | 4096 | 8× |

> 📊 The SG-DRF integration achieved **74.59% mIoU** on the complete OmniSAM dataset, representing a **+24% improvement** through large-scale data and domain adaptation.

## 🚀 Key Innovations

### 1. Memory Token Compressor (MTC)
- 4× spatial pooling compression (64×64 → 32×32)
- Parameter-free, O(N) complexity
- Maintains 99.7% of baseline performance

### 2. Panoramic Semantic Memory Budget (Pano-SMB)
- FoV-level adaptive token quota allocation
- High semantic density FoVs receive more tokens
- Dynamic allocation: [700, 600, ..., 256]

### 3. Boundary-Guarded Token Selection
- Reserves 6.25%-25% quota for boundary tokens
- Remaining tokens selected by feature norm Top-K
- Directly improves BFScore and small class performance

### 4. SG-DRF (Self-Guided Dense Refinement Framework)
- Unsupervised domain adaptation framework
- Integrates with OmniSAM's memory-efficient architecture
- Achieves state-of-the-art results on panoramic datasets

### 5. Circular Memory Management (SCM)
- Advanced memory management for sequential processing
- Enables efficient processing of long panoramic sequences
- Validates memory compression effectiveness

## 🏗️ Architecture

```
┌──────────────────────┐
│  SAM2 Memory Encoder │
│  Per FoV: 4096 tokens│
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  MTC (Module D)      │  ← Innovation 1
│  4× Spatial Compress │
│  4096 → 1024/FoV     │
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│ Pano-SMB (Module E1) │  ← Innovation 2
│ Adaptive Quota       │
│ [700,600,...,256]    │
└──────────┬───────────┘
           │
    ┌──────┼──────┐
    │      │      │
┌───▼──┐ ┌─▼──┐ ┌▼────┐
│Route1│ │Rt 3│ │Rt 2 │  ← Innovations 3,4,5
│Bound │ │Cls │ │Ovlp │
└───┬──┘ └─┬──┘ └┬────┘
    └──────┴─────┘
           │
┌──────────▼───────────┐
│   SAM2 Memory Attn   │
│   4096 tokens        │
└──────────────────────┘
```

## 📊 Results

### Memory Efficiency Experiments
Detailed results on Cityscapes → DensePASS adaptation:

| Configuration | mIoU (%) | BFScore | Memory | Notes |
|--------------|----------|---------|---------|-------|
| Baseline (no compression) | 60.22 | 39.71 | 32768 | Full memory |
| MTC only | 60.01 | 39.45 | 4096 | 8× compression |
| MTC + Pano-SMB | 60.15 | 39.68 | 4096 | Adaptive allocation |
| MTC + Boundary Guard | 60.28 | 39.80 | 4096 | Boundary protection |
| **Full System** | **60.31** | **39.82** | **4096** | All modules |

### SG-DRF Integration Results
Results on complete dataset with domain adaptation:

| Method | Dataset | mIoU (%) | Improvement |
|--------|---------|----------|-------------|
| Baseline | Standard | 60.22 | - |
| SG-DRF | Standard | 62.15 | +1.93% |
| **SG-DRF + Large-scale** | **Complete** | **74.59** | **+24%** |

## 🔧 Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.7+ (for GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/21Liujj/OmniSAM.git
cd OmniSAM

# Create conda environment
conda create -n omnisam python=3.8
conda activate omnisam

# Install dependencies
pip install -r requirements.txt

# Install SAM2
cd sam
pip install -e .
cd ..
```

### Dataset Preparation

Please refer to [DATASET.md](docs/DATASET.md) for detailed dataset preparation instructions.

## 🚀 Quick Start

### 1. Download Pretrained Models

```bash
# Download OmniSAM model weights
wget https://path-to-models/omnisam_s.pth -P ckpt_zoo/

# Download SAM2 base model
wget https://path-to-models/sam2_base.pth -P ckpt_zoo/
```

### 2. Run Inference

```bash
# Run inference on DensePASS validation set
python validation.py \
    --config configs/omnisam_densepass.yaml \
    --checkpoint ckpt_zoo/omnisam_s.pth
```

### 3. Train from Scratch

```bash
# Train OmniSAM on Cityscapes
python train.py \
    --config configs/train_cityscapes.yaml \
    --output_dir outputs/omnisam_train
```

### 4. Domain Adaptation with SG-DRF

```bash
# Run UDA with SG-DRF
python run_adaptation.py \
    --config configs/sgdrf_adaptation.yaml \
    --source cityscapes \
    --target densepass
```

## 🔬 Reproducing Experiments

For detailed instructions on reproducing all experiments in the paper, see [REPRODUCE.md](docs/REPRODUCE.md).

### Core Experiments

```bash
# 1. Memory compression ablation
bash experiments/tecma/run_memory_ablation.sh

# 2. Pano-SMB configurations
bash scripts/run_pano_smb_experiments.sh

# 3. Boundary guard experiments
bash scripts/run_boundary_experiments.sh

# 4. SG-DRF integration
bash launch_sgdrf_exp_complete.sh

# 5. Full system evaluation
bash scripts/run_full_system_eval.sh
```

## 📁 Project Structure

```
OmniSAM/
├── adaptation/          # Domain adaptation modules
│   └── engine.py       # SG-DRF adaptation engine
├── configs/            # Configuration files
├── dataset/            # Dataset loaders
├── docs/               # Documentation
│   ├── SG-DRF_INTEGRATION_GUIDE.md
│   └── innovation_analysis/
├── experiments/        # Experiment scripts and results
│   ├── baseline/
│   ├── pano_smb/
│   ├── route1/        # Boundary guard
│   ├── route3/        # Class-aware
│   └── tecma/         # Memory experiments
├── model/              # Model implementations
│   ├── model.py       # Main OmniSAM model
│   └── modules/
│       ├── dynamic_scm.py      # Circular memory
│       └── sgdrf_net.py        # SG-DRF network
├── paper/              # Paper manuscripts and figures
├── sam/                # SAM2 integration
├── scripts/            # Utility scripts
├── train/              # Training modules
├── utils/              # Utility functions
├── run_adaptation.py   # Domain adaptation entry
├── train.py           # Training entry
└── validation.py      # Validation entry
```

## 📖 Documentation

- [Installation Guide](docs/INSTALL.md)
- [Dataset Preparation](docs/DATASET.md)
- [Experiment Reproduction](docs/REPRODUCE.md)
- [SG-DRF Integration Guide](docs/SG-DRF_INTEGRATION_GUIDE.md)
- [Innovation Analysis](INNOVATION_SUMMARY.md)
- [Technical Details](INNOVATION_DETAILED_EXPLANATION.md)

## 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@article{omnisam2026,
  title={OmniSAM: Efficient Panoramic Segmentation with Memory-Optimized SAM2},
  author={Your Name},
  journal={arXiv preprint},
  year={2026}
}
```

## 🙏 Acknowledgments

This project builds upon:
- [SAM2](https://github.com/facebookresearch/segment-anything-2) - Meta's Segment Anything Model 2
- [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
- [DensePASS Dataset](https://github.com/DENSE-PASS/DENSE-PASS)

## 📄 License

This project is for research purposes only. See [LICENSE](LICENSE) for details.

## 📧 Contact

For questions and discussions:
- GitHub Issues: [Create an issue](https://github.com/21Liujj/OmniSAM/issues)
- Email: [your-email@example.com]

---

**Status**: 🚧 Active Research Project | Last Updated: September 2026
