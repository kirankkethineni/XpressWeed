<div align="center">

# XpressWeed
### Meta-Inspired Few-Shot Adaptation for Plant Weed Segmentation Using Texture Priors

[![Paper](https://img.shields.io/badge/Paper-IEEE%20SusTech%202026-blue)](https://github.com/kirankkethineni/XpressWeed)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://kirankkethineni.github.io/XpressWeed/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)

**[Kiran K. Kethineni](mailto:kirankumar.kethineni@unt.edu) · [Rishi Raj Kanukuntla](mailto:rishirajkanukuntla@my.unt.edu) · [Saraju P. Mohanty](mailto:saraju.mohanty@unt.edu) · [Elias Kougianos](mailto:elias.kougianos@unt.edu)**

*Department of Computer Science and Engineering & Electrical Engineering, University of North Texas, USA*

**[Project Page](https://kirankkethineni.github.io/XpressWeed/) | [Paper (IEEE SusTech 2026)](#citation)**

</div>

---

## Overview

Weeds compete with crops for nutrients, water, and sunlight — causing significant yield loss. While CNNs have been applied to weed segmentation, two fundamental problems remain unsolved:

1. **Scale problem** — building one large model for all weed species is computationally infeasible
2. **Variation problem** — leaves overlap, twist, and change appearance under different lighting, making object-centric few-shot methods unreliable

**XpressWeed** solves both by reframing the problem: *treat leaves as textures, not objects*.

Leaf textures (vein patterns, surface roughness, micro-structures) are **stable across deformations, occlusions, and lighting changes**. XpressWeed pre-trains on diverse leaf textures, then uses **MAML** (Model-Agnostic Meta-Learning) to adapt to completely new weed species with only **80 images** — achieving **85% accuracy** vs **48%** for traditional MAML.

---

## The Problem in Depth

| Challenge | Why It's Hard |
|-----------|--------------|
| Leaf shape variability | Same species looks completely different when twisted, overlapping, or partially occluded |
| Lighting & color shifts | Color hue changes dramatically under field lighting — color-based methods fail |
| New weed species | Weeds evolve regionally; a model trained in one region fails in another |
| Data scarcity | Pixel-level annotation of weed images is expensive and time-consuming |
| Object-centric failure | CNNs trained for shapes generalize poorly when leaves are interleaved |

Traditional few-shot learning fails here because a handful of images **cannot cover** the enormous variation in leaf orientation, shape, and illumination.

---

## The XpressWeed Approach

The key insight: **each plant species has a distinctive texture fingerprint** that remains relatively invariant to shape deformations and occlusions.

```
Plant Leaves  ──►  Texture Representation  ──►  Robust CNN Encoder
                         (not object-centric)         (species-agnostic filters)
                                                              │
                                                              ▼
New Weed Species  ──►  MAML Adaptation  ──►  Segmentation Model
  (4 unseen)          (12 shots, 80 imgs)     (85% accuracy)
```

### Two-Stage Pipeline

```
Stage 1: Supervised Texture Pre-training
┌─────────────────────────────────────────────────────┐
│  PlantVillage (12 species) → Texture Collages       │
│  → Train FCN with Weighted Cross-Entropy Loss       │
│  → Learn stable, species-agnostic texture filters   │
│  Result: 96% train acc | 92% val acc | 92% mIoU    │
└─────────────────────────────────────────────────────┘
              │
              ▼
Stage 2: MAML-Based Few-Shot Adaptation
┌─────────────────────────────────────────────────────┐
│  Real Farmland Dataset (4 unseen weed species)      │
│  → 80 cropped patches (512×768)                     │
│  → 3-way episodes: 12 support + 10 query per class  │
│  → 25 epochs of MAML inner/outer loop updates       │
│  Result: 85% val acc | 66% mIoU                     │
└─────────────────────────────────────────────────────┘
```

**Why supervised pre-training first?** Applying meta-learning at scale requires expensive second-order differentiation over thousands of images. XpressWeed uses cheap supervised training to build the texture backbone, then applies MAML only at the lightweight adaptation stage.

**Why MAML?** Its n-way episodic training implicitly performs contrastive learning — same-class embeddings cluster together, different classes are pushed apart — without needing explicit contrastive loss functions.

---

## Network Architecture

The backbone is an encoder-decoder inspired by U-Net, but modified specifically for **texture discrimination** and **multi-scale fusion**:

```
Input (512 × 768 × 3)
        │
   ┌────▼────┐  Encoder (Progressive Downsampling)
   │ Conv Blocks + Skip Connections
   │ 512×768×3 → 256×384×64 → 128×192×128 → 64×96×256 → 32×48×256 → 16×24×512
   │                                                                      │
   │                                              Bottleneck (8×12×512) ◄─┘
   │
   ┌────▼────┐  Decoder (Progressive Upsampling + Skip Fusion)
   │ UpSample + Concat(encoder feature maps at matching scales)
   │ Multi-scale fusion → global texture context + local boundary details
   │
   └────▼────┐  Output Head
        1×1 Conv → Dense segmentation mask (512×768×N_classes)
```

Key design choices:
- **Separable convolutions** — reduce parameters for faster MAML gradient updates
- **Skip connections** — fuse high-level texture context with fine-grained edge/vein details
- **Feature fusion (not transposed conv)** — boundary refinement via concatenation, not deconvolution
- **Lightweight by design** — MAML requires second-order gradients; smaller models converge faster

---

## Results

### Stage 1: Supervised Pre-training (PlantVillage, 12 texture classes)

| Metric | Value |
|--------|-------|
| Training Accuracy | **96%** |
| Validation Accuracy | **92%** |
| Mean IoU | **92%** |
| mAP@[0.50:0.95] | **0.90** |

All 12 classes showed strong PR curves with AP values near 0.99, confirming robust texture discrimination across species.

### Stage 2: MAML Adaptation (Real Farmland, 4 unseen weed species, 80 images)

| Method | Accuracy | Mean IoU |
|--------|----------|----------|
| Traditional MAML (baseline) | 48% | 32% |
| **XpressWeed (ours)** | **85%** | **66%** |

**Per-class Average Precision after adaptation:**

| Class | AP |
|-------|----|
| Background | 0.9973 |
| Primary Crop (Soybean) | 0.9807 |
| Lavhala | 0.7747 |
| Obscure Morning Glory | 0.7614 |
| Kena | 0.6966 |
| Lamber *(fewest annotations)* | 0.4869 |

**Macro mAP: 0.783 | Micro AUPRC: 0.962**

XpressWeed achieves **37 percentage points higher accuracy** than traditional MAML using the same number of images and the same network, demonstrating that texture priors provide a fundamentally stronger starting point for few-shot adaptation.

---

## Repository Structure

```
XpressWeed/
├── maml.py          # Full MAML adaptation training script (converted from Colab)
├── MAML.ipynb       # Jupyter notebook: MAML training with 6-class farmland dataset
└── README.md        # This file
```

### Key Components in `maml.py`

| Component | Description |
|-----------|-------------|
| `color2class` | Maps RGB mask colors to class IDs (Background, Soybean, Kena, Lavhala, Lamber, Obscure Morning Glory) |
| `rgb_to_class()` | Converts RGB PIL mask images to integer class masks |
| `load_images_and_masks()` | Loads image-mask pairs with support/query set handling; supports binary union masks and class-aware masks |
| `build_per_class_images()` | Scans dataset directories, builds per-class image lists, identifies rare classes |
| `augment_images_and_masks()` | Data augmentation: flip, zoom, random shadow/highlight, color jitter, blur, noise |
| MAML inner/outer loop | Episodic 3-way, 12-shot training over 25 epochs |

### Class Color Mapping (MAML Dataset)

```python
color2class = {
    (0, 0, 0):       0,  # Background
    (61, 61, 245):   1,  # Plant data (Soybean - seen in pre-training)
    (28, 101, 232):  2,  # Kena        (unseen weed)
    (255, 53, 94):   3,  # Lavhala     (unseen weed)
    (255, 106, 77):  4,  # Lamber      (unseen weed)
    (238, 164, 15):  5,  # Obscure Morning Glory (unseen weed)
}
```

---

## Datasets

### Pre-training Dataset
- **Source:** [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Setup:** Healthy leaves from 12 species arranged into collages with overlapping leaves and background-removed variants
- **Purpose:** Learn texture priors with diverse, challenging boundaries that simulate real farmland (e.g., overlapping grasses)

### MAML Adaptation Dataset
- **Source:** [Indian Annotated Weed Dataset](https://doi.org/10.1016/j.dib.2025.111691) — Shinde & Attar, *Data in Brief*, 2025
- **Setup:** 20 high-resolution (2112×1600) farmland images → 80 cropped patches (512×768)
- **Classes:** Background, Soybean (crop), + 4 unseen weed species
- **Episodes:** 3-way classification, 12 support + 10 query samples per class

---

## Setup & Usage

### Requirements

```bash
pip install tensorflow numpy pillow scikit-learn matplotlib opencv-python
```

### Running MAML Adaptation

The primary training is done in Google Colab (requires Google Drive with datasets). The `maml.py` script contains the full pipeline:

```python
# 1. Mount Google Drive and set dataset paths
DRIVE_MOUNTED = True  # handled automatically in Colab

# 2. Build per-class image index
per_class_images, rare_classes, class_counts, unique_pairs = build_per_class_images(
    image_dir="path/to/images",
    mask_dir="path/to/masks",
    NUM_CLASSES_MAML=6
)

# 3. Load pre-trained FCN model (from Stage 1)
model = load_model("path/to/pretrained_fcn.h5")

# 4. Run MAML adaptation
# Episodes: 3-way, 12-support, 10-query, 25 epochs
# See MAML.ipynb for complete training loop
```

### MAML Episode Structure

```
Each episode:
  ├── Support Set: 3 classes × 12 samples = 36 images
  │     └── Inner loop: gradient update on support set
  └── Query Set:  3 classes × 10 samples = 30 images
        └── Outer loop: meta-update based on query loss
```

---

## Comparison with Related Work

| Method | Year | Approach | Limitation vs XpressWeed |
|--------|------|----------|--------------------------|
| Syed & Suganthi | 2023 | Fuzzy active contour | No deep features; poor scalability |
| Naik / Hu et al. | 2024/22 | U-Net / U-Net++ | Static; needs large labeled datasets |
| Shorewala et al. | 2021 | Semi-supervised ResNet-SVM | Binary only; no multi-class |
| Amac et al. | 2021 | Self-supervised MAML | Depends on saliency mask quality |
| Cao et al. | 2019 | FCN + Meta-Seg | Fails on plant image variability |
| Logeshwaran et al. | 2024 | Meta-stacking | Computationally heavy full-pipeline meta-learning |
| **XpressWeed** | **2026** | **Texture pre-training + MAML** | — Resource-efficient; 85% accuracy with 80 images |

---

## Limitations

1. **Extreme class imbalance** — very rare weed classes (e.g., Lamber with few annotated pixels) still underperform. A minimum support set is needed for reliable adaptation.

2. **Commercial deployment** — real precision-spraying systems require near-zero crop false-positive rates. XpressWeed still provides value through selective spraying compared to broadcast-spraying baselines, but confidence-based thresholding is needed before deployment.

---

## Citation

If you use XpressWeed in your research, please cite:

```bibtex
@inproceedings{kethineni2026xpressweed,
  title     = {XpressWeed: Meta-Inspired Few-Shot Adaptation for Plant Weed Segmentation Using Texture Priors},
  author    = {Kethineni, Kiran K. and Kanukuntla, Rishi Raj and Mohanty, Saraju P. and Kougianos, Elias},
  booktitle = {Proceedings of the IEEE SusTech 2026},
  year      = {2026},
  institution = {University of North Texas}
}
```

---

## Related Work from Our Group

- **WeedOut** (Kethineni et al., 2023) — Semi-supervised weed spraying in IoT smart agriculture: [Link](https://link.springer.com/chapter/10.1007/978-3-031-45878-1_28)
- **SprayCraft** (Kethineni et al., 2024) — Graph-based route optimization for variable-rate precision spraying: [arXiv:2412.12176](https://arxiv.org/abs/2412.12176)

---

## License

This project is for academic research purposes. Please contact the authors for commercial use inquiries.

---

<div align="center">
University of North Texas · Department of Computer Science and Engineering<br>
<a href="https://kirankkethineni.github.io/XpressWeed/">Project Page</a> ·
<a href="mailto:kirankumar.kethineni@unt.edu">Contact</a>
</div>
