# XpressWeed — Project Context for Claude

> **Trigger:** When user says `\temp`, load this file and resume with full context. No need to re-read the paper or re-explore the repo.

---

## Paper

- **Title:** XpressWeed: Meta-Inspired Few-Shot Adaptation for Plant Weed Segmentation Using Texture Priors
- **Venue:** IEEE SusTech 2026
- **Authors:** Kiran K. Kethineni, Rishi Raj Kanukuntla, Saraju P. Mohanty, Elias Kougianos
- **Affiliation:** University of North Texas (CSE + EE)
- **PDF:** `C:\Users\kethi\OneDrive - UNT System\Kiran_Saraju_Shared\Papers\IEEE-SusTech_2026_XpressWeed\XpressWeed.pdf`
- **Images folder:** `C:\Users\kethi\OneDrive - UNT System\Kiran_Saraju_Shared\Papers\IEEE-SusTech_2026_XpressWeed\Images_XpressWeed\` (PDFs + PPTXs — convert with PyMuPDF/fitz)

---

## Repository

- **GitHub:** https://github.com/kirankkethineni/XpressWeed/tree/main
- **Local:** `C:\Users\kethi\PycharmProjects\XpressWeed`
- **GitHub Pages:** https://kirankkethineni.github.io/XpressWeed/ (source: `main` branch, `/docs` folder)
- **Branch:** `main`

### File Structure
```
XpressWeed/
├── maml.py            # Full MAML training script (converted from Colab)
├── MAML.ipynb         # Jupyter notebook: MAML training, 6-class farmland dataset
├── README.md          # Comprehensive repo guide (built in this session)
├── CLAUDE.md          # This file
└── docs/
    ├── index.html     # GitHub Pages project site (built in this session)
    └── images/        # 11 paper figures converted PDF→PNG (2.5× zoom via fitz)
        ├── illustration.png   # Fig 1: feature landscape (2222×1494)
        ├── method.png         # Fig 2: two-stage pipeline workflow (2345×279)
        ├── network.png        # Fig 3: CNN architecture diagram (4435×10080 — very tall)
        ├── fcn_dataset.png    # Fig 4: leaf texture collage dataset (1350×1823)
        ├── fcn_pr.png         # Fig 5: PR curves supervised model (1313×1120)
        ├── fcn_map.png        # Fig 5b: mAP chart (423×395)
        ├── maml_dataset.png   # Fig 6: farmland dataset samples (1305×1037)
        ├── maml_dist.png      # Fig 7: pixel class distribution (1310×830)
        ├── maml_train.png     # Fig 8: training predictions (2030×962)
        ├── maml_test.png      # Fig 9: validation predictions (2400×1706)
        └── maml_pr.png        # Fig 10: PR curves adapted model (1350×979)
```

---

## What the Paper Is About

### Problem
- Weeds compete with crops; automated segmentation is needed
- Can't build one large CNN for all species (infeasible)
- Object-centric CNNs fail: leaves overlap, twist, change under lighting
- Traditional few-shot learning fails with only a handful of images

### Key Insight
Plant leaves are better represented as **textures** (vein patterns, surface roughness, micro-structures) than rigid objects. Textures are invariant to shape deformation, occlusion, and lighting changes.

### Solution: Two-Stage Pipeline
1. **Supervised texture pre-training** on 12 PlantVillage species arranged into collages → learns species-agnostic texture filters
2. **MAML adaptation** to 4 unseen weed species using only 80 images (12-shot, 3-way, 25 epochs)

### Architecture
- U-Net-inspired encoder-decoder with **separable convolutions** (lightweight for MAML)
- Input: 512×768×3 → Bottleneck: 8×12×512 → Output: 512×768×N_classes
- Skip connections + feature-map concatenation (not transposed conv)
- Loss: Weighted cross-entropy (handles class imbalance)

---

## Results

### Stage 1 — Pre-training (PlantVillage, 12 classes)
| Metric | Value |
|--------|-------|
| Train accuracy | 96% |
| Val accuracy | 92% |
| Mean IoU | 92% |
| mAP@[0.50:0.95] | 0.90 |

### Stage 2 — MAML Adaptation (farmland, 4 unseen weeds, 80 images)
| Method | Accuracy | mIoU |
|--------|----------|------|
| **XpressWeed** | **85%** | **66%** |
| Traditional MAML | 48% | 32% |

### Per-class AP (adapted model)
| Class | AP |
|-------|----|
| Background | 0.9973 |
| Soybean (crop, seen) | 0.9807 |
| Lavhala | 0.7747 |
| Obscure Morning Glory | 0.7614 |
| Kena | 0.6966 |
| Lamber (fewest pixels) | 0.4869 |
| **Macro mAP** | **0.783** |
| **Micro AUPRC** | **0.962** |

---

## Code Details

### maml.py key components
- `color2class` — RGB→class mapping: Background(0), Soybean(1), Kena(2), Lavhala(3), Lamber(4), Obscure Morning Glory(5)
- `rgb_to_class()` — PIL RGB mask → integer class mask
- `load_images_and_masks()` — support/query loading; union vs class-aware mask modes
- `build_per_class_images()` — scans dataset dirs, builds per-class index, finds rare classes
- `augment_images_and_masks()` — flip, zoom, RSH shadows, color jitter, blur, noise
- MAML loop: 3-way, 12 support + 10 query per class, 25 epochs

### Datasets
- **Pre-training:** [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) — 12 species collages
- **MAML:** [Indian Annotated Weed Dataset](https://doi.org/10.1016/j.dib.2025.111691) — 20 images (2112×1600) → 80 cropped patches

### Environment
- Python 3.13, TensorFlow 2.x, PyMuPDF 1.26.5 (for PDF→PNG conversion)
- Training done in Google Colab (Google Drive mount in maml.py)

---

## What Was Done in This Session

1. Read the full 7-page paper PDF
2. Explored repo structure (maml.py, MAML.ipynb)
3. Built `README.md` — comprehensive one-stop guide
4. Converted 11 paper figures from PDF→PNG using fitz (2.5× zoom) into `docs/images/`
5. Built `docs/index.html` — GitHub Pages project site matching Semantic-Search style:
   - Sticky nav, metrics bar, 9 numbered sections
   - All 11 paper figures embedded
   - Accuracy bar charts, per-class AP cards, comparison table
   - Problem/solution cards, workflow diagram, code blocks
6. Committed and pushed everything to `main` (commit `fbeb77d`)

---

## Related Work (same group)
- **WeedOut (2023):** Semi-supervised weed spraying, IoT framework — https://link.springer.com/chapter/10.1007/978-3-031-45878-1_28
- **SprayCraft (2024):** Graph-based precision spraying route optimization — https://arxiv.org/abs/2412.12176
- **Semantic-Search:** Plant disease classification with CLIP — https://kirankkethineni.github.io/Semantic-Search/ (reference design for the Pages site)

---

## To-Do / Pending
- Nothing pending from this session. GitHub Pages activation requires: repo Settings → Pages → Branch: `main`, Folder: `/docs` → Save.
