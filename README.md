# 🛰️ EuroSAT: Land Use and Land Cover Classification using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Dataset](https://img.shields.io/badge/Dataset-EuroSAT-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📖 Table of Contents

- [About EuroSAT](#-about-eurosat)
- [Dataset Structure](#-dataset-structure)
- [Spectral Bands](#-spectral-bands)
- [Project Overview](#-project-overview)
- [Models & Results](#-models--results)
  - [Two-Layer CNN](#1-two-layer-cnn-baseline)
  - [ResNet-50 Pretrained (No Fine-Tuning)](#2-resnet-50-pretrained-no-fine-tuning)
  - [ResNet-50 Fine-Tuned](#3-resnet-50-fine-tuned)
- [Comparison with Paper](#-comparison-with-paper)
- [Setup & Usage](#-setup--usage)
- [References](#-references)

---

## 🌍 About EuroSAT

EuroSAT is a benchmark dataset for **Land Use and Land Cover (LULC) classification** built on top of freely available satellite imagery from the **Sentinel-2 satellite**, part of the European Space Agency's (ESA) **Copernicus Earth Observation Programme**.

### Why EuroSAT Exists

Before EuroSAT, most remote sensing datasets were either:
- **Too small** — datasets like UC Merced had only 100 images per class
- **Based on commercial imagery** — e.g., Google Earth, which cannot be used in real-world observation pipelines
- **Not georeferenced** — making it impossible to do change detection over time

EuroSAT was introduced by Helber et al. (2019) to solve all of these problems at once — providing a large-scale, freely accessible, georeferenced, multi-spectral dataset specifically designed for machine learning.

### What the Dataset Contains

EuroSAT consists of **27,000 labeled and georeferenced satellite image patches** covering **10 distinct land use and land cover classes**. The images were collected from **34 European countries**, ensuring high intra-class diversity (e.g., different types of forests, different styles of industrial buildings across countries).

Each image patch measures **64×64 pixels** at a spatial resolution of **10 meters per pixel**, meaning each patch covers a real-world area of roughly 640m × 640m.

The dataset is available in two versions:

| Version | Channels | File Format | Notes |
|---------|----------|-------------|-------|
| **RGB** | 3 (Red, Green, Blue) | `.jpg` | Standard visible-light images |
| **Multispectral (MS)** | 13 bands | `.tif` | Full Sentinel-2 spectral range |

---

## 🗂️ Dataset Structure

The dataset is organized as a standard image classification folder structure, with one subfolder per class:

```
EuroSAT_RGB/
├── AnnualCrop/          (3000 images)
├── Forest/              (3000 images)
├── HerbaceousVegetation/(3000 images)
├── Highway/             (2500 images)
├── Industrial/          (2500 images)
├── Pasture/             (2000 images)
├── PermanentCrop/       (2500 images)
├── Residential/         (3000 images)
├── River/               (2500 images)
└── SeaLake/             (3000 images)
```

### The 10 Classes

| # | Class | Description |
|---|-------|-------------|
| 1 | **Annual Crop** | Fields with seasonal crops such as wheat, corn, etc. |
| 2 | **Forest** | Dense tree cover — includes various forest types across Europe |
| 3 | **Herbaceous Vegetation** | Low-lying vegetation such as grasslands and shrublands |
| 4 | **Highway** | Major roads, motorways and surrounding infrastructure |
| 5 | **Industrial** | Factories, warehouses, industrial estates |
| 6 | **Pasture** | Open grazing land for livestock |
| 7 | **Permanent Crop** | Orchards, vineyards, olive groves — crops that persist year-round |
| 8 | **Residential** | Urban housing — apartments, suburbs, residential neighbourhoods |
| 9 | **River** | Natural river bodies and waterways |
| 10 | **Sea & Lake** | Large open water bodies — oceans, seas and lakes |

---

## 📡 Spectral Bands

The full Multispectral version of EuroSAT covers all 13 bands captured by Sentinel-2's Multispectral Imager (MSI):

| Band | Name | Resolution | Wavelength |
|------|------|-----------|------------|
| B01 | Aerosols | 60m | 443 nm |
| B02 | Blue | 10m | 490 nm |
| B03 | Green | 10m | 560 nm |
| B04 | Red | 10m | 665 nm |
| B05 | Red Edge 1 | 20m | 705 nm |
| B06 | Red Edge 2 | 20m | 740 nm |
| B07 | Red Edge 3 | 20m | 783 nm |
| B08 | NIR | 10m | 842 nm |
| B08A | Red Edge 4 | 20m | 865 nm |
| B09 | Water Vapor | 60m | 945 nm |
| B10 | Cirrus | 60m | 1375 nm |
| B11 | SWIR 1 | 20m | 1610 nm |
| B12 | SWIR 2 | 20m | 2190 nm |

In this project, we use the **RGB version** (B04, B03, B02) to stay consistent with the paper's comparative benchmark.

---

## 🧪 Project Overview

This project reimplements and extends the benchmarks from the original EuroSAT paper. We train and evaluate three models on the EuroSAT RGB dataset using an **80/20 train-test split** — identical to the paper's setup.

The goal is to understand how model complexity and transfer learning affect classification accuracy on satellite imagery:

```
Simple CNN  ──►  Pretrained ResNet (frozen)  ──►  Fine-Tuned ResNet
 (baseline)        (transfer learning,              (full adaptation
                    no adaptation)                   to EuroSAT)
```

All experiments were run on **Google Colab** using a **T4 GPU** with **PyTorch**.

### Training Setup

| Setting | Value |
|---------|-------|
| Framework | PyTorch |
| Train/Test Split | 80% / 20% |
| Batch Size | 64 |
| Image Size (CNN) | 64 × 64 |
| Image Size (ResNet) | 224 × 224 |
| Normalization (CNN) | EuroSAT RGB stats |
| Normalization (ResNet) | ImageNet stats |
| Optimizer | Adam |
| Loss Function | Cross Entropy |

---

## 📊 Models & Results

### 1. Two-Layer CNN (Baseline)

A lightweight Convolutional Neural Network trained **from scratch** on EuroSAT RGB images. This serves as the baseline to understand what a simple architecture can achieve without any pretrained knowledge.

**Architecture:**

```
Input (3 × 64 × 64)
    │
    ▼
Conv Block 1: Conv2d(3→32) → BatchNorm → ReLU → MaxPool  [→ 32×32]
    │
    ▼
Conv Block 2: Conv2d(32→64) → BatchNorm → ReLU → MaxPool [→ 16×16]
    │
    ▼
Flatten → Linear(16384→256) → ReLU → Dropout(0.5)
    │
    ▼
Linear(256→10) → Output
```

**Training Config:**

| Parameter | Value |
|-----------|-------|
| Epochs | 30 |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-4 |
| Scheduler | StepLR (step=10, γ=0.5) |
| Trainable Params | ~4.2M |

**Result: `89.35% validation accuracy`**

---

### 2. ResNet-50 Pretrained (No Fine-Tuning)

ResNet-50 loaded with weights pretrained on **ImageNet (ILSVRC-2012)** — the entire backbone is **completely frozen**. Only the final classification head (a single linear layer mapping 2048 → 10) is trained on EuroSAT.

This tests how well ImageNet features transfer to satellite imagery **without any adaptation of the backbone**.

**What is frozen vs trainable:**

```
ResNet-50 Backbone (25M params)  ←  🔒 FROZEN — ImageNet weights unchanged
        │
        ▼
   Global Avg Pool
        │
        ▼
   FC Layer: 2048 → 10           ←  🔓 TRAINED — only this layer learns
        │
        ▼
      Output
```

**Training Config:**

| Parameter | Value |
|-----------|-------|
| Backbone | Frozen (ImageNet weights) |
| Epochs | 20 |
| Learning Rate | 1e-3 (head only) |
| Scheduler | StepLR (step=7, γ=0.5) |
| Trainable Params | ~20K (head only) |

**Result: `95.13% validation accuracy`**

> **Key Insight:** Even with a completely frozen backbone never trained on satellite data, ResNet-50 achieves 95.13% — nearly 6% better than the two-layer CNN. This demonstrates the remarkable generalization power of ImageNet features, even across very different visual domains.

---

### 3. ResNet-50 Fine-Tuned

The same ResNet-50 architecture, but now the **entire network is unfrozen** and trained end-to-end on EuroSAT. This directly replicates the setup from the original paper (Helber et al., 2019).

**Fine-tuning strategy (two-phase, as described in the paper):**

```
Phase 1 — Head Warmup
  • Backbone frozen
  • Train only the FC head
  • Learning rate: 0.01
  • Run for a few epochs to stabilize the head

Phase 2 — Full Network Fine-Tuning
  • Unfreeze entire backbone
  • Train all layers end-to-end
  • Learning rate: 0.001 → 0.0001 (very low, to avoid destroying ImageNet features)
```

**Result: `____% validation accuracy`** ← *(to be updated)*

> **Paper benchmark for reference:** ResNet-50 with fine-tuning achieves **98.57%** on EuroSAT RGB with an 80/20 split (Helber et al., 2019).

---

## 📈 Comparison with Paper

| Model | Our Result | Paper Reports |
|-------|-----------|---------------|
| Two-Layer CNN | **89.35%** | ~87.96% |
| ResNet-50 (pretrained, no fine-tune) | **95.13%** | *(not reported)* |
| ResNet-50 (fine-tuned) | **____%** | **98.57%** |

> **Note:** The paper does not explicitly report a frozen-backbone result — they go straight from scratch training to full fine-tuning. Our frozen ResNet experiment fills this gap and demonstrates that ImageNet features alone account for the majority of the performance gain, with fine-tuning contributing the remaining improvement.

---

## ⚙️ Setup & Usage

### Requirements

```bash
pip install torch torchvision matplotlib numpy scikit-learn Pillow
```

### Running in Google Colab

1. Open the notebook in Google Colab
2. Mount your Google Drive or upload the EuroSAT RGB dataset
3. Set the `rgb_root` path to your dataset folder
4. Run all cells in order

### Dataset

Download EuroSAT from:
- **Kaggle:** [EuroSAT Dataset](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)
- **GitHub (official):** [https://github.com/phelber/eurosat](https://github.com/phelber/eurosat)

---

## 📚 References

```bibtex
@article{helber2019eurosat,
  title     = {EuroSAT: A Novel Dataset and Deep Learning Benchmark
               for Land Use and Land Cover Classification},
  author    = {Helber, Patrick and Bischke, Benjamin and
               Dengel, Andreas and Borth, Damian},
  journal   = {IEEE Journal of Selected Topics in Applied
               Earth Observations and Remote Sensing},
  year      = {2019},
  publisher = {IEEE}
}
```

- Helber et al. (2019) — *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification* — [arXiv:1709.00029](https://arxiv.org/abs/1709.00029)
- He et al. (2016) — *Deep Residual Learning for Image Recognition* (ResNet)
- ESA Copernicus Programme — [https://www.copernicus.eu](https://www.copernicus.eu)

---

<p align="center">
  Made with 🛰️ using PyTorch & EuroSAT
</p>
