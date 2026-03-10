# CIFAR-10 Image Classification with Transfer Learning

> MSIT Data Science Course — Computer Vision Project

## Overview

This project uses **transfer learning with ResNet50** (pretrained on ImageNet) to classify 32×32 RGB images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) into 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

The core pipeline follows a **freeze → unfreeze** strategy on a single model, supplemented by additional experiments to optimize accuracy through regularization, data augmentation, and partial fine-tuning.

## Results

| Model | Test Accuracy | Overfitting Gap | Key Technique |
|-------|:------------:|:---------------:|---------------|
| Phase 1 — Frozen Head | 59% | 36 pts | Frozen ResNet50 backbone |
| Phase 2 — Fine-Tuned | 53% | 32 pts | Full backbone unfreezing |
| v3 — Regularized | 59% | 24 pts | Dropout(0.5) + Early Stopping |
| **v4 — Enhanced** ★ | **65%** | **13 pts** | **Augmentation + Partial Unfreeze + LR Scheduling** |
| MobileNetV2 | 41% | 4 pts | Lightweight architecture comparison |

**Best model: v4 Enhanced** — 64.68% test accuracy with only a 13-point overfitting gap.

## Key Findings

- **Preprocessing matters most:** Using ResNet50's `preprocess_input()` instead of `/255.0` normalization improved accuracy by **+25 points** across all models
- **Partial unfreezing > Full unfreezing:** Unfreezing only the last 35 of 175 layers avoided BatchNormalization disruption and achieved the best results
- **Data augmentation broke the ceiling:** RandomFlip, RandomRotation, and RandomZoom pushed accuracy from 59% to 65%
- **Cat is the hardest class:** Max F1-score of 0.49 across all models — a 32×32 resolution limitation

## Project Structure

```
├── CIFAR10_CV_Project.ipynb          # Main notebook (Google Colab)
├── CIFAR10_Executive_Summary.md      # Executive summary of findings
├── CIFAR10_Technical_Analysis.md     # In-depth technical analysis
├── CIFAR10_Presentation.pptx         # Presentation slides
└── README.md                         # This file
```

## Pipeline

```
CIFAR-10 (10,000 images)
  │
  ├── preprocess_input()  →  RGB→BGR + ImageNet mean subtraction
  │
  ├── Phase 1: Freeze ResNet50 backbone, train classification head (10 epochs)
  ├── Phase 2: Unfreeze same model, fine-tune at lower LR (10 epochs)
  │
  ├── v3: New model + Dropout(0.5) + EarlyStopping (20 epochs)
  ├── v4: Augmentation + Partial Unfreeze (last 35 layers) + LR Scheduling (20 epochs) ← BEST
  │
  └── MobileNetV2: Alternative architecture comparison (20 epochs)
```

## Tech Stack

- **Framework:** TensorFlow / Keras
- **Pretrained Models:** ResNet50, MobileNetV2 (ImageNet weights)
- **Environment:** Google Colab (T4 GPU)
- **Language:** Python 3

## How to Run

1. Upload `CIFAR10_CV_Project.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells sequentially
4. Total training time: ~30-40 minutes

## Deliverables

| File | Description |
|------|-------------|
| **Notebook** | Complete iterative pipeline with 5 model versions, visualizations, and analysis |
| **Executive Summary** | High-level overview of results, key findings, and conclusions |
| **Technical Analysis** | Deep-dive into preprocessing impact, transfer learning strategies, per-class performance, and architecture comparison |
| **Presentation** | 14-slide summary covering the full project journey |

## Author

**Reha Rabi**

MSIT Data Science Course
