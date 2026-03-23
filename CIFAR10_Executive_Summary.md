# Executive Summary

**CIFAR-10 Image Classification with Transfer Learning (ResNet50)**
Masterschool Data Science Coursework

---

## Results at a Glance

| Model | Train Acc | Test Acc | Overfitting Gap | Key Technique |
|-------|:---------:|:--------:|:---------------:|---------------|
| Phase 1 — Frozen Head | 95% | 59% | 36 pts | Frozen ResNet50 backbone |
| Phase 2 — Fine-Tuned | 85% | 53% | 32 pts | Full backbone unfreezing |
| v3 — Regularized | 83% | 59% | 24 pts | Dropout(0.5) + Early Stopping |
| **v4 — Enhanced** ★ | **78%** | **65%** | **13 pts** | **Augmentation + Partial Unfreeze + LR Scheduling** |
| MobileNetV2 | 45% | 41% | 4 pts | Lightweight alternative architecture |

---

## Project Overview

This project classifies 32×32 RGB images from the CIFAR-10 dataset into 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) using **transfer learning with ResNet50** — a convolutional neural network pretrained on 1.2 million ImageNet images.

The core approach follows a **freeze → unfreeze** training strategy on the same model, supplemented by three additional experimental iterations to push accuracy further.

### Training Pipeline

```
CIFAR-10 (10,000 images) → preprocess_input() → ResNet50 (ImageNet weights)
  → Phase 1: Freeze backbone, train head (10 epochs)           — Core Requirement
  → Phase 2: Unfreeze same model, fine-tune at lower LR (10 epochs)  — Core Requirement
  → v3: New model + Dropout(0.5) + Early Stopping (20 epochs)       — Additional Experiment
  → v4: Augmentation + Partial Unfreezing + LR Scheduling (20 epochs) ← BEST
  → MobileNetV2: Alternative architecture comparison (20 epochs)      — Additional Experiment
```

---

## Key Findings

1. **Best model: v4 Enhanced at 64.68% test accuracy.** Data augmentation, partial unfreezing (last 35 of 175 layers), and ReduceLROnPlateau combined to achieve the highest accuracy with the smallest overfitting gap (13 points) among all ResNet50 models.

2. **Preprocessing is the single most impactful factor.** Switching from simple `/255.0` normalization to ResNet50's `preprocess_input()` improved Phase 1 test accuracy from 34% to 59% — a **25-point gain** without changing any model code.

3. **More techniques CAN mean better results — when the foundation is correct.** v4's advanced techniques only worked after fixing preprocessing. With incorrect preprocessing, the same model scored just 40%.

4. **Overfitting is the primary challenge** with only 10,000 training images and a 23.5M-parameter model. Every iteration shows a train-test gap, but v4's combination of augmentation and partial unfreezing reduced it most effectively (from 36 pts to 13 pts).

5. **Cat, bird, and dog remain the hardest classes** at 32×32 resolution across all models — a fundamental limitation of image size rather than model architecture. Cat achieves a maximum F1-score of only 0.49 even in the best model.

---

## What Worked & What Didn't

### ✓ What Worked
- **Correct preprocessing** — single biggest improvement (+25 pts)
- **Data augmentation** — broke through the 59% accuracy ceiling
- **Partial unfreezing** — avoided BN disruption while allowing backbone adaptation
- **ReduceLROnPlateau** — automatically fine-tuned convergence
- **Dropout(0.5)** — reduced overfitting gap by 12 points
- **EarlyStopping** — saved best weights from epoch 11

### ✗ Challenges & Limitations
- **Full unfreezing (Phase 2)** — BN disruption caused test accuracy to drop 59% → 53%
- **32×32 resolution** — cat/dog/bird confusion persists across ALL models
- **10K training samples** — insufficient for a 23.5M-param model
- **MobileNetV2 preprocessing** — received ResNet50's scheme instead of its own
- **Limited training epochs** — v3 was still improving at epoch 20

---

## Conclusion

This project demonstrated that **transfer learning with ResNet50 can achieve 65% test accuracy on CIFAR-10** with only 10,000 training images — a dataset that presents significant challenges due to its low 32×32 resolution.

The iterative approach revealed that **correct preprocessing and regularization techniques matter more than model complexity**. The single biggest improvement came not from architectural changes, but from aligning the input preprocessing with what the pretrained model expects.

With the full 50,000-image training set, architecture-specific preprocessing for MobileNetV2, and extended training with early stopping, accuracy could improve substantially beyond the current results.
