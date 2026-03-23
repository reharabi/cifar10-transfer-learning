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

## The Preprocessing Discovery

Before any model training, the most impactful decision was **how to preprocess the input images**. ResNet50 was pretrained on ImageNet using caffe-style preprocessing: converting RGB to BGR and subtracting per-channel means (R=103.939, G=116.779, B=123.68). This produces input values in approximately [-128, 151].

A common mistake is to simply divide by 255, scaling pixels to [0, 1]. This creates a massive distribution mismatch — the pretrained filters expect values 100x larger than what they receive, effectively turning learned features into noise.

| Preprocessing | Phase 1 Test Acc | v4 Test Acc | Best Overall |
|:-------------:|:----------------:|:-----------:|:------------:|
| `/255.0` (wrong) | 34% | 40% | 57% |
| `preprocess_input()` (correct) | 59% | 65% | 65% |
| **Improvement** | **+25 pts** | **+25 pts** | **+8 pts** |


## Step-by-Step Walkthrough

### Phase 1 — Frozen Backbone (Core Requirement)

The ResNet50 backbone was loaded with ImageNet weights and **completely frozen** (all 23.5M parameters locked). Only a small classification head was trainable: GlobalAveragePooling2D → Dense(128) → Dense(64) → Dense(10, softmax).

| Metric | Value |
|--------|:-----:|
| Train Accuracy | 95% |
| Test Accuracy | 59% |
| Overfitting Gap | 36 pts |

**Finding:** The frozen ResNet50 features are powerful enough to reach 59% even on 32×32 images — proving that ImageNet features transfer well to CIFAR-10. However, with only 10,000 training images, the model severely memorized the training data (95% train vs 59% test). The head simply learned to match training samples rather than generalize.

**Problem identified:** Severe overfitting. The frozen features are good, but the model needs to adapt to CIFAR-10 specifically.

---

### Phase 2 — Full Fine-Tuning (Core Requirement)

The **same model** from Phase 1 was unfrozen and recompiled with a 100x lower learning rate (1e-5) to gently fine-tune all layers.

| Metric | Value |
|--------|:-----:|
| Train Accuracy | 85% |
| Test Accuracy | 53% |
| Overfitting Gap | 32 pts |

**Finding:** Test accuracy **dropped from 59% to 53%** — fine-tuning actually made things worse. This happened because of **BatchNormalization (BN) disruption**: when the frozen model was unfrozen, the BN layers started updating their running statistics (mean and variance) that were calibrated on ImageNet. The sudden shift in these statistics caused training accuracy to collapse to 28% at epoch 1 before gradually recovering. With only 10 epochs and 10,000 samples, the model couldn't fully recover from this disruption.

**Problem identified:** Full unfreezing causes BN disruption that damages well-calibrated frozen features. The model needs regularization and a smarter unfreezing strategy.

---

### v3 — Regularized Model (Additional Experiment)

A fresh ResNet50 model was built with **Dropout(0.5)** after GlobalAveragePooling and **EarlyStopping** (patience=3, restore best weights). All layers were unfrozen, trained for 20 epochs.

| Metric | Value |
|--------|:-----:|
| Train Accuracy | 83% |
| Test Accuracy | 59% |
| Overfitting Gap | 24 pts |

**Finding:** Dropout reduced the overfitting gap from 36 points to 24 points — a 33% improvement in generalization. Test accuracy recovered to 59%, matching Phase 1. However, Dropout alone couldn't push accuracy **beyond** 59%. It prevented memorization but didn't help the model learn better features. Notably, EarlyStopping did not trigger — the model was still improving at epoch 20, suggesting it could benefit from more training time.

**Problem identified:** We hit a 59% ceiling. Regularization controls overfitting but doesn't improve the features themselves. We need a way to expose the model to more diverse training examples.

---

### v4 — Enhanced Model ★ BEST (Additional Experiment)

Three new techniques were applied simultaneously:
1. **Data Augmentation** — RandomFlip, RandomRotation(0.1), RandomZoom(0.1) to create varied training examples each epoch
2. **Partial Unfreezing** — Only the last 35 of 175 ResNet50 layers were unfrozen, keeping the first 140 frozen to preserve well-calibrated early features
3. **ReduceLROnPlateau** — Learning rate started at 1e-4 and automatically halved when validation loss plateaued (triggered at epochs 14 and 17)

| Metric | Value |
|--------|:-----:|
| Train Accuracy | 78% |
| Test Accuracy | **65%** |
| Overfitting Gap | **13 pts** |

**Finding:** This was the breakthrough. Data augmentation broke through the 59% ceiling by effectively expanding the training set — the model saw different versions of each image every epoch, forcing it to learn robust features instead of memorizing pixel patterns. Partial unfreezing avoided BN disruption entirely by keeping early layers (edges, textures, basic shapes) frozen while letting the deeper, more task-specific layers adapt. ReduceLROnPlateau fine-tuned convergence automatically, and EarlyStopping saved the best weights from epoch 11.

v4 achieved both the **highest accuracy (65%)** and the **smallest overfitting gap (13 pts)** among all ResNet50 models — proving that the right combination of techniques matters more than any single approach.

---

### MobileNetV2 — Architecture Comparison (Additional Experiment)

MobileNetV2 (3.4M parameters) was tested as a lightweight alternative to ResNet50 (23.5M parameters). Fully unfrozen with Dropout(0.3), trained for 20 epochs.

| Metric | Value |
|--------|:-----:|
| Train Accuracy | 45% |
| Test Accuracy | 41% |
| Overfitting Gap | 4 pts |

**Finding:** MobileNetV2 showed almost zero overfitting (4-point gap), indicating healthy learning dynamics — it was underfitting rather than overfitting, and still improving at epoch 20. However, an important caveat: MobileNetV2 received **ResNet50's preprocessing** instead of its own. MobileNetV2 expects pixels scaled to [-1, 1], not BGR conversion with mean subtraction. This preprocessing mismatch likely accounts for a significant portion of its lower performance, making this an unfair comparison. With correct preprocessing and more training epochs, results would improve.

## Key Findings

1. **Preprocessing is everything.** Using `preprocess_input()` instead of `/255.0` improved accuracy by +25 points — the single biggest factor in the entire project. Pretrained models expect specific input distributions, and violating this turns learned features into noise.

2. **Partial unfreezing > Full unfreezing.** Unfreezing all layers caused BN disruption and actually decreased accuracy (59% → 53%). Unfreezing only the last 35 of 175 layers preserved early features and produced the best results (65%).

3. **Data augmentation breaks accuracy ceilings.** Dropout and regularization control overfitting but can't push accuracy higher. Augmentation generates diverse training examples that force the model to learn more robust, generalizable features.

4. **The right combination matters more than any single technique.** v4 combined augmentation, partial unfreezing, and LR scheduling to achieve both the highest accuracy and the smallest overfitting gap. No single technique achieved this alone.

5. **Cat, bird, and dog remain the hardest classes** at 32×32 resolution. Cat achieved a maximum F1-score of only 0.49 even in the best model. This is a fundamental resolution limitation — at 32×32 pixels, these animals look like similar blurry blobs.

6. **Each architecture needs its own preprocessing.** Using ResNet50's preprocessing for MobileNetV2 was a mistake that unfairly penalized the lighter model. Transfer learning requires architecture-specific input pipelines.

## Conclusion

This project demonstrated a complete, iterative transfer learning pipeline on CIFAR-10 using ResNet50, progressing from a frozen baseline (59%) to an optimized model achieving **64.68% test accuracy** with only 10,000 training images.

The most important takeaway is that **correct preprocessing matters more than model architecture or training strategy**. A single line of code — switching from `/255.0` to `preprocess_input()` — improved accuracy by 25 points across every model. This highlights a fundamental principle of transfer learning: the pretrained weights are only useful when the input distribution matches what the model was trained on.

The iterative approach proved valuable: each step revealed a specific problem (overfitting → BN disruption → accuracy ceiling → resolution limitations) and motivated a targeted solution. The final model (v4) combined data augmentation, partial unfreezing, and learning rate scheduling to achieve the best balance of accuracy and generalization.

With the full 50,000-image training set, architecture-specific preprocessing for MobileNetV2, and extended training with early stopping, performance could improve substantially beyond the current results.

## Project Structure

| File | Description |
|------|-------------|
| [`CIFAR10_CV_Project.ipynb`](./CIFAR10_CV_Project.ipynb) | Main notebook (Google Colab) |
| [`CIFAR10_Executive_Summary.md`](./CIFAR10_Executive_Summary.md) | Executive summary of findings |
| [`CIFAR10_Technical_Analysis.md`](./CIFAR10_Technical_Analysis.md) | In-depth technical analysis |
| [`CIFAR10_Presentation.pptx`](./CIFAR10_Presentation.pptx) | Presentation slides |

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
| [**Notebook**](./CIFAR10_CV_Project.ipynb) | Complete iterative pipeline with 5 model versions, visualizations, and analysis |
| [**Executive Summary**](./CIFAR10_Executive_Summary.md) | High-level overview of results, key findings, and conclusions |
| [**Technical Analysis**](./CIFAR10_Technical_Analysis.md) | Deep-dive into preprocessing impact, transfer learning strategies, per-class performance, and architecture comparison |
| [**Presentation**](./CIFAR10_Presentation.pptx) | 14-slide summary covering the full project journey |

## Author

**Reha Rabi**

MSIT Data Science Course
