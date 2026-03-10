# Technical Analysis

**CIFAR-10 Image Classification with Transfer Learning (ResNet50)**
Masterschool Data Science & Machine Learning Program

---

## 1. Preprocessing Impact Analysis

The most significant finding in this project is the impact of preprocessing on transfer learning performance. ResNet50's `preprocess_input()` applies **caffe-style preprocessing**:

- Converts RGB → BGR channel ordering
- Subtracts per-channel ImageNet means: `R=103.939, G=116.779, B=123.68`
- Produces values in approximately `[-128, 151]`

Simple `/255.0` scaling produces values in `[0, 1]`, creating a massive distribution mismatch with the pretrained convolutional filters. This single change improved Phase 1 test accuracy from 34% to 59% — a **73% relative improvement** without any model changes.

### Before vs After Preprocessing Fix

| Metric | /255.0 Scaling | preprocess_input() | Improvement |
|--------|:--------------:|:------------------:|:-----------:|
| Phase 1 Test Accuracy | 34% | **59%** | +25 pts |
| v4 Test Accuracy | 40% | **65%** | +25 pts |
| Best Model Overall | 57% (v3) | **65% (v4)** | +8 pts |

> **Key Insight:** Preprocessing alone improved accuracy by 25 percentage points across all models. No architectural change, regularization technique, or training strategy can compensate for feeding incorrectly scaled inputs to pretrained weights. The pretrained convolutional filters expect specific input distributions — violating this expectation produces weak, noisy activations throughout the entire network.

### Why This Matters for Transfer Learning

ResNet50's first convolutional layer was trained on ImageNet data preprocessed with BGR conversion and mean subtraction. Its learned filter weights produce meaningful feature maps only when inputs match this distribution. When inputs are scaled to `[0, 1]` instead, the activation magnitudes are approximately **100x smaller** than expected, effectively reducing the pretrained features to near-zero noise.

---

## 2. Transfer Learning Strategy Comparison

### Frozen vs. Unfrozen Backbone

With correct preprocessing, the frozen backbone (Phase 1) achieved 59% test accuracy — equal to the fully unfrozen v3 (59%). However, the frozen model severely overfits (95% train), while v3's Dropout controls this (83% train). Phase 2's full unfreezing actually *decreased* test accuracy to 53%, because BatchNormalization (BN) disruption damaged the well-calibrated frozen features.

### Full vs. Partial Unfreezing

| Strategy | Layers Updated | Test Acc | Train-Test Gap | BN Disruption |
|----------|:--------------:|:--------:|:--------------:|:-------------:|
| Frozen (Phase 1) | Head only (~700 params) | 59% | 36 pts | None |
| Full Unfreeze (Phase 2) | All 175 layers | 53% | 32 pts | Moderate (val_loss=2.83) |
| Full Unfreeze + Dropout (v3) | All 175 layers | 59% | 24 pts | Mild (val_loss=2.56) |
| **Partial Unfreeze + Aug (v4)** | **Last 35 layers** | **65%** | **13 pts** | **Minimal** |

> **BatchNormalization Disruption Explained:** When a frozen model is unfrozen, the BN layers begin updating their running mean and variance statistics. These statistics were calibrated during ImageNet pretraining and may not match the CIFAR-10 distribution. The sudden shift in normalization statistics causes the training accuracy to temporarily collapse (Phase 2 dropped to 28% at epoch 1 before recovering). Partial unfreezing avoids this by keeping the earlier BN layers frozen, preserving their well-calibrated statistics.

### Why Phase 2 Underperformed Phase 1

A common assumption is that fine-tuning should always improve upon a frozen model. In our case, Phase 2 (53%) actually performed *worse* than Phase 1 (59%). This occurred because:

- The frozen features were already well-suited for CIFAR-10 classification (59% accuracy)
- Full unfreezing disrupted BN statistics, temporarily corrupting feature maps
- With only 10 epochs and 10k samples, the model couldn't fully recover from this disruption
- The low learning rate (1e-5) meant slow recovery, while the gap widened in early epochs

---

## 3. Regularization Effectiveness

### Dropout(0.5)

Applied after GlobalAveragePooling2D in v3 and v4. Reduced the train-test gap from 36 points (Phase 1) to 24 points (v3) — a **33% reduction in overfitting**.

However, Dropout alone did *not* improve absolute test accuracy (both Phase 1 and v3 achieved 59%). It simply made the model more honest about its uncertainty — preventing memorization without improving generalization.

### Data Augmentation

Applied in v4: `RandomFlip`, `RandomRotation(0.1)`, `RandomZoom(0.1)`. This was the technique that **broke through the 59% ceiling**, pushing v4 to 65%.

Augmentation effectively expanded the 10,000-image dataset by presenting varied versions of each image during training, forcing the model to learn more robust features rather than memorizing pixel patterns.

### Callback Strategies

| Callback | Configuration | Triggered? | Effect |
|----------|:-------------:|:----------:|--------|
| ReduceLROnPlateau | factor=0.5, patience=2 | Yes — epochs 14 & 17 | Reduced LR: 1e-4 → 5e-5 → 2.5e-5 |
| EarlyStopping | patience=3, restore_best | Yes — saved epoch 11 | Restored best weights (val_loss=1.04) |

> **Key Insight:** ReduceLROnPlateau and EarlyStopping worked synergistically in v4. When validation loss plateaued, the LR was halved automatically, allowing finer weight updates. When further reduction no longer helped, EarlyStopping restored the checkpoint from the best epoch. This automated the training duration and learning rate — two hyperparameters that are notoriously difficult to tune manually.

---

## 4. Per-Class Performance Analysis

Test F1-scores across all five model iterations (highest per class in bold):

| Class | Phase 1 | Phase 2 | v3 | v4 (Best) | MobileNetV2 |
|-------|:-------:|:-------:|:--:|:---------:|:-----------:|
| airplane | 0.63 | 0.53 | 0.59 | **0.68** | 0.44 |
| automobile | 0.66 | 0.57 | 0.70 | **0.71** | 0.54 |
| bird | 0.51 | 0.46 | 0.49 | **0.57** | 0.24 |
| cat | 0.40 | 0.36 | 0.43 | **0.49** | 0.27 |
| deer | 0.53 | 0.48 | 0.54 | **0.56** | 0.37 |
| dog | 0.53 | 0.47 | 0.54 | **0.60** | 0.40 |
| frog | 0.66 | 0.59 | 0.66 | **0.69** | 0.49 |
| horse | 0.58 | 0.57 | 0.64 | **0.70** | 0.38 |
| ship | 0.69 | 0.61 | 0.67 | **0.73** | 0.50 |
| truck | 0.67 | 0.62 | 0.68 | **0.70** | 0.42 |

### Class Difficulty Tiers

**Easy Classes (F1 > 0.65 in v4):**
- **Ship** (0.73) — distinctive shape + water background
- **Automobile** (0.71) — consistent rectangular shape
- **Truck** (0.70) — larger body than cars, distinctive
- **Horse** (0.70) — unique body profile
- **Frog** (0.69) — distinctive green color + shape
- **Airplane** (0.68) — sky background helps

**Hard Classes (F1 < 0.60 in v4):**
- **Dog** (0.60) — confused with cat, diverse breeds
- **Bird** (0.57) — confused with airplane, diverse poses
- **Deer** (0.56) — confused with horse, similar body shape
- **Cat** (0.49) — hardest class, confused with dog/deer

> **Key Insight:** v4 achieves the highest F1 on 9 out of 10 classes. The "cat" class has the lowest maximum F1 (0.49) across all models, with the primary confusion being with "dog." At 32×32 pixels, both animals appear as small furry blobs with similar color distributions — this is a fundamental resolution limitation, not a model limitation.

---

## 5. Architecture Comparison: ResNet50 vs MobileNetV2

| Property | ResNet50 | MobileNetV2 |
|----------|:--------:|:-----------:|
| Total Parameters | ~23.5M | ~3.4M |
| Architecture Type | Residual blocks (skip connections) | Inverted residuals (depthwise separable) |
| Best Test Accuracy | **65% (v4)** | 41% |
| Training Speed | ~8-12s/epoch | ~4s/epoch |
| Overfitting Risk | High (needs regularization) | Low (inherently constrained) |
| Preprocessing Required | RGB→BGR + ImageNet mean subtraction | Scale to [-1, 1] |
| Preprocessing Used | ✅ Correct (ResNet50's) | ⚠️ Wrong (received ResNet50's) |

> **Important Caveat:** MobileNetV2's lower accuracy (41%) is partly attributable to receiving ResNet50's `preprocess_input()` instead of its own. MobileNetV2 expects pixels scaled to `[-1, 1]` via `tensorflow.keras.applications.mobilenet_v2.preprocess_input()`, not BGR conversion with ImageNet mean subtraction. A fair comparison would require architecture-specific preprocessing for each model.

### When to Choose Each Architecture

**Choose ResNet50 when:**
- Maximum accuracy is the priority
- GPU resources are available (Colab T4 is sufficient)
- You have enough data + regularization to control overfitting
- Inference speed is not critical

**Choose MobileNetV2 when:**
- Deployment target is mobile/edge devices
- Training time and compute budget are limited
- Model size matters (3.4M vs 23.5M params)
- Low overfitting risk is preferred over peak accuracy

---

## 6. Computational Considerations

| Model | Trainable Params | Approx. Speed | Epochs Run | Total Time |
|-------|:----------------:|:-------------:|:----------:|:----------:|
| Phase 1 (Frozen) | ~700 (head only) | ~2-3s/epoch | 10 | ~30s |
| Phase 2 (Full Unfreeze) | ~23.5M | ~8-9s/epoch | 10 | ~90s |
| v3 (Full Unfreeze) | ~23.5M | ~8-9s/epoch | 20 | ~180s |
| v4 (Partial Unfreeze) | ~7M (last 35 layers) | ~11-12s/epoch | 20 (stopped at 11) | ~220s |
| MobileNetV2 | ~3.4M | ~4s/epoch | 20 | ~80s |

> **Note:** v4 is slightly slower per-epoch than fully unfrozen models despite having fewer trainable parameters, because data augmentation adds per-batch computation. However, EarlyStopping at epoch 11 reduced total training time. The entire pipeline (all 5 models) completes in approximately 30-40 minutes on a Google Colab T4 GPU.

---

## 7. Summary of Techniques & Their Impact

| Technique | Introduced In | Impact on Test Acc | Impact on Overfitting | Verdict |
|-----------|:-------------:|:------------------:|:---------------------:|---------|
| Correct preprocessing | All models | **+25 pts** | Changed overfitting pattern | **Essential** |
| Frozen backbone | Phase 1 | 59% baseline | 36 pt gap | Good starting point |
| Full unfreezing | Phase 2 | **-6 pts** | -4 pt gap | ⚠️ Harmful (BN disruption) |
| Dropout(0.5) | v3 | ±0 pts | **-12 pt gap** | Good for generalization |
| EarlyStopping | v3 | Preserved peak | Prevented late overfitting | Recommended always |
| **Data augmentation** | **v4** | **+6 pts** | **-11 pt gap** | **High impact** |
| **Partial unfreezing** | **v4** | **+6 pts (combined)** | **Avoided BN disruption** | **High impact** |
| ReduceLROnPlateau | v4 | Fine-tuned convergence | Stabilized training | Recommended for fine-tuning |

---

## Recommendations for Future Work

### High-Impact Improvements
1. **Use all 50,000 training samples** — most impactful change for reducing overfitting and boosting accuracy
2. **Fix MobileNetV2 preprocessing** — use `mobilenet_v2.preprocess_input()` for a fair architecture comparison
3. **Train for 100+ epochs** with EarlyStopping — v3 was still improving at epoch 20
4. **Apply v4's techniques to Phase 2** — partial unfreezing + augmentation on the same model

### Worth Exploring
1. Try **EfficientNetB0** — designed for varying input sizes, may handle 32×32 better
2. Advanced augmentation: CutMix, MixUp, ColorJitter
3. Learning rate warmup before unfreezing
4. Upsample images to 64×64 or 96×96 before feeding to model
5. Add Dropout between dense layers (not just after GAP)
6. Test-time augmentation (TTA) for inference boost
