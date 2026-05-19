# Traffic Density & Incident Classifier

A PyTorch image classification system that categorises traffic scenes into four classes — **heavy**, **incident**, **light**, and **moderate** — using a custom CNN and EfficientNet-B0 transfer learning. Includes a Streamlit app for video upload and live camera inference.

---

## Results Summary

| Model | Val Acc | Test Acc | Macro F1 |
|---|---|---|---|
| Baseline CNN | 83.6% | 86.5% | 0.862 |
| Tuned CNN | 86.1% | 89.0% | 0.880 |
| EfficientNet-B0 | **91.9%** | **93.6%** | **0.930** |

---

## Dataset

Three traffic image sources merged into a single dataset with `train / val / test` splits.

| Split | Samples |
|---|---|
| Train | 3,767 |
| Val | 438 |
| Test | 417 |

**Class distribution (train):**

| Class | Count | Loss weight |
|---|---|---|
| light | 2,124 | 0.443 |
| heavy | 743 | 1.268 |
| moderate | 690 | 1.365 |
| incident | 210 | 4.485 |

Class imbalance is handled via inverse-frequency weighting passed to `CrossEntropyLoss`. For EfficientNet training, offline augmentation physically generates 4× copies of incident images before training, raising that class to ~1,060 samples.

---

## Models

### Baseline CNN

Three convolutional blocks (32 → 64 → 128 channels), each followed by ReLU and 2×2 max-pooling. A single fully-connected layer (512 → 4) with Dropout(0.5) forms the classifier.

- Input: 128×128
- Optimizer: Adam, lr=0.001
- Epochs: 15
- Normalization: ImageNet stats

### Tuned CNN

Same convolutional backbone as the baseline, with the following improvements found via 486-configuration grid search:

| Change | Detail |
|---|---|
| BatchNorm | Added after every Conv2d and Linear layer |
| Classifier depth | 512 → 256 → 4 (extra FC layer) |
| Label smoothing | 0.1 |
| Normalization | Dataset-specific mean/std (computed from training split) |
| Augmentation | + GaussianBlur, + RandomErasing |
| LR schedule | CosineAnnealingWarmRestarts, T_0=10 |
| Epochs | 35 |

**Best hyperparameters (grid search over 486 configs, 15 epochs each):**

| Hyperparameter | Value |
|---|---|
| Learning rate | 0.001 |
| Batch size | 32 |
| Dropout rate | 0.3 |
| Optimizer | Adam |
| Weight decay | 1e-4 |
| LR schedule | CosineAnnealingWarmRestarts |

**Why the initial tuned run underperformed the baseline:**
The first retraining used ImageNet normalization on traffic images whose actual per-channel mean/std differ significantly ([0.429, 0.426, 0.426] vs [0.485, 0.456, 0.406]). There was also a bug where `optimizer.zero_grad()` was called after `scaler.step()` instead of at the top of the batch loop, causing gradient accumulation across batches. Adding BatchNorm and fixing these two issues brought the tuned model above baseline.

**3-fold cross-validation results (train split, 25 epochs):**

| Fold | Best Val Acc |
|---|---|
| 1 | 82.0% |
| 2 | 80.3% |
| 3 | 82.9% |
| **Mean** | **81.7%** |

### EfficientNet-B0

Pretrained ImageNet backbone with a custom classifier head (Dropout(0.3) + Linear(1280 → 4)), trained in two phases:

**Phase 1 — Head only (10 epochs):**
- Backbone frozen (only 5,124 / 4,012,672 parameters trainable)
- Adam, lr=1e-3, CosineAnnealingLR
- Best val acc: **86.0%**

**Phase 2 — Full fine-tuning (15 epochs):**
- All weights unfrozen
- Adam with layer-wise lr: backbone=1e-4, head=5e-4, weight_decay=1e-4
- CosineAnnealingWarmRestarts (T_0=5)
- Best val acc: **91.9%**

Preprocessing additions over baseline:
- **CLAHE** (Contrast Limited Adaptive Histogram Equalisation) applied in LAB colour space to normalise lighting across source datasets
- ColorJitter
- RandomErasing
- Input resized to 224×224 (EfficientNet-B0 native resolution)

---

## Per-Class Test Performance (EfficientNet-B0)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| heavy | 0.939 | 0.947 | 0.943 | 114 |
| incident | 1.000 | 0.964 | 0.982 | 56 |
| light | 0.973 | 0.948 | 0.961 | 271 |
| moderate | 0.802 | 0.871 | 0.835 | 93 |
| **accuracy** | | | **0.936** | **534** |

---

## Project Structure

```
Traffic_Project/
├── notebook/
│   ├── Baseline_CNN.ipynb                  # Baseline training
│   ├── Hyperparameter_Tuning_CNN.ipynb     # Grid search + tuned retraining
│   ├── EfficientNet_B0_Transfer_Learning.ipynb
│   └── model_comparison.ipynb             # ROC curves, confusion matrices
├── models/
│   ├── baseline_cnn_weights.pth
│   └── tuned_cnn_weights.pth
├── merged_dataset/
│   ├── train/
│   ├── val/
│   └── test/
├── app.py          # Streamlit app (video upload + live camera)
├── infer.py        # Single-image inference script
└── pyproject.toml
```

---

## Streamlit App

```bash
streamlit run app.py
```

Two tabs:

- **Video Upload** — upload an MP4/AVI/MOV file; predictions are overlaid on each frame as it plays at original speed. Inference runs every 4th frame.
- **Live Camera** — WebRTC real-time stream from webcam with the same overlay. Requires browser camera permission.

The app loads EfficientNet-B0 (`models/efficientnet_b0_weights.pth`) with CLAHE preprocessing and displays a label pill, confidence percentage, and per-class probability bars.

---

## Setup

```bash
uv sync          # install dependencies from pyproject.toml
```

Requires Python 3.13+. GPU (CUDA) or Apple Silicon (MPS) is used automatically if available, otherwise falls back to CPU.

---

## Training Reproducibility

All notebooks set `SEED = 42` for `torch`, `numpy`, and `random`. Weights files are saved at the end of each training run and can be loaded directly for inference without retraining.
