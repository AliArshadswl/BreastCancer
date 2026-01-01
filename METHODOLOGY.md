# Methodology

This document describes the scientific methodology and design decisions behind the Adipose-LOASO RoPE Transformer for breast microwave sensing tumor detection.

## Table of Contents

1. [Problem Formulation](#problem-formulation)
2. [Data Description](#data-description)
3. [LOASO Cross-Validation](#loaso-cross-validation)
4. [Feature Engineering](#feature-engineering)
5. [Model Architecture](#model-architecture)
6. [Training Strategy](#training-strategy)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Ablation Studies](#ablation-studies)

---

## Problem Formulation

### Clinical Context

Breast microwave imaging (BMI) exploits the dielectric contrast between healthy and malignant breast tissue. Unlike X-ray mammography, microwave systems are:
- Non-ionizing (safer for repeated screening)
- Lower cost
- Not compression-based (more comfortable)

### Machine Learning Task

**Input**: Complex-valued S-parameter measurements from a 72-antenna array in the frequency domain.

**Output**: Binary classification
- Class 0: No tumor present
- Class 1: Tumor present

**Challenge**: Generalizing across different tissue compositions (adipose/fibroglandular ratios) without overfitting to tissue-specific artifacts.

---

## Data Description

### UM-BMID Dataset

The University of Manitoba Breast Imaging Dataset (UM-BMID) consists of measurements from anthropomorphic breast phantoms with realistic tissue-mimicking materials.

### Phantom Structure

Each phantom is identified by its composition:
- **Adipose shell**: A2, A3, A14, A16 (different fat content levels)
- **Fibroglandular component**: F1, F2, F3, etc. (glandular tissue patterns)

Example: `A2F3` = Adipose shell type 2 with fibroglandular pattern 3

### Signal Characteristics

- **Frequency range**: ~1-9 GHz
- **Antenna array**: 72 elements
- **Data format**: Complex S-parameters `S[f, antenna]`

---

## LOASO Cross-Validation

### Motivation

Standard k-fold cross-validation can leak information through tissue-specific patterns. If training and test sets contain the same adipose shell type, the model might learn shell-specific artifacts rather than tumor signatures.

### Protocol

```
For each adipose shell S ∈ {A2, A3, A14, A16}:
    Test set:      All samples with shell = S
    Train/Val set: All samples with shell ≠ S
```

This ensures:
1. **No data leakage**: Test phantoms have different tissue composition than training
2. **Generalization testing**: Model must work on unseen tissue types
3. **Clinical relevance**: Real patients have varying tissue compositions

### Validation Split

Within the training shells, we further split:
- 80% for training
- 20% for validation (early stopping)

```python
VAL_FRACTION = 0.2
# Split performed per shell to maintain class balance
```

---

## Feature Engineering

### Overview

Rather than feeding raw complex signals directly, we extract interpretable features that capture relevant signal characteristics.

### Feature Categories

#### 1. Statistical Moments

For each frequency bin, compute over the 72 antennas:

```python
features = {
    'mean': np.mean(x),
    'std': np.std(x),
    'min': np.min(x),
    'max': np.max(x),
    'ptp': np.ptp(x),  # peak-to-peak
    'q25': np.percentile(x, 25),
    'q50': np.percentile(x, 50),
    'q75': np.percentile(x, 75),
    'skew': scipy.stats.skew(x),
    'kurt': scipy.stats.kurtosis(x),
}
```

Applied to: real, imaginary, magnitude, phase components.

#### 2. Circular Differential Features

Exploit the circular antenna array geometry:

```python
def circ_diff(x):
    """First circular difference"""
    return np.roll(x, -1) - x

def circ_diff2(x):
    """Second circular difference"""
    return circ_diff(circ_diff(x))
```

These capture local spatial gradients in the antenna pattern.

#### 3. Topological Features

**Minimum Spanning Tree (MST) Length**: 
Treat complex values as 2D points, compute MST total weight.

```python
def prim_mst_total_length(points_2d):
    """Prim's algorithm for MST on complex plane"""
    # Longer MST → more spread out antenna responses
```

**Connected Components**:
Count clusters at multiple distance thresholds (ε = 10%, 20%, ..., 50% of pairwise distances).

```python
features = {
    'cc_eps10': num_components(d, eps=percentile(d, 10)),
    'cc_eps20': num_components(d, eps=percentile(d, 20)),
    ...
}
```

#### 4. Spectral Features

Low-frequency FFT coefficients of the antenna circular pattern:

```python
def circ_fft_lowfreq(x, k=4):
    """Extract first k non-DC FFT magnitudes"""
    spec = np.fft.rfft(x)
    return np.abs(spec[1:k+1])
```

These capture periodic patterns in the antenna responses.

#### 5. Phase Coherence

Circular statistics on phase values:

```python
def phase_circular_stats(ang):
    C = np.mean(np.cos(ang))
    S = np.mean(np.sin(ang))
    R = np.hypot(C, S)  # Resultant length
    return {'phase_R': R, 'phase_var': 1 - R}
```

High `phase_R` indicates coherent scattering; low values suggest random/diffuse scattering.

#### 6. Temporal Features

- **Zero-crossing rate**: Counts sign changes in antenna sequence
- **Lag-1 autocorrelation**: Measures smoothness of antenna pattern

### Feature Normalization

Features are standardized per-fold to prevent leakage:

```python
# Fit scaler on training data only
scaler = StandardScaler().fit(X_train.reshape(-1, F))
# Transform all data
X_train_scaled = scaler.transform(X_train.reshape(-1, F))
X_test_scaled = scaler.transform(X_test.reshape(-1, F))
```

---

## Model Architecture

### Transformer Design

We use a lightweight Transformer encoder for sequence (frequency bin) modeling.

```
Input: [B, T, F_features]
  ↓
Linear Projection: F_features → D_model
  ↓
[CLS] Token Prepend
  ↓
╔═══════════════════════════════════════╗
║  Encoder Layer (×2)                   ║
║  ├─ Pre-LayerNorm                     ║
║  ├─ Multi-Head Self-Attention (RoPE)  ║
║  ├─ Residual Connection + Dropout     ║
║  ├─ Pre-LayerNorm                     ║
║  ├─ FFN (Linear-GELU-Linear)          ║
║  └─ Residual Connection + Dropout     ║
╚═══════════════════════════════════════╝
  ↓
Final LayerNorm
  ↓
Extract [CLS] representation
  ↓
MLP Classification Head → 2 logits
```

### Rotary Position Embeddings (RoPE)

RoPE encodes position by rotating query and key vectors:

```python
def apply_rope(q, k, cos, sin):
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot
```

**Advantages over learned positional embeddings:**
- Better extrapolation to unseen sequence lengths
- Encodes relative position inherently
- No additional parameters

**Ablation**: Toggle with `--no_rope` flag to compare with standard MHA.

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `D_MODEL` | 192 | Balance capacity vs. overfitting |
| `N_HEAD` | 2 | Small dataset → fewer heads |
| `NUM_LAYERS` | 2 | Shallow network for limited data |
| `D_FF` | 384 | 2× hidden dimension (standard) |
| `DROPOUT` | 0.2 | Regularization |

---

## Training Strategy

### Optimizer

AdamW with decoupled weight decay:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1.5e-4,
    weight_decay=3e-4
)
```

### Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Prevents gradient explosion with small batches.

### Early Stopping

Monitor validation loss with patience:

```python
if val_loss < best_loss - 1e-6:
    best_loss = val_loss
    save_checkpoint()
    patience_counter = 0
else:
    patience_counter += 1
    if patience_counter >= PATIENCE:
        stop_training()
```

### Optimal Window Search

Instead of using the full frequency spectrum, we search for an optimal sub-window:

1. **Generate candidates**: All (start, length) combinations within bounds
2. **Score candidates**: CV-AUC with quick features (5D) and logistic regression
3. **Select per shell**: Each held-out shell gets its own optimal window

```python
# Window search bounds
WIN_SEARCH_MIN_LEN_BINS = 24   # ~3 ns
WIN_SEARCH_MAX_LEN_BINS = 96   # ~12 ns
WIN_SEARCH_STEP_BINS = 1
```

**Rationale**: Different tissue compositions may have different optimal frequency bands for tumor detection.

---

## Evaluation Metrics

### Primary Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **AUC** | Area under ROC curve | Ranking quality (threshold-independent) |
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| **F1 Score** | 2·(Pre·Rec)/(Pre+Rec) | Balance of precision/recall |

### Calibration

**Expected Calibration Error (ECE)**:

$$ECE = \sum_{b=1}^{B} \frac{|B_b|}{N} |accuracy(B_b) - confidence(B_b)|$$

Low ECE indicates well-calibrated probabilities.

### Attention Analysis

**Attention Rollout**: Tracks information flow from [CLS] token through all layers:

```python
rollout = (I + A_1) @ (I + A_2) @ ... @ (I + A_L)
cls_attention = rollout[0, 1:]  # [CLS] → all frequency tokens
```

Identifies which frequency bins the model considers most informative.

---

## Ablation Studies

### RoPE vs. Standard MHA

```bash
# With RoPE
python adipose_loaso_rope.py

# Without RoPE
python adipose_loaso_rope.py --no_rope
```

Compare AUC, calibration, and attention patterns.

### Window Length Sensitivity

Modify bounds in CONFIG:

```python
# Narrow window
cfg.WIN_SEARCH_MIN_LEN_BINS = 16
cfg.WIN_SEARCH_MAX_LEN_BINS = 48

# Wide window
cfg.WIN_SEARCH_MIN_LEN_BINS = 48
cfg.WIN_SEARCH_MAX_LEN_BINS = 128
```

### Model Capacity

Test different architectures:

```python
# Larger model
cfg.D_MODEL = 256
cfg.NUM_LAYERS = 4
cfg.N_HEAD = 4

# Smaller model
cfg.D_MODEL = 128
cfg.NUM_LAYERS = 1
cfg.N_HEAD = 2
```

### Feature Ablation

To test feature subsets, modify `extract_patch_features_one_timepoint()`:

```python
# Statistics only
features = {**safe_stats(mag), **safe_stats(ang)}

# Topology only
features = topology_summaries(z_t_72)
```

---

## References

1. **UM-BMID Dataset**: [University of Manitoba Breast Imaging Database]
2. **RoPE**: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
3. **Attention Rollout**: Abnar & Zuidema, "Quantifying Attention Flow in Transformers"
4. **ECE**: Guo et al., "On Calibration of Modern Neural Networks"
