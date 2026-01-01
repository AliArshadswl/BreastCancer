
# Adipose-LOASO RoPE Transformer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Leave-One-Adipose-Shell-Out (LOASO) cross-validation framework using a Transformer architecture with optional Rotary Position Embeddings (RoPE) for breast microwave sensing tumor detection.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Data Requirements](#data-requirements)
- [Configuration](#configuration)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Output Structure](#output-structure)
- [Citation](#citation)
- [License](#license)

## Overview

This repository implements a deep learning pipeline for binary tumor classification using frequency-domain microwave sensing data from the University of Manitoba Breast Imaging Dataset (UM-BMID). The framework employs a rigorous Leave-One-Adipose-Shell-Out (LOASO) cross-validation strategy to ensure generalization across different phantom tissue compositions.

### Problem Statement

Breast microwave imaging (BMI) offers a non-ionizing, low-cost alternative to mammography. This work addresses the challenge of tumor detection using complex-valued S-parameter measurements from a 72-antenna array, extracting rich signal features and leveraging attention mechanisms to identify discriminative frequency patterns.

## Key Features

- **LOASO Cross-Validation**: Ensures no data leakage by holding out entire adipose shell groups (A2, A3, A14, A16)
- **Optimal Window Search**: Automated frequency window selection using train-only CV scoring
- **RoPE Transformer**: Optional Rotary Position Embeddings for improved positional encoding
- **Rich Feature Extraction**: 100+ signal features per frequency bin including:
  - Statistical moments (mean, std, skew, kurtosis)
  - Topological summaries (MST length, connected components)
  - Circular FFT coefficients
  - Phase coherence metrics
- **Comprehensive Analysis Suite**: 
  - ROC curves (overall and per-shell)
  - Attention rollout visualization
  - Feature importance analysis
  - Calibration metrics (ECE)

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Dependencies

```bash
pip install numpy torch scikit-learn scipy matplotlib
```

### Clone Repository

```bash
git clone https://github.com/yourusername/adipose-loaso-transformer.git
cd adipose-loaso-transformer
```

## Data Requirements

### Input Files

The pipeline expects two pickle files:

1. **Frequency-domain data** (`fd_data_s11_adi.pickle`):
   - Shape: `[N, T, 72]` complex-valued array
   - N: Number of samples
   - T: Number of frequency bins
   - 72: Number of antenna channels

2. **Metadata** (`md_list_s11_adi.pickle`):
   - List of dictionaries, one per sample
   - Required fields:
     - `phant_id`: Phantom identifier (e.g., "A2F3")
     - `tum_diam`: Tumor diameter (NaN if no tumor)
   - Optional fields: `id`, `n_expt`, `n_session`, `adi_ref_id`, `fib_ref_id`, `fib_ang`, `ant_rad`, `ant_z`, `date`

### Data Format

```python
# Example metadata entry
{
    'phant_id': 'A2F3',      # Adipose shell A2, Fibroglandular F3
    'tum_diam': 15.0,        # Tumor diameter in mm (NaN if absent)
    'id': 1,
    'n_expt': 5,
    'n_session': 2,
    ...
}
```

## Configuration

All hyperparameters are defined in the `CONFIG` dataclass:

```python
@dataclass
class CONFIG:
    # Data paths
    DATA_PATH: str = "path/to/fd_data_s11_adi.pickle"
    METADATA_PATH: str = "path/to/md_list_s11_adi.pickle"
    
    # Model architecture
    D_MODEL: int = 192        # Transformer hidden dimension
    N_HEAD: int = 2           # Number of attention heads
    NUM_LAYERS: int = 2       # Number of encoder layers
    D_FF: int = 384           # Feed-forward dimension
    DROPOUT: float = 0.2
    USE_CLS: bool = True      # Use [CLS] token for classification
    USE_ROPE: bool = False    # Enable Rotary Position Embeddings
    
    # Training
    BATCH_SIZE: int = 32
    MAX_EPOCHS: int = 80
    PATIENCE: int = 12        # Early stopping patience
    LR: float = 1.5e-4
    WEIGHT_DECAY: float = 3e-4
    
    # Window search
    WIN_SEARCH_MIN_LEN_BINS: int = 24   # ~3.0 ns minimum
    WIN_SEARCH_MAX_LEN_BINS: int = 96   # ~12.0 ns maximum
    WIN_SEARCH_KFOLDS: int = 3          # CV folds for window scoring
```

### Key Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_ROPE` | `False` | Toggle RoPE positional encoding |
| `USE_CLS` | `True` | Use [CLS] token vs. mean pooling |
| `VAL_FRACTION` | `0.2` | Train/val split within training shells |
| `RUN_SEEDS` | `(42,43,44,45,46)` | Seeds for 5-run experiments |

## Usage

### Basic Run

```bash
python adipose_loaso_rope.py
```

### RoPE Ablation (Disable RoPE)

```bash
python adipose_loaso_rope.py --no_rope
```

### Programmatic Configuration

```python
from adipose_loaso_rope import cfg, main

# Modify configuration
cfg.DATA_PATH = "/your/data/path.pickle"
cfg.METADATA_PATH = "/your/metadata/path.pickle"
cfg.USE_ROPE = True
cfg.NUM_LAYERS = 3

# Run pipeline
main()
```

## Model Architecture

### Transformer Encoder

```
Input: [B, T, F_signal + F_meta]
    ↓
Linear Projection → [B, T, D_model]
    ↓
[CLS] Token Prepend → [B, T+1, D_model]
    ↓
┌─────────────────────────────────┐
│  Encoder Layer (× NUM_LAYERS)  │
│  ├─ LayerNorm                   │
│  ├─ Multi-Head Attention        │
│  │   └─ (Optional RoPE)         │
│  ├─ Residual + Dropout          │
│  ├─ LayerNorm                   │
│  ├─ Feed-Forward (GELU)         │
│  └─ Residual + Dropout          │
└─────────────────────────────────┘
    ↓
LayerNorm
    ↓
[CLS] Token Extraction → [B, D_model]
    ↓
MLP Head → [B, 2] (logits)
```

### RoPE (Rotary Position Embeddings)

When `USE_ROPE=True`, queries and keys are rotated using sinusoidal embeddings:

```python
# Rotation applied to q, k
q_rot = q * cos + rotate_half(q) * sin
k_rot = k * cos + rotate_half(k) * sin
```

### Feature Extraction Pipeline

For each frequency bin, 100+ features are extracted from the 72-antenna complex signal:

1. **Statistical Features**: Mean, std, min, max, percentiles, skew, kurtosis for real, imaginary, magnitude, and phase components
2. **Differential Features**: First and second circular differences
3. **Spectral Features**: Low-frequency FFT coefficients (k=1,2,3,4)
4. **Topological Features**: MST length, connected components at multiple epsilon thresholds
5. **Phase Features**: Circular statistics (resultant length, circular variance)
6. **Temporal Features**: Zero-crossing rate, lag-1 autocorrelation

## Output Structure

```
analysis/
├── loaso_runs_metrics.csv       # Per-run metrics (AUC, ACC, F1, ECE, timing)
├── freq/
│   ├── freq_global_window.png   # Full spectrum with window highlight
│   └── freq_window_mean_abs.png # Mean |Z| within selected window
├── heatmaps/
│   └── heatmap_abs_shell_*.png  # |Z| heatmaps per shell/sample
├── features/
│   ├── feature_means_by_shell.csv
│   ├── feature_cohens_d_by_shell.csv
│   ├── feature_anova_F_p.csv
│   ├── top_features_by_ANOVA.png
│   ├── pca_features_by_shell.png
│   └── profile_bins_*.png       # Feature profiles across bins
├── roc/
│   ├── roc_best_run_overall.png
│   ├── roc_best_run_shell_*.png
│   └── roc_best_run_shells_all.png  # Combined ROC figure
└── rollout_best_run/
    ├── rollout_profile_*.png    # Per-shell attention profiles
    ├── rollout_profiles_all.png # Combined rollout overlay
    └── attention_peak_tokens.csv # Peak attention bin per shell
```

### Metrics CSV Format

| Column | Description |
|--------|-------------|
| `run` | Run index (1-5) |
| `overall_auc` | Aggregate AUC across all shells |
| `{shell}_auc` | Per-shell AUC (A2, A3, A14, A16) |
| `{shell}_acc` | Per-shell accuracy |
| `{shell}_f1` | Per-shell F1 score |
| `{shell}_ece` | Per-shell Expected Calibration Error |
| `{shell}_train_time_s` | Training time in seconds |
| `{shell}_infer_ms_per_sample` | Inference latency per sample |

## LOASO Cross-Validation Strategy

The Leave-One-Adipose-Shell-Out (LOASO) protocol ensures rigorous evaluation:

```
For each adipose shell S ∈ {A2, A3, A14, A16}:
    1. Test set: All samples with adipose shell = S
    2. Train/Val set: All samples with adipose shell ≠ S
    3. Optimal window: Selected using only train shells (no test leakage)
    4. Feature scaling: Fit on train, transform all
    5. Metadata encoding: Fit on train, transform all
```

This prevents optimistic bias from tissue-specific patterns leaking into evaluation.

## Optimal Window Search

The frequency window is automatically optimized per held-out shell:

1. Generate candidate windows: `(start_idx, end_idx)` pairs within configured bounds
2. For each candidate, compute quick features (5D: mean mag, std, intra-antenna variability, dynamic range)
3. Score using k-fold CV AUC with Logistic Regression on train shells only
4. Select window with highest CV-AUC for each test shell

```python
# Window search configuration
WIN_SEARCH_MIN_LEN_BINS: 24   # Minimum window length
WIN_SEARCH_MAX_LEN_BINS: 96   # Maximum window length
WIN_SEARCH_STEP_BINS: 1       # Search stride
WIN_SEARCH_MAX_CANDIDATES: 800 # Random subsample cap
```

## Attention Rollout Analysis

Attention rollout from the [CLS] token reveals which frequency bins the model attends to:

```python
# Rollout computation
rollout = I + A_1  # Initialize with residual connection
for layer in layers[1:]:
    rollout = rollout @ (I + A_layer)
cls_attention = rollout[0, 1:]  # [CLS] → all tokens
```

The output CSV maps peak attention to:
- Token index (within window)
- Absolute frequency bin index
- Approximate frequency in Hz (if `F0_HZ` and `DF_HZ` configured)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourname2024adipose,
  title={Adipose-LOASO RoPE Transformer for Breast Microwave Imaging Tumor Detection},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2024},
  doi={10.xxxx/xxxxx}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- University of Manitoba Breast Imaging Dataset (UM-BMID)
- PyTorch team for the deep learning framework
- scikit-learn for preprocessing and evaluation utilities

## Contact

For questions or issues, please open a GitHub issue or contact [your.email@domain.com](mailto:your.email@domain.com).
