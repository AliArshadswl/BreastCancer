# API Reference

This document provides detailed documentation for all classes, functions, and modules in the Adipose-LOASO RoPE Transformer codebase.

## Table of Contents

- [Configuration](#configuration)
- [Data Loading](#data-loading)
- [Feature Extraction](#feature-extraction)
- [Model Components](#model-components)
- [Training & Evaluation](#training--evaluation)
- [Analysis & Visualization](#analysis--visualization)

---

## Configuration

### `CONFIG` (dataclass)

Central configuration object controlling all pipeline parameters.

#### Data Parameters

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `DATA_PATH` | `str` | `"..."` | Path to frequency-domain pickle file |
| `METADATA_PATH` | `str` | `"..."` | Path to metadata pickle file |
| `IS_PICKLE` | `bool` | `True` | Whether data is in pickle format |
| `SILENCE_NUMPY_PICKLE_WARNING` | `bool` | `True` | Suppress numpy deprecation warnings |

#### Timing/Sampling Parameters

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `SAMPLE_RATE_HZ` | `float` | `8e9` | Sampling rate (8 Gs/s, Δt = 0.125 ns) |
| `START_TIME_S` | `Optional[float]` | `None` | Start time for window (legacy) |
| `STOP_TIME_S` | `Optional[float]` | `None` | Stop time for window (legacy) |
| `F0_HZ` | `Optional[float]` | `1.0e9` | Starting frequency for axis labels |
| `DF_HZ` | `Optional[float]` | `8.0e6` | Frequency bin spacing |

#### Model Architecture

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `D_MODEL` | `int` | `192` | Transformer hidden dimension |
| `N_HEAD` | `int` | `2` | Number of attention heads |
| `NUM_LAYERS` | `int` | `2` | Number of encoder layers |
| `D_FF` | `int` | `384` | Feed-forward hidden dimension |
| `DROPOUT` | `float` | `0.2` | Dropout probability |
| `USE_CLS` | `bool` | `True` | Use [CLS] token for classification |
| `USE_ROPE` | `bool` | `False` | Enable Rotary Position Embeddings |

#### Training Parameters

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `BATCH_SIZE` | `int` | `32` | Training batch size |
| `MAX_EPOCHS` | `int` | `80` | Maximum training epochs |
| `PATIENCE` | `int` | `12` | Early stopping patience |
| `LR` | `float` | `1.5e-4` | Learning rate |
| `WEIGHT_DECAY` | `float` | `3e-4` | AdamW weight decay |
| `VAL_FRACTION` | `float` | `0.2` | Validation split fraction |

#### Window Search Parameters

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `WIN_SEARCH_MIN_LEN_BINS` | `int` | `24` | Minimum window length |
| `WIN_SEARCH_MAX_LEN_BINS` | `int` | `96` | Maximum window length |
| `WIN_SEARCH_STEP_BINS` | `int` | `1` | Search stride |
| `WIN_SEARCH_MAX_CANDIDATES` | `int` | `800` | Max candidate windows |
| `WIN_SEARCH_KFOLDS` | `int` | `3` | CV folds for scoring |

---

## Data Loading

### `load_complex_data(path: str, is_pickle: bool) -> np.ndarray`

Load complex-valued frequency-domain data from file.

**Parameters:**
- `path`: Path to data file (pickle, .npy, or .npz)
- `is_pickle`: Whether file is pickle format

**Returns:**
- `np.ndarray`: Complex array of shape `[N, T, 72]`

**Raises:**
- `ValueError`: If shape doesn't match `[N, T, 72]` or data isn't complex

**Example:**
```python
X = load_complex_data("data.pickle", is_pickle=True)
print(X.shape)  # (500, 1024, 72)
print(X.dtype)  # complex128
```

---

### `make_labels(meta: List[dict]) -> np.ndarray`

Generate binary labels from metadata.

**Parameters:**
- `meta`: List of metadata dictionaries

**Returns:**
- `np.ndarray`: Binary labels (0 = no tumor, 1 = tumor present)

**Logic:**
- Label 0: `tum_diam` is NaN or missing
- Label 1: `tum_diam` has a valid numeric value

---

### `extract_safe_metadata(meta: List[dict]) -> Tuple[np.ndarray, dict, np.ndarray]`

Extract leakage-safe metadata features.

**Parameters:**
- `meta`: List of metadata dictionaries

**Returns:**
- `A_comp`: Adipose shell IDs (e.g., "A2", "A3")
- `meta_struct`: Dictionary containing:
  - `F_comp`: Fibroglandular component IDs
  - `numX`: Numeric features array
  - `year`: Experiment years (if `INCLUDE_YEAR=True`)
- `phant_ids`: Full phantom IDs (e.g., "A2F3")

---

## Feature Extraction

### `extract_patch_features_one_timepoint(z_t_72: np.ndarray) -> Dict[str, float]`

Extract comprehensive features from one frequency bin across all antennas.

**Parameters:**
- `z_t_72`: Complex array of shape `[72]` (one bin, all antennas)

**Returns:**
- Dictionary with 100+ features including:

| Category | Features |
|----------|----------|
| **Statistics** | `{re,im,mag,ang}_k_{mean,std,min,max,ptp,q25,q50,q75,skew,kurt}` |
| **Differentials** | `{re,im,mag,ang}_d1_{...}`, `{...}_d2_{...}` |
| **Spectral** | `{mag,re,im}_fftk{1,2,3,4}` |
| **Topology** | `mst_len`, `cc_eps{10,20,30,40,50}` |
| **Phase** | `phase_R`, `phase_var`, `ang_unw_slope_var` |
| **Temporal** | `{re,im,mag}_zcr`, `{re,im,mag}_ac1` |
| **Energy** | `mag_energy`, `mag_entropy`, `neighbor_corr_mag` |

---

### `build_signal_seq_features(Xw: np.ndarray) -> Tuple[np.ndarray, List[str]]`

Build feature sequences for all samples and frequency bins.

**Parameters:**
- `Xw`: Complex array `[N, T, 72]`

**Returns:**
- `seq`: Feature array `[N, T, F_sig]`
- `keys`: List of feature names

---

### `quick_window_features(Xw: np.ndarray) -> np.ndarray`

Compute fast, lightweight features for window scoring.

**Parameters:**
- `Xw`: Complex array `[N, T_window, 72]`

**Returns:**
- `np.ndarray`: Shape `[N, 5]` with:
  1. Mean magnitude
  2. Std over bins
  3. Mean intra-antenna std
  4. Std of intra-antenna std
  5. Dynamic range

---

## Model Components

### `RotaryEmbedding`

Precomputes sinusoidal position embeddings for RoPE.

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        """
        Args:
            dim: Head dimension (must be even)
            max_seq_len: Maximum sequence length
            base: RoPE frequency base
        """
    
    def forward(self, seq_len: int) -> Tuple[Tensor, Tensor]:
        """Returns (cos, sin) tensors for positions 0..seq_len-1"""
```

---

### `MHA_RoPE`

Multi-head attention with optional RoPE.

```python
class MHA_RoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        rope_max_len: int = 512,
        use_rope: bool = True
    ):
        """
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dropout: Attention dropout
            rope_max_len: Max sequence for RoPE precomputation
            use_rope: Whether to apply RoPE
        """
    
    def forward(
        self,
        x: Tensor,
        return_attn: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            x: Input tensor [B, T, D]
            return_attn: Whether to return attention weights
        
        Returns:
            output: [B, T, D]
            attn (optional): [B, H, T, T]
        """
```

---

### `EncoderLayerRoPE`

Single transformer encoder layer.

```python
class EncoderLayerRoPE(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_ff: int,
        dropout: float,
        rope_max_len: int = 512,
        use_rope: bool = True
    ):
        """Standard pre-norm transformer layer with optional RoPE"""
```

**Architecture:**
```
x → LayerNorm → MHA(RoPE) → + → LayerNorm → FFN(GELU) → +
↑__________________________|   ↑_______________________|
```

---

### `SeqClassifierRoPE`

Complete sequence classification model.

```python
class SeqClassifierRoPE(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_model: int,
        nhead: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
        use_cls: bool = True,
        num_classes: int = 2,
        max_len: int = 512,
        use_rope: bool = True
    ):
        """
        Args:
            in_dim: Input feature dimension
            d_model: Hidden dimension
            nhead: Attention heads
            n_layers: Number of encoder layers
            d_ff: FFN hidden dimension
            dropout: Dropout rate
            use_cls: Use [CLS] token (vs. mean pooling)
            num_classes: Output classes
            max_len: Maximum sequence length
            use_rope: Enable RoPE
        """
    
    def forward(
        self,
        x: Tensor,
        return_attn: bool = False
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """
        Args:
            x: [B, T, F] input features
            return_attn: Return attention weights per layer
        
        Returns:
            logits: [B, num_classes]
            attns (optional): List of [B, H, S, S] per layer
        """
```

---

## Training & Evaluation

### `train_one_fold(...) -> Tuple[Module, float, int]`

Train model for one LOASO fold.

```python
def train_one_fold(
    tr_loader: DataLoader,
    va_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    max_epochs: int,
    patience: int,
    lr: float,
    wd: float
) -> Tuple[nn.Module, float, int]:
    """
    Returns:
        model: Best model (by validation loss)
        train_time_s: Total training time
        epochs_ran: Number of epochs completed
    """
```

**Features:**
- AdamW optimizer with gradient clipping (max_norm=1.0)
- Early stopping on validation loss
- Automatic best checkpoint restoration

---

### `eval_loader(...) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray, ...]`

Evaluate model on a DataLoader.

```python
def eval_loader(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    return_attn: bool = False
) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray, Optional[List]]:
    """
    Returns:
        metrics: Dict with acc, pre, rec, f1, auc, ece, cm, timing
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        rollout (if return_attn): List of rollout vectors
    """
```

**Metrics Dictionary:**
| Key | Description |
|-----|-------------|
| `acc` | Accuracy |
| `pre` | Precision |
| `rec` | Recall |
| `f1` | F1 score |
| `auc` | ROC-AUC |
| `ece` | Expected Calibration Error |
| `cm` | Confusion matrix |
| `infer_time_total_s` | Total inference time |
| `infer_time_ms_per_sample` | Per-sample latency |

---

### `expected_calibration_error(y_true, y_prob, n_bins=15) -> float`

Compute Expected Calibration Error.

**Formula:**
$$ECE = \sum_{b=1}^{B} \frac{|B_b|}{N} |acc(B_b) - conf(B_b)|$$

---

### `attention_rollout_from_attns(attn_list) -> List[np.ndarray]`

Compute attention rollout from [CLS] to all tokens.

**Parameters:**
- `attn_list`: List of attention tensors `[B, H, S, S]` per layer

**Returns:**
- List of `[S-1]` arrays (one per sample, excluding CLS)

**Algorithm:**
```python
rollout = I + mean(A_1, axis=heads)
for A in A_layers[1:]:
    rollout = rollout @ (I + mean(A, axis=heads))
return normalize(rollout[:, 0, 1:])  # CLS row, exclude self
```

---

## Analysis & Visualization

### `compute_optimal_windows_per_shell(...) -> Dict[str, Tuple[int,int]]`

Find best frequency window per held-out shell.

```python
def compute_optimal_windows_per_shell(
    X: np.ndarray,          # [N, T, 72] complex
    y: np.ndarray,          # [N] labels
    A_comp: np.ndarray,     # [N] shell IDs
    seeds: Tuple[int, ...]
) -> Dict[str, Tuple[int, int]]:
    """
    Returns:
        Dict mapping shell ID -> (start_idx, end_idx)
    """
```

---

### `plot_roc_curves_for_best_run(...)`

Generate ROC curve plots for the best-performing run.

**Outputs:**
- `roc_best_run_overall.png`: Aggregate ROC
- `roc_best_run_shell_{A2,A3,A14,A16}.png`: Per-shell ROCs
- `roc_best_run_shells_all.png`: Combined multi-curve figure

---

### `aggregate_and_plot_rollout(...)`

Aggregate attention rollout across samples and generate visualizations.

```python
def aggregate_and_plot_rollout(
    attn_pattern: str = "attention_*_rollout_*.npy",
    outdir: str = "analysis",
    shell_windows: Optional[Dict[str, Tuple[int,int]]] = None
):
    """
    Outputs:
        - rollout_profile_{shell}.png per shell
        - rollout_profiles_all.png combined
        - attention_peak_tokens.csv with peak bin analysis
    """
```

---

### `feature_stats_by_shell(...)`

Compute and visualize feature statistics grouped by adipose shell.

**Outputs:**
- `feature_means_by_shell.csv`: Mean feature values per shell
- `feature_cohens_d_by_shell.csv`: Effect sizes (shell vs. rest)
- `feature_anova_F_p.csv`: ANOVA F-statistics and p-values
- `top_features_by_ANOVA.png`: Bar chart of most discriminative features
- `pca_features_by_shell.png`: 2D PCA visualization
- `profile_bins_*.png`: Feature profiles across frequency bins

---

## Utility Functions

### `set_seed(seed=42)`

Set random seeds for reproducibility across numpy, torch, and CUDA.

### `finite_or_zero(a) -> np.ndarray`

Replace non-finite values with zero.

### `finite_complex_or_zero(z) -> np.ndarray`

Replace non-finite complex values with 0+0j.

### `split_aphant(phant_id: str) -> Tuple[str, str]`

Parse phantom ID into adipose and fibroglandular components.

```python
>>> split_aphant("A2F3")
('A2', 'F3')
>>> split_aphant("invalid")
('UNK', 'UNK')
```