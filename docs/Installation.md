# Installation & Quick Start Guide

This guide will help you get the Adipose-LOASO RoPE Transformer running on your system.

## System Requirements

### Hardware
- **CPU**: Any modern x86-64 processor
- **RAM**: 8 GB minimum, 16 GB recommended
- **GPU**: NVIDIA GPU with CUDA support (recommended for training)
  - Tested on: RTX 3090, A100, V100
  - VRAM: 4 GB minimum

### Software
- **Operating System**: Linux (Ubuntu 20.04+), Windows 10/11, macOS
- **Python**: 3.8, 3.9, 3.10, or 3.11
- **CUDA**: 11.7+ (if using GPU)

## Installation

### Option 1: pip (Recommended)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install numpy torch scikit-learn scipy matplotlib

# Clone repository
git clone https://github.com/yourusername/adipose-loaso-transformer.git
cd adipose-loaso-transformer
```

### Option 2: Conda

```bash
# Create conda environment
conda create -n loaso python=3.10
conda activate loaso

# Install PyTorch (adjust for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
conda install numpy scikit-learn scipy matplotlib

# Clone repository
git clone https://github.com/yourusername/adipose-loaso-transformer.git
cd adipose-loaso-transformer
```

### Verify Installation

```python
python -c "
import torch
import numpy as np
import sklearn
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'NumPy: {np.__version__}')
print(f'scikit-learn: {sklearn.__version__}')
"
```

Expected output:
```
PyTorch: 2.x.x
CUDA available: True
NumPy: 1.2x.x
scikit-learn: 1.x.x
```

## Data Preparation

### Required Files

1. **Frequency-domain data**: `fd_data_s11_adi.pickle`
2. **Metadata**: `md_list_s11_adi.pickle`

### Data Format Verification

```python
import pickle
import numpy as np

# Load and verify data
with open("path/to/fd_data_s11_adi.pickle", "rb") as f:
    X = pickle.load(f)

with open("path/to/md_list_s11_adi.pickle", "rb") as f:
    meta = pickle.load(f)

# Check data
print(f"Data shape: {X.shape}")          # Should be [N, T, 72]
print(f"Data dtype: {X.dtype}")          # Should be complex
print(f"Metadata entries: {len(meta)}")  # Should equal N
print(f"Sample metadata: {meta[0].keys()}")
```

### Expected Metadata Structure

```python
{
    'phant_id': 'A2F3',      # Required: Phantom identifier
    'tum_diam': 15.0,        # Required: Tumor diameter (NaN if absent)
    'id': 1,                 # Optional
    'n_expt': 5,             # Optional
    'n_session': 2,          # Optional
    'adi_ref_id': 1,         # Optional
    'fib_ref_id': 3,         # Optional
    'fib_ang': 45.0,         # Optional
    'ant_rad': 50.0,         # Optional
    'ant_z': 0.0,            # Optional
    'date': '2023-01-15',    # Optional
}
```

## Quick Start

### 1. Configure Paths

Edit the configuration in `adipose_loaso_rope.py`:

```python
@dataclass
class CONFIG:
    DATA_PATH: str = "/your/path/to/fd_data_s11_adi.pickle"
    METADATA_PATH: str = "/your/path/to/md_list_s11_adi.pickle"
    # ... rest of config
```

Or set programmatically:

```python
from adipose_loaso_rope import cfg
cfg.DATA_PATH = "/your/path/to/fd_data_s11_adi.pickle"
cfg.METADATA_PATH = "/your/path/to/md_list_s11_adi.pickle"
```

### 2. Run Training

**Basic run (5 seeds, all shells):**
```bash
python adipose_loaso_rope.py
```

**With RoPE disabled (ablation):**
```bash
python adipose_loaso_rope.py --no_rope
```

### 3. Monitor Progress

Training output shows:
```
[WIN] Candidate windows: 800
[WIN] Shell A2: best [32:80) len=48 (4.000–10.000 ns) CV-AUC=0.7234
[WIN] Shell A3: best [28:76) len=48 (3.500–9.500 ns) CV-AUC=0.7189
...

===== RUN 1 (seed=42) — heads=2, layers=2, use_rope=False =====
[A2] Using window [32:80) len=48 bins (4.000–10.000 ns)
[Hold-out A2] samples -> train=380, val=95, test=125
  Epoch 001 | Train 0.6823 | Val 0.6512
  Epoch 002 | Train 0.6234 | Val 0.6089
  ...
[A2] ACC=0.752 PRE=0.789 REC=0.712 F1=0.749 AUC=0.8234 ECE=0.0523
```

### 4. View Results

After training completes:

```
analysis/
├── loaso_runs_metrics.csv    # Main results table
├── freq/                     # Frequency analysis
├── heatmaps/                 # Signal visualizations
├── features/                 # Feature importance
├── roc/                      # ROC curves
└── rollout_best_run/         # Attention analysis
```

## Quick Experiments

### Test with Single Seed

```python
from adipose_loaso_rope import cfg
cfg.RUN_SEEDS = (42,)  # Single seed for faster testing
```

### Reduce Training Time

```python
cfg.MAX_EPOCHS = 20
cfg.PATIENCE = 5
cfg.WIN_SEARCH_MAX_CANDIDATES = 100
```

### CPU-Only Testing

```python
import torch
device = torch.device("cpu")
# The code automatically falls back to CPU if CUDA unavailable
```

### Subset of Shells

To test on specific shells only:

```python
# In main(), modify ADIPOSE_LIST
ADIPOSE_LIST = ["A2", "A3"]  # Skip A14, A16
```

## Common Issues & Solutions

### Issue: CUDA Out of Memory

**Solution:** Reduce batch size
```python
cfg.BATCH_SIZE = 16  # or 8
```

### Issue: Pickle Loading Errors

**Solution:** Check Python/NumPy version compatibility
```bash
# Data was likely saved with different numpy version
pip install numpy==1.23.5  # Match original version
```

### Issue: Empty Window Grid

**Solution:** Adjust window search parameters
```python
cfg.WIN_SEARCH_MIN_LEN_BINS = 16  # Smaller minimum
cfg.WIN_SEARCH_MAX_LEN_BINS = 128  # Larger maximum
```

### Issue: Poor Performance

**Possible solutions:**
1. Increase model capacity:
   ```python
   cfg.D_MODEL = 256
   cfg.NUM_LAYERS = 3
   cfg.N_HEAD = 4
   ```
2. Adjust learning rate:
   ```python
   cfg.LR = 1e-4  # or 3e-4
   ```
3. Increase patience:
   ```python
   cfg.PATIENCE = 20
   ```

## Next Steps

- **[API Reference](API_REFERENCE.md)**: Detailed function documentation
- **[Methodology](METHODOLOGY.md)**: Scientific background and design decisions
- **[Contributing](CONTRIBUTING.md)**: How to contribute to the project