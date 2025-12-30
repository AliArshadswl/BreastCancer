#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adipose-LOASO RoPE Transformer (with optimal window search + ROC + RoPE ablation + timing)
DIRECT FREQUENCY-DOMAIN TOKENS VERSION

CHANGES vs handcrafted-feature version:
- Uses DIRECT frequency-domain per-bin tokens as Transformer inputs (no handcrafted 150-D features).
- Each frequency bin is one token built from the complex 72-antenna measurement:
    * mode="reim":      token = [Re(z_1..72), Im(z_1..72)]          => 144 dims
    * mode="mag_sincos":token = [|z|, sin(angle), cos(angle)]       => 216 dims
- All leakage-safe steps preserved: LOASO, window selection (train-only), scaling (train-only),
  metadata encoding (train-only), etc.

Outputs:
- analysis/loaso_runs_metrics.csv
- analysis/freq/*              (frequency window visualization)
- analysis/heatmaps/*          (|Z| heatmaps)
- analysis/roc/*               (ROC curves for best run)
- analysis/rollout_best_run/*  (attention rollout and peak tokens for best run)

Update cfg.DATA_PATH and cfg.METADATA_PATH to your local paths.
"""

import os, math, pickle, warnings, csv, glob, time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, auc
)
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ========================== CONFIG ==========================

@dataclass
class CONFIG:
    # Data paths
    DATA_PATH: str = r"E:/Breast_Cancer/Ali Arshad/BMS/ML_in_BMS/umbmid/gen-three/clean/fd_data_s11_adi.pickle"
    METADATA_PATH: str = r"E:/Breast_Cancer/Ali Arshad/BMS/ML_in_BMS/umbmid/gen-three/clean/md_list_s11_adi.pickle"
    IS_PICKLE: bool = True
    SILENCE_NUMPY_PICKLE_WARNING: bool = True

    # Frequency axis labeling for plots (optional; Gen-3 ≈ 1–9 GHz with Δf ≈ 8 MHz)
    F0_HZ: Optional[float] = 1.0e9
    DF_HZ: Optional[float] = 8.0e6

    # Geometry
    EXPECTED_CHANNELS: int = 72

    # --------- DIRECT TOKEN MODE (NEW) ----------
    # "reim" => per-bin token is Re/Im across 72 antennas (144 dims)
    # "mag_sincos" => per-bin token is |Z| + sin(phase) + cos(phase) (216 dims)
    DIRECT_TOKEN_MODE: str = "reim"
    DO_SIGNAL_SCALING: bool = True  # leakage-safe scaling fit on train only (recommended)

    # Model
    D_MODEL: int = 192
    N_HEAD: int = 2
    NUM_LAYERS: int = 2
    D_FF: int = 384
    DROPOUT: float = 0.2
    USE_CLS: bool = True

    # RoPE ablation switch (True = use RoPE, False = standard MHA)
    USE_ROPE: bool = False

    # Training
    BATCH_SIZE: int = 32
    MAX_EPOCHS: int = 80
    PATIENCE: int = 12
    LR: float = 1.5e-4
    WEIGHT_DECAY: float = 3e-4
    VAL_FRACTION: float = 0.2

    # Safe metadata toggles
    INCLUDE_YEAR: bool = False
    INCLUDE_EMP_REF: bool = False

    # Runs
    RUN_SEEDS: tuple = (42, 43, 44, 45, 46)

    # Output metrics CSV (now inside analysis/)
    OUT_DIR: str = "analysis"
    RESULTS_CSV: str = os.path.join("analysis", "loaso_runs_metrics.csv")

    # Analysis outputs
    N_HEATMAP_SAMPLES_PER_SHELL: int = 3

    # ---------- Window search controls ----------
    WIN_SEARCH_MIN_LEN_BINS: int = 24
    WIN_SEARCH_MAX_LEN_BINS: int = 96
    WIN_SEARCH_STEP_BINS: int = 1
    WIN_SEARCH_MAX_CANDIDATES: int = 800
    WIN_SEARCH_KFOLDS: int = 3
    WIN_SEARCH_RANDOM_STATE: int = 1337


cfg = CONFIG()

ADIPOSE_LIST = ["A2", "A3", "A14", "A16"]

SAFE_NUMERIC = ["id","n_expt","n_session","adi_ref_id","fib_ref_id","fib_ang","ant_rad","ant_z"]
if cfg.INCLUDE_EMP_REF:
    SAFE_NUMERIC.append("emp_ref_id")


# ========================== SEED ==========================

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ========================== IO ==========================

def _silenced_pickle_load(f):
    if not cfg.SILENCE_NUMPY_PICKLE_WARNING:
        return pickle.load(f)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="numpy.core.numeric is deprecated", category=DeprecationWarning)
        return pickle.load(f)

def load_complex_data(path: str, is_pickle: bool) -> np.ndarray:
    """Load complex-valued frequency-domain data: expects [N, T, 72] complex."""
    if is_pickle:
        with open(path, "rb") as f:
            data = _silenced_pickle_load(f)
    else:
        if path.endswith(".npy"):
            data = np.load(path, allow_pickle=False)
        elif path.endswith(".npz"):
            z = np.load(path, allow_pickle=False)
            data = z[list(z.files)[0]]
        else:
            raise ValueError("Use .npy/.npz for non-pickle.")
    arr = np.asarray(data)
    if arr.ndim != 3 or arr.shape[-1] != cfg.EXPECTED_CHANNELS:
        raise ValueError(f"Expect [N,T,72], got {arr.shape}")
    if not np.iscomplexobj(arr):
        raise ValueError("Data must be complex-valued (frequency-domain).")
    return arr


# ========================== SANITIZERS ==========================

def finite_or_zero(a):
    out = np.array(a, copy=True)
    out[~np.isfinite(out)] = 0.0
    return out

def finite_complex_or_zero(z):
    z = np.asarray(z)
    re, im = np.real(z), np.imag(z)
    mask = ~np.isfinite(re) | ~np.isfinite(im)
    if not mask.any():
        return z
    zc = z.copy()
    zc[mask] = 0.0 + 0.0j
    return zc


# ========================== WINDOW SEARCH (TRAIN-ONLY) ==========================

def quick_window_features(Xw: np.ndarray) -> np.ndarray:
    """
    Xw: [N, Tw, 72] complex. Collapse antennas, summarize over bins quickly.
    Returns [N, 5] with inexpensive, robust stats used only for window selection.
    """
    mag = np.abs(Xw)                          # (N,Tw,72)
    mA = mag.mean(axis=2)                     # (N,Tw) mean over antennas
    sA = mag.std(axis=2)                      # (N,Tw) std over antennas
    f = np.stack([
        mA.mean(axis=1),                      # mean magnitude
        mA.std(axis=1),                       # std over bins
        sA.mean(axis=1),                      # mean antenna-wise std
        sA.std(axis=1),                       # variability of intra-antenna std
        (mA.max(axis=1) - mA.min(axis=1)),    # dynamic range
    ], axis=1).astype(np.float32)
    return f  # (N,5)

def score_window_cv_auc(X: np.ndarray, y: np.ndarray, s_idx: int, e_idx: int,
                        train_idx: np.ndarray, n_splits: int = 3, seed: int = 0) -> float:
    """
    Train-only scoring: k-fold CV AUC on quick features from X[:, s:e, :].
    Uses Logistic Regression.
    """
    Xw = X[:, s_idx:e_idx, :]
    Fq = quick_window_features(Xw)
    Xtr = Fq[train_idx]
    ytr = y[train_idx]
    if len(np.unique(ytr)) < 2:
        return 0.5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, va in skf.split(Xtr, ytr):
        scaler = StandardScaler().fit(Xtr[tr])
        Ztr = scaler.transform(Xtr[tr])
        Zva = scaler.transform(Xtr[va])
        clf = LogisticRegression(
            solver="lbfgs", max_iter=200, C=1.0,
            class_weight="balanced", n_jobs=None
        )
        clf.fit(Ztr, ytr[tr])
        try:
            prob = clf.predict_proba(Zva)[:, 1]
        except Exception:
            prob = clf.decision_function(Zva)
            prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-12)
        if len(np.unique(ytr[va])) == 2:
            aucs.append(roc_auc_score(ytr[va], prob))
    return float(np.mean(aucs)) if aucs else 0.5

def generate_window_grid(T: int, min_len: int, max_len: int, step: int) -> List[Tuple[int,int]]:
    grid: List[Tuple[int,int]] = []
    max_len = min(max_len, T)
    for L in range(min_len, max_len + 1, step):
        last_start = T - L
        if last_start < 0:
            break
        for s in range(0, last_start + 1, step):
            grid.append((s, s + L))
    return grid

def compute_optimal_windows_per_shell(X: np.ndarray, y: np.ndarray, A_comp: np.ndarray, seeds: Tuple[int,...]) -> Dict[str, Tuple[int,int]]:
    """
    For each shell S (held out later), choose the best window using only train shells (!= S).
    One (s,e) per shell chosen and reused across runs.
    """
    N, T, _ = X.shape
    rng = np.random.RandomState(cfg.WIN_SEARCH_RANDOM_STATE)
    grid = generate_window_grid(
        T,
        min_len=max(4, cfg.WIN_SEARCH_MIN_LEN_BINS),
        max_len=max(5, cfg.WIN_SEARCH_MAX_LEN_BINS),
        step=max(1, cfg.WIN_SEARCH_STEP_BINS)
    )
    if len(grid) == 0:
        raise ValueError("Window grid is empty; adjust WIN_SEARCH_* limits.")
    if len(grid) > cfg.WIN_SEARCH_MAX_CANDIDATES:
        grid = [grid[i] for i in rng.choice(len(grid), cfg.WIN_SEARCH_MAX_CANDIDATES, replace=False)]

    by_shell: Dict[str, Tuple[int,int]] = {}
    print(f"[WIN] Candidate windows: {len(grid)}")

    base_seed = int(seeds[0]) if len(seeds) else 0
    shells = sorted(list(set(A_comp)))
    for S in shells:
        train_idx = np.where(A_comp != S)[0]
        best_auc = -1.0
        best_pair = grid[0]
        for (s_idx, e_idx) in grid:
            if e_idx - s_idx < 2:
                continue
            auc_val = score_window_cv_auc(
                X, y, s_idx, e_idx, train_idx,
                n_splits=cfg.WIN_SEARCH_KFOLDS, seed=base_seed
            )
            if auc_val > best_auc:
                best_auc = auc_val
                best_pair = (s_idx, e_idx)
        by_shell[S] = best_pair
        print(f"[WIN] Shell {S}: best [{best_pair[0]}:{best_pair[1]}) len={best_pair[1]-best_pair[0]} CV-AUC={best_auc:.4f}")
    return by_shell


# ========================== LABELS & PHANTOM PARSING ==========================

def make_labels(meta: List[dict]) -> np.ndarray:
    """Binary label: 0 = no tumor (NaN tum_diam), 1 = tumor present."""
    return np.array([0 if np.isnan(m.get("tum_diam", np.nan)) else 1 for m in meta], dtype=int)

def split_aphant(phant_id: str) -> tuple[str, str]:
    """Split phantom id like 'A2F3' into ('A2','F3')."""
    if not isinstance(phant_id, str) or not phant_id.startswith("A"):
        return "UNK", "UNK"
    p = phant_id.find("F")
    if p <= 1:
        return "UNK", "UNK"
    return phant_id[:p], phant_id[p:]


# ========================== METADATA (SAFE) ==========================

def extract_safe_metadata(meta: List[dict]) -> Tuple[np.ndarray, dict, np.ndarray]:
    """
    Extract leakage-safe metadata:
    - A_comp: adipose shell ID (A2, A3, A14, A16, ...)
    - meta_struct: dict containing F_comp, numeric features, and year (optional).
    - phant_ids: phantom IDs as strings.
    """
    N = len(meta)
    A_list, F_list, phant_ids = [], [], []
    numX = np.zeros((N, len(SAFE_NUMERIC)), dtype=float)
    years = np.zeros(N, dtype=int)
    for i, m in enumerate(meta):
        ph = m.get("phant_id", "UNK")
        phant_ids.append(ph)
        A, F = split_aphant(ph)
        A_list.append(A)
        F_list.append(F)
        for j, k in enumerate(SAFE_NUMERIC):
            v = m.get(k, 0.0)
            try:
                numX[i, j] = float(v) if v is not None else 0.0
            except Exception:
                numX[i, j] = 0.0
        if cfg.INCLUDE_YEAR:
            yyyy = 0
            if isinstance(m.get("date",""), str) and len(m["date"]) >= 4:
                try:
                    yyyy = int(m["date"][:4])
                except Exception:
                    yyyy = 0
            years[i] = yyyy

    meta_struct = {
        "F_comp": np.array(F_list, dtype=object),
        "numX": np.nan_to_num(numX, 0.0, 0.0, 0.0),
        "year": years
    }
    return np.array(A_list, dtype=object), meta_struct, np.array(phant_ids, dtype=object)


# ========================== DIRECT TOKEN BUILDER (NEW) ==========================

def build_signal_seq_direct(Xw: np.ndarray, mode: str = "reim") -> Tuple[np.ndarray, List[str]]:
    """
    Build direct frequency-domain per-bin tokens from complex windowed data.

    Xw: [N, Tw, 72] complex
    mode:
      - "reim": token = [Re(z_1..72), Im(z_1..72)]          => dim 144
      - "mag_sincos": token = [|z|, sin(angle), cos(angle)] => dim 216
    """
    Xw = finite_complex_or_zero(Xw)
    N, T, C = Xw.shape
    assert C == cfg.EXPECTED_CHANNELS

    mode = mode.lower().strip()
    if mode == "reim":
        re = np.real(Xw).astype(np.float32)
        im = np.imag(Xw).astype(np.float32)
        seq = np.concatenate([re, im], axis=2)  # [N,T,144]
        keys = [f"re_{i}" for i in range(C)] + [f"im_{i}" for i in range(C)]
        return seq, keys

    if mode == "mag_sincos":
        mag = np.abs(Xw).astype(np.float32)
        ang = np.angle(Xw).astype(np.float32)
        sinp = np.sin(ang).astype(np.float32)
        cosp = np.cos(ang).astype(np.float32)
        seq = np.concatenate([mag, sinp, cosp], axis=2)  # [N,T,216]
        keys = [f"mag_{i}" for i in range(C)] + [f"sinp_{i}" for i in range(C)] + [f"cosp_{i}" for i in range(C)]
        return seq, keys

    raise ValueError(f"Unknown DIRECT_TOKEN_MODE='{mode}'. Use 'reim' or 'mag_sincos'.")


# ========================== DATASET ==========================

class PatchSequenceDataset(Dataset):
    def __init__(self, Xseq: np.ndarray, y: np.ndarray):
        self.X = Xseq.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# ========================== RoPE Transformer (with attn return) ==========================

def rotate_half(x):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)

def apply_rope(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512, base=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)
    def forward(self, seq_len):
        return self.cos[..., :seq_len, :], self.sin[..., :seq_len, :]

class MHA_RoPE(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, rope_max_len=512, use_rope: bool = True):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)
        self.use_rope = use_rope
        self.rope = RotaryEmbedding(self.head_dim, rope_max_len) if use_rope else None

    def forward(self, x, return_attn: bool=False):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        if self.use_rope and self.rope is not None:
            cos, sin = self.rope(T)
            cos = cos.expand(B, self.nhead, T, self.head_dim)
            sin = sin.expand(B, self.nhead, T, self.head_dim)
            q, k = apply_rope(q, k, cos, sin)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_drop(self.o(out))
        if return_attn:
            return out, attn
        return out

class EncoderLayerRoPE(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout, rope_max_len=512, use_rope: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MHA_RoPE(d_model, nhead, dropout, rope_max_len, use_rope=use_rope)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, return_attn: bool=False):
        y = self.attn(self.norm1(x), return_attn=return_attn)
        if return_attn:
            y_out, attn = y
            x = x + self.drop1(y_out)
            y2 = self.ff(self.norm2(x))
            x = x + self.drop2(y2)
            return x, attn
        else:
            x = x + self.drop1(y)
            y2 = self.ff(self.norm2(x))
            x = x + self.drop2(y2)
            return x

class EncoderRoPE(nn.Module):
    def __init__(self, n_layers, d_model, nhead, d_ff, dropout, rope_max_len=512, use_rope: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayerRoPE(d_model, nhead, d_ff, dropout, rope_max_len, use_rope=use_rope)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, return_attn: bool=False):
        attns = []
        for l in self.layers:
            if return_attn:
                x, a = l(x, return_attn=True)
                attns.append(a)
            else:
                x = l(x, return_attn=False)
        x = self.norm(x)
        if return_attn:
            return x, attns
        return x

class SeqClassifierRoPE(nn.Module):
    def __init__(self, in_dim, d_model, nhead, n_layers, d_ff, dropout,
                 use_cls=True, num_classes=2, max_len=512, use_rope: bool = True):
        super().__init__()
        self.use_cls = use_cls
        self.proj = nn.Linear(in_dim, d_model)
        if use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls, std=0.02)
        self.enc = EncoderRoPE(
            n_layers, d_model, nhead, d_ff, dropout,
            rope_max_len=max_len, use_rope=use_rope
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, return_attn: bool=False):
        B, T, _ = x.shape
        h = self.proj(x)
        if self.use_cls:
            cls = self.cls.expand(B, -1, -1)
            h = torch.cat([cls, h], dim=1)  # [B,T+1,D]
        if return_attn:
            h, attns = self.enc(h, return_attn=True)
        else:
            h = self.enc(h, return_attn=False)
        pooled = h[:, 0] if self.use_cls else h.mean(1)
        logits = self.head(pooled)
        if return_attn:
            return logits, attns
        return logits


# ========================== TRAIN / EVAL ==========================

def train_one_fold(tr_loader, va_loader, model, device, max_epochs, patience, lr, wd):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss()
    best = float("inf")
    best_state = None
    wait = 0
    start_time = time.time()
    epochs_ran = 0

    for ep in range(1, max_epochs+1):
        epochs_ran = ep
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(tr_loader.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device); yb = yb.to(device)
                va_loss += crit(model(xb), yb).item() * xb.size(0)
        va_loss /= len(va_loader.dataset)

        print(f"  Epoch {ep:03d} | Train {tr_loss:.4f} | Val {va_loss:.4f}")

        if va_loss + 1e-6 < best:
            best = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("  Early stopping.")
                break

    total_train_time = time.time() - start_time
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, total_train_time, epochs_ran

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    y_true = y_true.astype(int)
    conf = np.maximum(y_prob, 1.0 - y_prob)
    pred = (y_prob >= 0.5).astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        if i < n_bins - 1:
            m = (conf >= lo) & (conf < hi)
        else:
            m = (conf >= lo) & (conf <= hi)
        if not np.any(m):
            continue
        bin_conf = conf[m].mean()
        bin_acc = (pred[m] == y_true[m]).mean()
        ece += np.sum(m) / len(y_true) * abs(bin_acc - bin_conf)
    return float(ece)

def attention_rollout_from_attns(attn_list: List[torch.Tensor]) -> List[np.ndarray]:
    heads_mean = []
    for A in attn_list:
        A = A.detach().cpu().numpy()      # [B,H,S,S]
        A = A.mean(axis=1)                # [B,S,S]
        S = A.shape[-1]
        A = A + np.eye(S, dtype=A.dtype)[None, :, :]
        A = A / (A.sum(axis=-1, keepdims=True) + 1e-12)
        heads_mean.append(A)
    rollout = heads_mean[0]
    for A in heads_mean[1:]:
        rollout = np.matmul(rollout, A)
    cls_to_tokens = rollout[:, 0, 1:]     # [B, S-1]
    cls_to_tokens = cls_to_tokens / (cls_to_tokens.sum(axis=1, keepdims=True) + 1e-12)
    return [cls_to_tokens[i] for i in range(cls_to_tokens.shape[0])]

def eval_loader(loader, model, device, return_attn: bool=False):
    model.eval()
    y_true=[]; y_pred=[]; y_prob=[]; rollout_all=[]
    start_time = time.time()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            if return_attn:
                logits, attns = model(xb, return_attn=True)
                prob = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                rollout = attention_rollout_from_attns(attns)
                rollout_all.extend([r for r in rollout])
            else:
                logits = model(xb)
                prob = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            pred = (prob >= 0.5).astype(int)
            y_true.append(yb.numpy()); y_pred.append(pred); y_prob.append(prob)
    total_infer_time = time.time() - start_time

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)

    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "pre": precision_score(y_true, y_pred),
        "rec": recall_score(y_true, y_pred),
        "f1":  f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan"),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=15),
        "cm":  confusion_matrix(y_true, y_pred),
        "infer_time_total_s": float(total_infer_time),
        "infer_time_ms_per_sample": float(1000.0 * total_infer_time / len(y_true))
    }
    if return_attn:
        return metrics, y_true, y_pred, y_prob, rollout_all
    return metrics, y_true, y_pred, y_prob


# ========================== VISUALIZATION HELPERS ==========================

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def _freq_axis_labels(T, start_idx, f0_full, df):
    if f0_full is None or df is None:
        x = np.arange(T)
        xlabel = "Frequency bin"
    else:
        x = f0_full + (start_idx + np.arange(T)) * df
        xlabel = "Frequency (Hz)"
    return x, xlabel

def visualize_frequency_window_full_context(X, s_idx, e_idx, f0_full, df, outdir):
    _ensure_dir(outdir)
    N, F, A = X.shape
    mag = np.abs(X)
    mean_over_NA = mag.mean(axis=(0,2))
    win_len = e_idx - s_idx

    fig, ax = plt.subplots(figsize=(9, 3.2))
    x_full, xlabel = _freq_axis_labels(F, 0, f0_full, df)
    ax.plot(x_full, mean_over_NA, lw=1.2)
    ax.axvspan(x_full[s_idx], x_full[e_idx-1], color='grey', alpha=0.25,
               label=f'Window [{s_idx}:{e_idx}) len={win_len}')
    ax.set_xlabel(xlabel); ax.set_ylabel('Mean |Z| over N, antennas')
    ax.set_title('Global spectrum with window highlighted')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "freq_global_window.png"), dpi=200)
    plt.close(fig)

    mean_over_NA_win = mag[:, s_idx:e_idx, :].mean(axis=(0,2))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x_win, xlabel = _freq_axis_labels(e_idx - s_idx, s_idx, f0_full, df)
    ax.plot(x_win, mean_over_NA_win, marker='o', ms=3, lw=1)
    ax.set_xlabel(xlabel); ax.set_ylabel('Mean |Z| over N, antennas')
    ax.set_title(f'Mean |Z| across window (len={win_len})')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "freq_window_mean_abs.png"), dpi=200)
    plt.close(fig)

def heatmaps_per_shell_examples(Xw, A_comp, shells, outdir, per_shell=3):
    _ensure_dir(outdir)
    mag = np.abs(Xw)
    T = mag.shape[1]
    for s in shells:
        idx = np.where(A_comp == s)[0]
        if len(idx) == 0:
            continue
        chosen = idx[:min(per_shell, len(idx))]
        for i in chosen:
            M = mag[i].T  # (72,T)
            fig, ax = plt.subplots(figsize=(5.6, 4.2))
            im = ax.imshow(M, aspect='auto', origin='lower')
            ax.set_xlabel(f'Window bin (0..{T-1})')
            ax.set_ylabel('Antenna index (0..71)')
            ax.set_title(f'|Z| heatmap — shell {s} — sample {i} — len={T}')
            fig.colorbar(im, ax=ax, shrink=0.86, label='|Z|')
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"heatmap_abs_shell_{s}_sample_{i}.png"), dpi=220)
            plt.close(fig)


# ========================== ROLLOUT AGGREGATION ==========================

def aggregate_and_plot_rollout(attn_pattern="attention_*_rollout_*.npy",
                               outdir="analysis",
                               shell_windows: Optional[Dict[str, Tuple[int,int]]] = None):
    _ensure_dir(outdir)
    files = sorted(glob.glob(attn_pattern))
    if not files:
        print("[rollout] No files found for pattern:", attn_pattern)
        return

    shell_curves = defaultdict(list)
    for path in files:
        base = os.path.basename(path)
        try:
            shell = base.split("_rollout_")[-1].split(".npy")[0]
        except Exception:
            shell = "UNK"
        arr = np.load(path, allow_pickle=True)
        vecs = np.vstack([np.asarray(v, dtype=float) for v in arr])
        shell_curves[shell].append(vecs)

    peak_rows = []
    fig, ax = plt.subplots(figsize=(7.8, 3.6))

    for shell, li in shell_curves.items():
        V = np.vstack(li)          # [Nshell, T_shell]
        T_shell = V.shape[1]
        x_shell = np.arange(T_shell)

        mu = V.mean(axis=0)
        se = V.std(axis=0, ddof=1) / np.sqrt(max(V.shape[0], 1))

        ax.plot(x_shell, mu, label=shell)
        ax.fill_between(x_shell, mu-se, mu+se, alpha=0.2)

        # Per-shell plot
        fig2, ax2 = plt.subplots(figsize=(7.2, 3.2))
        ax2.plot(x_shell, mu, lw=1.5)
        ax2.fill_between(x_shell, mu-se, mu+se, alpha=0.25)
        ax2.set_xlabel("Token index")
        ax2.set_ylabel("Rollout importance")
        ax2.set_title(f"Attention rollout — {shell} (len={T_shell})")
        fig2.tight_layout()
        fig2.savefig(os.path.join(outdir, f"rollout_profile_{shell}.png"), dpi=220)
        plt.close(fig2)

        peak_idx = int(np.argmax(mu))
        peak_val = float(mu[peak_idx])

        abs_bin_idx = ""
        freq_hz = ""
        if shell_windows is not None and shell in shell_windows:
            s_idx, _ = shell_windows[shell]
            abs_bin = int(s_idx + peak_idx)
            abs_bin_idx = abs_bin
            if cfg.F0_HZ is not None and cfg.DF_HZ is not None:
                freq_hz = float(cfg.F0_HZ + abs_bin * cfg.DF_HZ)

        peak_rows.append([shell, peak_idx, peak_val, abs_bin_idx, freq_hz])

    ax.set_xlabel("Token index")
    ax.set_ylabel("Rollout importance")
    ax.set_title("Attention rollout (mean±SEM)")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "rollout_profiles_all.png"), dpi=220)
    plt.close(fig)

    with open(os.path.join(outdir, "attention_peak_tokens.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shell", "peak_token_index", "peak_attention_value",
                    "absolute_freq_bin_index", "approx_freq_hz"])
        for row in peak_rows:
            w.writerow(row)
    print("[rollout] Saved attention_peak_tokens.csv with per-shell most-attended bin.")


# ========================== ROC PLOTS (BEST RUN) ==========================

def plot_roc_curves_for_best_run(best_run_idx: int,
                                 all_runs_ytrue: List[np.ndarray],
                                 all_runs_yprob: List[np.ndarray],
                                 all_runs_shell_ytrue: Dict[str, List[np.ndarray]],
                                 all_runs_shell_yprob: Dict[str, List[np.ndarray]],
                                 outdir: str):
    _ensure_dir(outdir)
    r = best_run_idx
    y_true = all_runs_ytrue[r]
    y_prob = all_runs_yprob[r]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.plot(fpr, tpr, lw=2, label=f'Overall (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], linestyle='--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC — Best run #{r+1} (overall)')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "roc_best_run_overall.png"), dpi=220)
    plt.close(fig)

    for shell in ADIPOSE_LIST:
        y_s_true = all_runs_shell_ytrue[shell][r]
        y_s_prob = all_runs_shell_yprob[shell][r]
        if len(np.unique(y_s_true)) < 2:
            continue
        fpr_s, tpr_s, _ = roc_curve(y_s_true, y_s_prob)
        auc_s = auc(fpr_s, tpr_s)

        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        ax.plot(fpr_s, tpr_s, lw=2, label=f'{shell} (AUC = {auc_s:.3f})')
        ax.plot([0, 1], [0, 1], linestyle='--', lw=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC — Best run #{r+1}, shell {shell}')
        ax.legend(loc='lower right')
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"roc_best_run_shell_{shell}.png"), dpi=220)
        plt.close(fig)

def plot_roc_shells_single_figure(best_run_idx: int,
                                  all_runs_shell_ytrue: Dict[str, List[np.ndarray]],
                                  all_runs_shell_yprob: Dict[str, List[np.ndarray]],
                                  outdir: str):
    _ensure_dir(outdir)
    r = best_run_idx

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    for shell in ADIPOSE_LIST:
        y_true = all_runs_shell_ytrue[shell][r]
        y_prob = all_runs_shell_yprob[shell][r]
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_s = auc(fpr, tpr)
        ax.step(fpr * 100.0, tpr * 100.0, where="post",
                linewidth=1.8,
                label=f"{shell}, AUC = {100.0 * auc_s:.1f}\\%")

    ax.plot([0, 100], [0, 100], linestyle="--", linewidth=1.2, label="Random classifier")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("False Positive Rate (\\%)")
    ax.set_ylabel("True Positive Rate (\\%)")
    ax.set_title("ROC curves per adipose shell (best run)")
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(loc="lower right", fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "roc_best_run_shells_all.png"), dpi=300)
    plt.close(fig)


# ========================== PER-FOLD PIPELINE (NO LEAKAGE) ==========================

def _ohe_fit():
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown="ignore")

def build_meta_fit_transform(F_comp: np.ndarray, numX: np.ndarray, years: np.ndarray,
                             train_idx: np.ndarray, apply_idx: np.ndarray) -> Tuple[np.ndarray, dict]:
    ohe = _ohe_fit()
    ohe.fit(F_comp[train_idx].reshape(-1, 1))
    F_ohe = ohe.transform(F_comp[apply_idx].reshape(-1, 1)).astype(np.float32)

    num_scaler = StandardScaler().fit(numX[train_idx])
    num_scaled = num_scaler.transform(numX[apply_idx]).astype(np.float32)

    mats = [F_ohe, num_scaled]
    meta_info = {"ohe_F": ohe, "num_scaler": num_scaler}

    if cfg.INCLUDE_YEAR:
        year_ohe = _ohe_fit()
        year_ohe.fit(years[train_idx].reshape(-1, 1))
        year_feats = year_ohe.transform(years[apply_idx].reshape(-1, 1)).astype(np.float32)
        mats.append(year_feats)
        meta_info["ohe_year"] = year_ohe

    meta_feats = np.concatenate(mats, axis=1)
    return meta_feats, meta_info

def scale_signal_seq_per_fold(seq_sig: np.ndarray, train_idx: np.ndarray, apply_idx: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    if not cfg.DO_SIGNAL_SCALING:
        return seq_sig[apply_idx].astype(np.float32), StandardScaler()  # dummy
    N, T, Fd = seq_sig.shape
    tr_flat = seq_sig[train_idx].reshape(-1, Fd)
    sig_scaler = StandardScaler().fit(tr_flat)
    out = seq_sig[apply_idx].reshape(-1, Fd)
    out = sig_scaler.transform(out).astype(np.float32)
    return out.reshape(len(apply_idx), T, Fd), sig_scaler

def fuse_seq_and_meta(seq_scaled, meta_feats):
    B, T, Fd = seq_scaled.shape
    M = meta_feats.shape[1]
    meta_tiled = np.repeat(meta_feats[:, None, :], T, axis=1)  # [B,T,M]
    return np.concatenate([seq_scaled, meta_tiled], axis=2).astype(np.float32)

def run_one_shell_fold(shell: str,
                       seq_sig: np.ndarray,
                       labels: np.ndarray,
                       A_comp: np.ndarray,
                       meta_struct: dict,
                       phant_ids: np.ndarray,
                       device,
                       seed: int,
                       save_rollout_prefix: Optional[str] = None) -> Tuple[dict, np.ndarray, np.ndarray]:
    set_seed(seed)

    N, T, Fsig = seq_sig.shape
    F_comp = meta_struct["F_comp"]
    numX = meta_struct["numX"]
    years = meta_struct["year"]

    test_idx = np.where(A_comp == shell)[0]
    trainval_idx = np.where(A_comp != shell)[0]

    rng = np.random.RandomState(seed + hash(("VALSPLIT", shell)) % 10000)
    perm = rng.permutation(trainval_idx)
    cut = max(1, int((1.0 - cfg.VAL_FRACTION) * len(trainval_idx)))
    tr_idx = perm[:cut]
    va_idx = perm[cut:]

    print(f"[Hold-out {shell}] samples -> train={len(tr_idx)}, val={len(va_idx)}, test={len(test_idx)}")
    ph_list = phant_ids[test_idx].tolist()
    unique_sorted = sorted(set(ph_list), key=lambda s: (s is None, s))
    print(f"[Hold-out {shell}] TEST phantoms (unique): {unique_sorted}")

    # Scale signal per fold (fit on train only)
    seq_tr_scaled, _ = scale_signal_seq_per_fold(seq_sig, tr_idx, tr_idx)
    seq_va_scaled, _ = scale_signal_seq_per_fold(seq_sig, tr_idx, va_idx)
    seq_te_scaled, _ = scale_signal_seq_per_fold(seq_sig, tr_idx, test_idx)

    # Metadata (fit on train only)
    meta_tr, _ = build_meta_fit_transform(F_comp, numX, years, tr_idx, tr_idx)
    meta_va, _ = build_meta_fit_transform(F_comp, numX, years, tr_idx, va_idx)
    meta_te, _ = build_meta_fit_transform(F_comp, numX, years, tr_idx, test_idx)

    # Fuse sequence + metadata
    Xtr = fuse_seq_and_meta(seq_tr_scaled, meta_tr)
    Xva = fuse_seq_and_meta(seq_va_scaled, meta_va)
    Xte = fuse_seq_and_meta(seq_te_scaled, meta_te)

    ytr = labels[tr_idx]; yva = labels[va_idx]; yte = labels[test_idx]

    train_loader = DataLoader(PatchSequenceDataset(Xtr, ytr), batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(PatchSequenceDataset(Xva, yva), batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(PatchSequenceDataset(Xte, yte), batch_size=cfg.BATCH_SIZE, shuffle=False)

    in_dim = Xtr.shape[-1]
    model = SeqClassifierRoPE(
        in_dim=in_dim,
        d_model=cfg.D_MODEL,
        nhead=cfg.N_HEAD,
        n_layers=cfg.NUM_LAYERS,
        d_ff=cfg.D_FF,
        dropout=cfg.DROPOUT,
        use_cls=cfg.USE_CLS,
        num_classes=2,
        max_len=(T+1 if cfg.USE_CLS else T),
        use_rope=cfg.USE_ROPE,
    ).to(device)

    print(f"[{shell}] Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M (in_dim={in_dim}, use_rope={cfg.USE_ROPE})")

    model, train_time_s, epochs_ran = train_one_fold(
        train_loader, val_loader, model, device,
        cfg.MAX_EPOCHS, cfg.PATIENCE, cfg.LR, cfg.WEIGHT_DECAY
    )

    metrics, y_true, y_pred, y_prob, rollout = eval_loader(test_loader, model, device, return_attn=True)
    metrics["train_time_s"] = float(train_time_s)
    metrics["train_epochs"] = int(epochs_ran)

    print(f"[{shell}] ACC={metrics['acc']:.3f} PRE={metrics['pre']:.3f} REC={metrics['rec']:.3f} "
          f"F1={metrics['f1']:.3f} AUC={metrics['auc']:.4f} ECE={metrics['ece']:.4f}")
    print(f"[{shell}] Confusion:\n{metrics['cm']}")
    print(f"[{shell}] train_time={train_time_s:.2f}s over {epochs_ran} epochs | infer={metrics['infer_time_ms_per_sample']:.3f} ms/sample")

    if save_rollout_prefix:
        np.save(f"{save_rollout_prefix}_rollout_{shell}.npy", np.array(rollout, dtype=object))

    return metrics, y_true, y_prob


# ========================== MAIN ==========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] USE_ROPE = {cfg.USE_ROPE}")
    print(f"[INFO] DIRECT_TOKEN_MODE = {cfg.DIRECT_TOKEN_MODE} | DO_SIGNAL_SCALING={cfg.DO_SIGNAL_SCALING}")

    _ensure_dir(cfg.OUT_DIR)

    # Load data + meta
    X = load_complex_data(cfg.DATA_PATH, cfg.IS_PICKLE)  # [N, T_full, 72]
    with open(cfg.METADATA_PATH, "rb") as f:
        meta = _silenced_pickle_load(f)
    if len(meta) != X.shape[0]:
        raise ValueError(f"Metadata length {len(meta)} != samples {X.shape[0]}")

    # Labels & metadata
    y = make_labels(meta)
    A_comp, meta_struct, phant_ids = extract_safe_metadata(meta)

    # Compute optimal windows per shell (train-only CV)
    shell_windows = compute_optimal_windows_per_shell(X, y, A_comp, seeds=cfg.RUN_SEEDS)

    # Visualize one shell's window for context (A2 if exists)
    vis_shell = "A2" if "A2" in shell_windows else next(iter(shell_windows))
    s_idx_vis, e_idx_vis = shell_windows[vis_shell]
    visualize_frequency_window_full_context(
        X, s_idx_vis, e_idx_vis, cfg.F0_HZ, cfg.DF_HZ,
        outdir=os.path.join(cfg.OUT_DIR, "freq")
    )

    # Aggregation
    rows = []
    per_run_overall = []
    per_run_shells = {s: [] for s in ADIPOSE_LIST}
    per_run_shells_acc = {s: [] for s in ADIPOSE_LIST}
    per_run_shells_f1  = {s: [] for s in ADIPOSE_LIST}
    per_run_shells_ece = {s: [] for s in ADIPOSE_LIST}

    all_runs_ytrue = []
    all_runs_yprob = []
    all_runs_shell_ytrue = {s: [] for s in ADIPOSE_LIST}
    all_runs_shell_yprob = {s: [] for s in ADIPOSE_LIST}

    for run_idx, seed in enumerate(cfg.RUN_SEEDS, start=1):
        print(f"\n===== RUN {run_idx} (seed={seed}) — heads={cfg.N_HEAD}, layers={cfg.NUM_LAYERS}, use_rope={cfg.USE_ROPE} =====")
        set_seed(seed)
        per_shell = {}
        all_true = []; all_prob = []
        save_prefix = f"attention_{run_idx}"

        for shell in ADIPOSE_LIST:
            s_idx, e_idx = shell_windows.get(shell, (s_idx_vis, e_idx_vis))
            Xw_shell = X[:, s_idx:e_idx, :]  # [N, Tw, 72]
            Tw = Xw_shell.shape[1]
            print(f"[{shell}] Using window [{s_idx}:{e_idx}) len={Tw} bins")

            # Build direct tokens (no handcrafted features)
            seq_sig, keys = build_signal_seq_direct(Xw_shell, mode=cfg.DIRECT_TOKEN_MODE)

            # Save a few heatmaps once per run (first shell only)
            if shell == ADIPOSE_LIST[0]:
                heatmaps_per_shell_examples(
                    Xw_shell, A_comp, ADIPOSE_LIST,
                    outdir=os.path.join(cfg.OUT_DIR, "heatmaps"),
                    per_shell=cfg.N_HEATMAP_SAMPLES_PER_SHELL
                )

            m, y_true, y_prob = run_one_shell_fold(
                shell, seq_sig, y, A_comp, meta_struct,
                phant_ids, device, seed,
                save_rollout_prefix=save_prefix
            )
            per_shell[shell] = m
            all_true.append(y_true)
            all_prob.append(y_prob)
            all_runs_shell_ytrue[shell].append(y_true)
            all_runs_shell_yprob[shell].append(y_prob)

        # Overall metrics for this run
        y_true = np.concatenate(all_true)
        y_prob = np.concatenate(all_prob)
        all_runs_ytrue.append(y_true)
        all_runs_yprob.append(y_prob)

        overall_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
        per_run_overall.append(overall_auc)

        for s in ADIPOSE_LIST:
            per_run_shells[s].append(per_shell[s]["auc"])
            per_run_shells_acc[s].append(per_shell[s]["acc"])
            per_run_shells_f1[s].append(per_shell[s]["f1"])
            per_run_shells_ece[s].append(per_shell[s]["ece"])

        row = {
            "run": run_idx,
            "overall_auc": float(overall_auc),
            **{f"{s}_auc": float(per_shell[s]["auc"]) for s in ADIPOSE_LIST},
            **{f"{s}_acc": float(per_shell[s]["acc"]) for s in ADIPOSE_LIST},
            **{f"{s}_f1":  float(per_shell[s]["f1"])  for s in ADIPOSE_LIST},
            **{f"{s}_ece": float(per_shell[s]["ece"]) for s in ADIPOSE_LIST},
            **{f"{s}_train_time_s": float(per_shell[s]["train_time_s"]) for s in ADIPOSE_LIST},
            **{f"{s}_infer_ms_per_sample": float(per_shell[s]["infer_time_ms_per_sample"]) for s in ADIPOSE_LIST},
        }
        rows.append(row)

        print(f"[RUN {run_idx}] overall AUC={overall_auc:.4f} | "
              f"A2={per_shell['A2']['auc']:.4f} A3={per_shell['A3']['auc']:.4f} "
              f"A14={per_shell['A14']['auc']:.4f} A16={per_shell['A16']['auc']:.4f}")

    # Save per-run CSV
    _ensure_dir(os.path.dirname(cfg.RESULTS_CSV) or ".")
    fields = (
        ["run","overall_auc"] +
        [f"{s}_auc" for s in ADIPOSE_LIST] +
        [f"{s}_acc" for s in ADIPOSE_LIST] +
        [f"{s}_f1"  for s in ADIPOSE_LIST] +
        [f"{s}_ece" for s in ADIPOSE_LIST] +
        [f"{s}_train_time_s" for s in ADIPOSE_LIST] +
        [f"{s}_infer_ms_per_sample" for s in ADIPOSE_LIST]
    )
    with open(cfg.RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[RESULTS] Saved per-run metrics -> {cfg.RESULTS_CSV}")

    # Best run by overall AUC -> ROC plots
    best_run_idx = int(np.argmax(np.array(per_run_overall)))
    print(f"[ROC] Best run index (0-based)={best_run_idx}, overall_auc={per_run_overall[best_run_idx]:.4f}")
    roc_outdir = os.path.join(cfg.OUT_DIR, "roc")
    plot_roc_curves_for_best_run(
        best_run_idx,
        all_runs_ytrue,
        all_runs_yprob,
        all_runs_shell_ytrue,
        all_runs_shell_yprob,
        outdir=roc_outdir
    )
    plot_roc_shells_single_figure(
        best_run_idx,
        all_runs_shell_ytrue,
        all_runs_shell_yprob,
        outdir=roc_outdir
    )

    # Summary
    def mstd(x):
        x = np.array(x, float)
        return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0

    ov_m, ov_s = mstd(per_run_overall)
    print("\n=== FINAL (5 runs) mean ± std ===")
    print(f"Overall AUC : {ov_m:.4f} ± {ov_s:.4f}")
    for s in ADIPOSE_LIST:
        am, as_ = mstd(per_run_shells[s])
        acm, acs = mstd(per_run_shells_acc[s])
        fm, fs = mstd(per_run_shells_f1[s])
        em, es = mstd(per_run_shells_ece[s])
        print(f"{s}: AUC {am:.4f}±{as_:.4f} | ACC {acm:.4f}±{acs:.4f} | F1 {fm:.4f}±{fs:.4f} | ECE {em:.4f}±{es:.4f}")

    # Attention rollout summary for best run only
    best_run_pattern = f"attention_{best_run_idx+1}_rollout_*.npy"
    aggregate_and_plot_rollout(
        attn_pattern=best_run_pattern,
        outdir=os.path.join(cfg.OUT_DIR, "rollout_best_run"),
        shell_windows=shell_windows
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_rope", action="store_true", help="Disable RoPE (use standard MHA).")
    parser.add_argument("--token_mode", type=str, default=None,
                        help="Direct token mode: 'reim' or 'mag_sincos'. Overrides cfg.DIRECT_TOKEN_MODE.")
    parser.add_argument("--no_signal_scaling", action="store_true",
                        help="Disable leakage-safe signal scaling (NOT recommended).")
    args = parser.parse_args()

    cfg.USE_ROPE = not args.no_rope
    if args.token_mode is not None:
        cfg.DIRECT_TOKEN_MODE = args.token_mode.strip().lower()
    if args.no_signal_scaling:
        cfg.DO_SIGNAL_SCALING = False

    main()
