#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adipose-LOGO RoPE Transformer (with optimal window search + ROC + RoPE ablation + timing)

Additions over your previous version:
- RoPE can be toggled on/off via cfg.USE_ROPE or CLI flag --no_rope (ablation).
- Training time per fold (seconds) and inference latency (ms/sample) are recorded.
- ROC curves (overall + per shell) are plotted for the best run (max overall AUC).
- A single ROC figure with all shells (A2, A3, A14, A16) is also saved.
- Attention rollout is summarized for the best run, including:
    * Per-shell rollout profiles.
    * CSV reporting which token index has maximum attention.
    * If shell_windows are provided, also the absolute frequency-bin index and
      approximate frequency in Hz for the peak attention bin.

Outputs:
- analysis/logo_runs_metrics.csv
- analysis/freq/*            (frequency window visualization)
- analysis/heatmaps/*        (|Z| heatmaps)
- analysis/features/*        (feature stats / PCA / profiles)
- analysis/roc/*             (ROC curves for best run)
- analysis/rollout_best_run/* (attention rollout and peak tokens for best run)

Update cfg.DATA_PATH and cfg.METADATA_PATH to your files.
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
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, auc)
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

from scipy.stats import f_oneway

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

    # Timing / sampling
    SAMPLE_RATE_HZ: float = 8e9                # 8 Gs/s (Δt = 0.125 ns)
    START_TIME_S: Optional[float] = None
    STOP_TIME_S:  Optional[float] = None

    # Window length behavior (legacy/fallback)
    TARGET_WINDOW_LEN: Optional[int] = None
    ENFORCE_TARGET_LEN: bool = False

    # Frequency axis labeling for plots (optional; Gen-3 ≈ 1–9 GHz with Δf ≈ 8 MHz)
    F0_HZ: Optional[float] = 1.0e9
    DF_HZ: Optional[float] = 8.0e6

    # Geometry
    EXPECTED_CHANNELS: int = 72

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
    RESULTS_CSV: str = os.path.join("analysis", "logo_runs_metrics.csv")

    # Analysis outputs
    N_HEATMAP_SAMPLES_PER_SHELL: int = 3

    # ---------- Window search controls ----------
    WIN_SEARCH_MIN_LEN_BINS: int = 24      # e.g., >= 3.0 ns
    WIN_SEARCH_MAX_LEN_BINS: int = 96      # e.g., <= 12.0 ns
    WIN_SEARCH_STEP_BINS: int = 1          # stride for start index & length
    WIN_SEARCH_MAX_CANDIDATES: int = 800   # to cap very large grids (randomly subsample)
    WIN_SEARCH_KFOLDS: int = 3             # CV folds for scoring a window
    WIN_SEARCH_RANDOM_STATE: int = 1337    # reproducible sampling of candidates

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

# ========================== IO & WINDOW (helpers) ==========================

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

def compute_window_indices(sample_rate_hz, total_len, target_len, start_time_s, stop_time_s, enforce=False):
    """Legacy helper if you ever want to use fixed time-based windows."""
    def clamp(a): return max(0, min(total_len, int(a)))
    if start_time_s is not None and stop_time_s is not None:
        start_idx = clamp(round(start_time_s * sample_rate_hz))
        stop_idx  = clamp(round(stop_time_s  * sample_rate_hz))
        if stop_idx <= start_idx:
            raise ValueError("STOP must be > START when both times are provided.")
        inferred = stop_idx - start_idx
        if target_len is not None and inferred != target_len:
            msg = f"[WARN] time-derived length={inferred} != TARGET_WINDOW_LEN={target_len}."
            if enforce:
                print(msg + " Enforcing target_len by adjusting stop_idx.")
                stop_idx = min(total_len, start_idx + target_len)
            else:
                print(msg + " Proceeding with time-derived length.")
        return int(start_idx), int(stop_idx)
    if start_time_s is not None:
        start_idx = clamp(round(start_time_s * sample_rate_hz))
        stop_idx = total_len if target_len is None else clamp(start_idx + target_len)
        if stop_idx - start_idx <= 0:
            raise ValueError("Empty window from START. Check TARGET_WINDOW_LEN.")
        return int(start_idx), int(stop_idx)
    if stop_time_s is not None:
        stop_idx = clamp(round(stop_time_s * sample_rate_hz))
        start_idx = 0 if target_len is None else clamp(stop_idx - target_len)
        if stop_idx - start_idx <= 0:
            raise ValueError("Empty window from STOP. Check TARGET_WINDOW_LEN.")
        return int(start_idx), int(stop_idx)
    if target_len is None:
        raise ValueError("Provide START/STOP times or a TARGET_WINDOW_LEN.")
    mid = total_len // 2
    start_idx = max(0, mid - target_len // 2)
    stop_idx = min(total_len, start_idx + target_len)
    start_idx = stop_idx - target_len
    return int(start_idx), int(stop_idx)

# ---------- NEW: fast features & window scoring (train-only) ----------

def quick_window_features(Xw: np.ndarray) -> np.ndarray:
    """
    Xw: [N, Tw, 72] complex. Collapse antennas, summarize over bins quickly.
    Returns [N, Fquick] with inexpensive, robust stats used only for window selection.
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
    return f  # shape (N,5)

def score_window_cv_auc(X: np.ndarray, y: np.ndarray, s_idx: int, e_idx: int,
                        train_idx: np.ndarray, n_splits: int = 3, seed: int = 0) -> float:
    """
    Train-only scoring: k-fold CV AUC on quick features from X[:, s:e, :].
    Uses a simple Logistic Regression classifier.
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

def generate_window_grid(T: int,
                         min_len: int, max_len: int,
                         step: int) -> List[Tuple[int,int]]:
    """Produce (s_idx, e_idx) pairs for window candidates."""
    grid: List[Tuple[int,int]] = []
    max_len = min(max_len, T)
    for L in range(min_len, max_len + 1, step):
        last_start = T - L
        if last_start < 0:
            break
        for s in range(0, last_start + 1, step):
            grid.append((s, s + L))
    return grid

def compute_optimal_windows_per_shell(X: np.ndarray,
                                      y: np.ndarray,
                                      A_comp: np.ndarray,
                                      seeds: Tuple[int,...]) -> Dict[str, Tuple[int,int]]:
    """
    For each shell S (held out later), choose the best window using only train shells (!= S).
    A single (s,e) per shell is chosen and reused across all runs for reproducibility.
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
        start_ns = (best_pair[0] / cfg.SAMPLE_RATE_HZ) * 1e9
        stop_ns  = (best_pair[1] / cfg.SAMPLE_RATE_HZ) * 1e9
        print(
            f"[WIN] Shell {S}: best [{best_pair[0]}:{best_pair[1]}) "
            f"len={best_pair[1]-best_pair[0]} ({start_ns:.3f}–{stop_ns:.3f} ns) "
            f"CV-AUC={best_auc:.4f}"
        )
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

# ========================== PATCH FEATURES (signal only) ==========================

def safe_stats(x):
    x = finite_or_zero(x)
    f = {}
    f["mean"] = float(np.mean(x)); f["std"] = float(np.std(x))
    f["min"] = float(np.min(x)); f["max"] = float(np.max(x))
    f["ptp"] = f["max"] - f["min"]
    f["q25"] = float(np.percentile(x,25)); f["q50"]=float(np.percentile(x,50)); f["q75"]=float(np.percentile(x,75))
    if x.size>1:
        m=f["mean"]; s=f["std"]+1e-12; z=(x-m)/s
        f["skew"] = float(np.mean(z**3)); f["kurt"]=float(np.mean(z**4)-3.0)
    else:
        f["skew"]=0.0; f["kurt"]=0.0
    return f

def shannon_entropy_hist(x, bins=16):
    x = finite_or_zero(x)
    mn, mx = float(np.min(x)), float(np.max(x)+1e-12)
    hist,_ = np.histogram(x, bins=bins, range=(mn,mx))
    p = hist.astype(float); s = p.sum()
    if s<=0:
        return 0.0
    p = p/s; p = p[p>0]
    return float(-(p*np.log(p)).sum()) if p.size else 0.0

def neighbor_corr_mag(mag72):
    mag72 = finite_or_zero(mag72)
    m = (mag72 - mag72.mean())/(mag72.std()+1e-12)
    return float(np.mean(np.abs(m*np.roll(m,-1))))

def prim_mst_total_length(points_2d):
    pts = finite_or_zero(points_2d)
    n=pts.shape[0]
    in_tree = np.zeros(n, dtype=bool)
    dist = np.full(n, np.inf)
    dist[0]=0.0
    total=0.0
    for _ in range(n):
        u=int(np.argmin(dist))
        du=np.linalg.norm(pts-pts[u],axis=1)
        du=finite_or_zero(du)
        total += float(dist[u] if np.isfinite(dist[u]) else 0.0)
        in_tree[u]=True
        dist[u]=np.inf
        dist=np.minimum(dist, np.where(in_tree, np.inf, du))
    return float(total) if np.isfinite(total) else 0.0

def topology_summaries(z72):
    z72 = finite_complex_or_zero(z72)
    pts = np.stack([z72.real, z72.imag], axis=1)
    mst_len = prim_mst_total_length(pts)
    d = np.linalg.norm(pts[:,None,:]-pts[None,:,:],axis=2)
    d=finite_or_zero(d)
    eps_list = [np.percentile(d,p) for p in (10,20,30,40,50)]
    eps_list = [0.0 if not np.isfinite(e) else float(e) for e in eps_list]
    comps=[]
    for eps in eps_list:
        if eps<=0:
            comps.append(72.0); continue
        seen=np.zeros(72,dtype=bool); c=0
        for i in range(72):
            if not seen[i]:
                c+=1; stack=[i]; seen[i]=True
                while stack:
                    u=stack.pop()
                    nbrs=np.where((d[u]<=eps)&(~seen))[0]
                    seen[nbrs]=True
                    stack.extend(nbrs.tolist())
        comps.append(float(c))
    return {"mst_len":mst_len,"cc_eps10":comps[0],"cc_eps20":comps[1],"cc_eps30":comps[2],
            "cc_eps40":comps[3],"cc_eps50":comps[4]}

def circ_diff(a): return np.roll(a,-1)-a
def circ_diff2(a): return circ_diff(circ_diff(a))

def circ_fft_lowfreq(x,k=4):
    x = finite_or_zero(x)
    spec=np.fft.rfft(x,n=len(x))
    mag=np.abs(spec)[1:k+1]
    return finite_or_zero(mag)

def phase_circular_stats(ang):
    ang = finite_or_zero(ang)
    C=np.mean(np.cos(ang)); S=np.mean(np.sin(ang))
    R=np.hypot(C,S)
    return {"phase_R":float(R),"phase_var":float(1-R)}

def zero_crossings(x):
    x=finite_or_zero(x)
    return float(np.sum(np.diff(np.signbit(x))!=0))

def autocorr_lag1(x):
    x=finite_or_zero(x)
    x=x-x.mean()
    d=np.dot(x,x)+1e-12
    return float(np.dot(x[:-1],x[1:])/d)

def extract_patch_features_one_timepoint(z_t_72: np.ndarray) -> Dict[str,float]:
    """
    Rich feature extractor for one timepoint (all 72 antennas).
    These become per-token features for the transformer.
    """
    z_t_72 = finite_complex_or_zero(z_t_72)
    re=z_t_72.real; im=z_t_72.imag; mag=np.abs(z_t_72); ang=np.angle(z_t_72); ang_unw=np.unwrap(ang)
    f={}
    for name,arr in (("re",re),("im",im),("mag",mag),("ang",ang)):
        for k,v in safe_stats(arr).items():
            f[f"{name}_k_{k}"]=v
    f["mag_energy"]=float(np.sum(mag**2))
    f["mag_entropy"]=shannon_entropy_hist(mag,16)
    f["neighbor_corr_mag"]=neighbor_corr_mag(mag)
    f.update(topology_summaries(z_t_72))
    for name,arr in (("mag",mag),("ang",ang_unw),("re",re),("im",im)):
        d1,d2=circ_diff(arr),circ_diff2(arr)
        for k,v in safe_stats(d1).items():
            f[f"{name}_d1_{k}"]=v
        for k,v in safe_stats(d2).items():
            f[f"{name}_d2_{k}"]=v
    for name,arr in (("mag",mag),("re",re),("im",im)):
        lf=circ_fft_lowfreq(arr,k=4)
        for i,v in enumerate(lf,1):
            f[f"{name}_fftk{i}"]=float(v)
    f.update(phase_circular_stats(ang))
    f["ang_unw_slope_var"]=float(np.var(np.diff(ang_unw)))
    for name,arr in (("re",re),("im",im),("mag",mag)):
        f[f"{name}_zcr"]=zero_crossings(arr)
        f[f"{name}_ac1"]=autocorr_lag1(arr)
    for k,v in list(f.items()):
        if not np.isfinite(v):
            f[k]=0.0
    return f

def build_signal_seq_features(Xw: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Build [N, T, Fsig] feature sequences from complex windowed data [N, T, 72]."""
    N,T,C = Xw.shape
    probe = extract_patch_features_one_timepoint(Xw[0,0])
    keys = sorted(probe.keys())
    Fsig = len(keys)
    seq = np.zeros((N,T,Fsig), dtype=np.float32)
    for i in range(N):
        for t in range(T):
            fdict = extract_patch_features_one_timepoint(Xw[i,t])
            seq[i,t] = np.array([fdict[k] for k in keys], dtype=np.float32)
    return seq, keys

# ========================== METADATA (safe) ==========================

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
        "numX": np.nan_to_num(numX, 0.0,0.0,0.0),
        "year": years
    }
    return np.array(A_list, dtype=object), meta_struct, np.array(phant_ids, dtype=object)

# ========================== DATASET ==========================

class PatchSequenceDataset(Dataset):
    """Simple (sequence, label) dataset wrapper."""
    def __init__(self, Xseq: np.ndarray, y: np.ndarray):
        self.X = Xseq.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ========================== RoPE Transformer (with attn return) ==========================

def rotate_half(x):
    """Helper for RoPE rotation."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)

def apply_rope(q, k, cos, sin):
    """Apply rotary position embeddings to q,k."""
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class RotaryEmbedding(nn.Module):
    """Precomputes cos/sin tables for RoPE."""
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
    """
    Multi-head attention with optional RoPE:
    - If use_rope=True: apply RoPE to q,k.
    - If use_rope=False: standard MHA (no positional rotation).
    """
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
            return out, attn  # [B,heads,T,T]
        return out

class EncoderLayerRoPE(nn.Module):
    """Transformer encoder layer with optional RoPE-based attention."""
    def __init__(self, d_model, nhead, d_ff, dropout, rope_max_len=512, use_rope: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MHA_RoPE(d_model, nhead, dropout, rope_max_len, use_rope=use_rope)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff,d_model)
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
    """Stack of encoder layers + final LayerNorm."""
    def __init__(self, n_layers, d_model, nhead, d_ff, dropout,
                 rope_max_len=512, use_rope: bool = True):
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
    """
    Sequence classifier:
    - Input: [B,T,F] features per bin.
    - Optional CLS token.
    - Transformer encoder with optional RoPE.
    - MLP head -> num_classes.
    """
    def __init__(self, in_dim, d_model, nhead, n_layers, d_ff, dropout,
                 use_cls=True, num_classes=2, max_len=512, use_rope: bool = True):
        super().__init__()
        self.use_cls = use_cls
        self.proj = nn.Linear(in_dim, d_model)
        if use_cls:
            self.cls = nn.Parameter(torch.zeros(1,1,d_model))
            nn.init.trunc_normal_(self.cls, std=0.02)
        self.enc = EncoderRoPE(
            n_layers, d_model, nhead, d_ff, dropout,
            rope_max_len=max_len, use_rope=use_rope
        )
        self.head = nn.Sequential(
            nn.Linear(d_model,d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model,num_classes)
        )

    def forward(self, x, return_attn: bool=False):  # [B,T,F]
        B,T,Fd = x.shape
        h = self.proj(x)
        if self.use_cls:
            cls = self.cls.expand(B, -1, -1)
            h = torch.cat([cls, h], dim=1)  # [B,T+1,D]
        if return_attn:
            h, attns = self.enc(h, return_attn=True)   # attns: list of L elements, each [B,H,S,S]
        else:
            h = self.enc(h, return_attn=False)
        pooled = h[:,0] if self.use_cls else h.mean(1)
        logits = self.head(pooled)
        if return_attn:
            return logits, attns
        return logits

# ========================== TRAIN / EVAL ==========================

def train_one_fold(tr_loader, va_loader, model, device, max_epochs, patience, lr, wd):
    """
    Train one LOGO fold.
    Returns:
    - best model (by val loss)
    - total training time in seconds
    - number of epochs actually run (including early stopping)
    """
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
        for xb,yb in tr_loader:
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()*xb.size(0)
        tr_loss /= len(tr_loader.dataset)

        model.eval()
        va_loss=0.0
        with torch.no_grad():
            for xb,yb in va_loader:
                xb=xb.to(device); yb=yb.to(device)
                va_loss += crit(model(xb), yb).item()*xb.size(0)
        va_loss /= len(va_loader.dataset)
        print(f"  Epoch {ep:03d} | Train {tr_loss:.4f} | Val {va_loss:.4f}")

        if va_loss + 1e-6 < best:
            best = va_loss
            best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            wait=0
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
    """Standard ECE: average |acc - conf| over bins."""
    y_true = y_true.astype(int)
    conf = np.maximum(y_prob, 1.0 - y_prob)
    pred = (y_prob >= 0.5).astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        if i < n_bins-1:
            m = (conf >= lo) & (conf < hi)
        else:
            m = (conf >= lo) & (conf <= hi)
        if not np.any(m):
            continue
        bin_conf = conf[m].mean()
        bin_acc  = (pred[m] == y_true[m]).mean()
        ece += np.sum(m) / len(y_true) * abs(bin_acc - bin_conf)
    return float(ece)

def attention_rollout_from_attns(attn_list: List[torch.Tensor]) -> List[np.ndarray]:
    """
    Compute attention rollout from CLS token to all tokens.
    Returns a list of vectors, one per sample: [S-1] (excluding CLS).
    """
    heads_mean = []
    for A in attn_list:
        A = A.detach().cpu().numpy()           # [B,H,S,S]
        A = A.mean(axis=1)                     # [B,S,S]
        S = A.shape[-1]
        A = A + np.eye(S, dtype=A.dtype)[None, :, :]
        A = A / (A.sum(axis=-1, keepdims=True) + 1e-12)
        heads_mean.append(A)
    rollout = heads_mean[0]
    for A in heads_mean[1:]:
        rollout = np.matmul(rollout, A)        # [B,S,S]
    cls_to_tokens = rollout[:, 0, 1:]          # [B, S-1]
    cls_to_tokens = cls_to_tokens / (cls_to_tokens.sum(axis=1, keepdims=True) + 1e-12)
    return [cls_to_tokens[i] for i in range(cls_to_tokens.shape[0])]

def eval_loader(loader, model, device, return_attn: bool=False):
    """
    Evaluate model on a dataloader.
    Returns:
    - metrics dict including acc/pre/rec/f1/auc/ece/cm
    - y_true, y_pred, y_prob arrays
    - if return_attn=True: also rollout vectors
    Also measures inference time and adds:
    - infer_time_total_s
    - infer_time_ms_per_sample
    """
    model.eval()
    y_true=[]; y_pred=[]; y_prob=[]; rollout_all=[]
    start_time = time.time()
    with torch.no_grad():
        for xb,yb in loader:
            xb=xb.to(device)
            if return_attn:
                logits, attns = model(xb, return_attn=True)
                prob = F.softmax(logits, dim=-1)[:,1].cpu().numpy()
                rollout = attention_rollout_from_attns(attns)
                rollout_all.extend([r for r in rollout])
            else:
                logits = model(xb)
                prob = F.softmax(logits, dim=-1)[:,1].cpu().numpy()
            pred = (prob>=0.5).astype(int)
            y_true.append(yb.numpy()); y_pred.append(pred); y_prob.append(prob)
    total_infer_time = time.time() - start_time

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)

    metrics = {
        "acc": accuracy_score(y_true,y_pred),
        "pre": precision_score(y_true,y_pred),
        "rec": recall_score(y_true,y_pred),
        "f1":  f1_score(y_true,y_pred),
        "auc": roc_auc_score(y_true,y_prob) if len(np.unique(y_true))==2 else float("nan"),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=15),
        "cm":  confusion_matrix(y_true,y_pred),
        "infer_time_total_s": float(total_infer_time),
        "infer_time_ms_per_sample": float(1000.0 * total_infer_time / len(y_true))
    }
    if return_attn:
        return metrics, y_true, y_pred, y_prob, rollout_all
    return metrics, y_true, y_pred, y_prob

# ========================== ANALYSIS & VISUALIZATION ==========================

def _ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def _freq_axis_labels(T, start_idx, f0_full, df):
    """
    Build x axis for frequency plots.
    If f0_full and df are given, returns physical frequencies in Hz.
    Otherwise returns simple bin indices.
    """
    if f0_full is None or df is None:
        x = np.arange(T)
        xlabel = "Frequency bin"
    else:
        x = f0_full + (start_idx + np.arange(T)) * df
        xlabel = "Frequency (Hz)"
    return x, xlabel

def visualize_frequency_window_full_context(X, s_idx, e_idx, f0_full, df, outdir):
    """Visualize global spectrum and chosen window on top, plus window mean |Z|."""
    _ensure_dir(outdir)
    N, F, A = X.shape
    mag = np.abs(X)                      # (N,F,A)
    mean_over_NA = mag.mean(axis=(0,2)) # (F,)
    win_len = e_idx - s_idx

    # full spectrum with window highlighted
    fig, ax = plt.subplots(figsize=(9,3.2))
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

    # mean |Z| within the window
    mean_over_NA_win = mag[:, s_idx:e_idx, :].mean(axis=(0,2))  # (win_len,)
    fig, ax = plt.subplots(figsize=(7.2,3.2))
    x_win, xlabel = _freq_axis_labels(e_idx - s_idx, s_idx, f0_full, df)
    ax.plot(x_win, mean_over_NA_win, marker='o', ms=3, lw=1)
    ax.set_xlabel(xlabel); ax.set_ylabel('Mean |Z| over N, antennas')
    ax.set_title(f'Mean |Z| across window (len={win_len})')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "freq_window_mean_abs.png"), dpi=200)
    plt.close(fig)

def heatmaps_per_shell_examples(Xw, A_comp, shells, outdir, per_shell=3):
    """Save |Z| heatmaps per shell for a few example samples."""
    _ensure_dir(outdir)
    mag = np.abs(Xw)  # (N,T,72)
    T = mag.shape[1]
    for s in shells:
        idx = np.where(A_comp == s)[0]
        if len(idx) == 0:
            continue
        chosen = idx[:min(per_shell, len(idx))]
        for k, i in enumerate(chosen, 1):
            M = mag[i].T  # (72,T)
            fig, ax = plt.subplots(figsize=(5.6,4.2))
            im = ax.imshow(M, aspect='auto', origin='lower')
            ax.set_xlabel(f'Window bin (0..{T-1})')
            ax.set_ylabel('Antenna index (0..71)')
            ax.set_title(f'|Z| heatmap — shell {s} — sample {i} — len={T}')
            fig.colorbar(im, ax=ax, shrink=0.86, label='|Z|')
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"heatmap_abs_shell_{s}_sample_{i}.png"), dpi=220)
            plt.close(fig)

def feature_stats_by_shell(seq_sig, A_comp, keys, outdir):
    """
    Compute per-shell feature summaries and ANOVA, plus PCA visualization and profiles.
    This is mostly unchanged, useful for interpretation in the paper.
    """
    _ensure_dir(outdir)
    N, T, F = seq_sig.shape
    shells = sorted(list(set(A_comp)))
    X_sample = seq_sig.mean(axis=1)
    shell_means = {s: X_sample[A_comp==s].mean(axis=0) for s in shells}

    def cohend(a, b):
        a = a.astype(np.float64); b = b.astype(np.float64)
        na, nb = len(a), len(b)
        sa2 = np.var(a, ddof=1) if na>1 else 0.0
        sb2 = np.var(b, ddof=1) if nb>1 else 0.0
        s = np.sqrt(((na-1)*sa2 + (nb-1)*sb2) / max(na+nb-2,1))
        return (np.mean(a) - np.mean(b)) / (s + 1e-12)
    d_by_shell = {}
    for s in shells:
        m = (A_comp==s)
        oth = ~m
        d = np.array([cohend(X_sample[m, j], X_sample[oth, j]) for j in range(F)])
        d_by_shell[s] = d

    Fvals = np.zeros(F); pvals = np.ones(F)
    groups = [X_sample[A_comp==s] for s in shells]
    for j in range(F):
        try:
            Fv, pv = f_oneway(*[g[:, j] for g in groups if len(g)>=2])
        except Exception:
            Fv, pv = np.nan, np.nan
        Fvals[j], pvals[j] = Fv, pv

    featnames = np.array(keys)
    with open(os.path.join(outdir, "feature_means_by_shell.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["feature"] + shells)
        for j,name in enumerate(featnames):
            row = [name] + [float(shell_means[s][j]) for s in shells]
            w.writerow(row)
    with open(os.path.join(outdir, "feature_cohens_d_by_shell.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["feature"] + [f"d_{s}" for s in shells])
        for j,name in enumerate(featnames):
            row = [name] + [float(d_by_shell[s][j]) for s in shells]
            w.writerow(row)
    with open(os.path.join(outdir, "feature_anova_F_p.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["feature","F","p"])
        for j,name in enumerate(featnames):
            w.writerow([name, float(Fvals[j]), float(pvals[j])])

    order = np.argsort(-(np.nan_to_num(Fvals, nan=-np.inf)))[:20]
    fig, ax = plt.subplots(figsize=(8.5,4.5))
    ax.barh(np.arange(len(order)), Fvals[order][::-1])
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(featnames[order][::-1], fontsize=8)
    ax.set_xlabel("ANOVA F-statistic")
    ax.set_title("Top features by between-shell variance")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "top_features_by_ANOVA.png"), dpi=220)
    plt.close(fig)

    from sklearn.preprocessing import StandardScaler
    X_std = StandardScaler().fit_transform(X_sample)
    Z = PCA(n_components=2, random_state=0).fit_transform(X_std)
    fig, ax = plt.subplots(figsize=(6,5))
    for s in shells:
        m = (A_comp==s)
        ax.scatter(Z[m,0], Z[m,1], s=18, alpha=0.85, label=s)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("PCA on per-sample mean features")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "pca_features_by_shell.png"), dpi=220)
    plt.close(fig)

    sel = [nm for nm in ["mag_energy","phase_R","mst_len","mag_fftk1","mag_fftk2","re_zcr","mag_ac1"] if nm in keys]
    if len(sel)>0:
        idx = {k:i for k,i in zip(keys, range(len(keys)))}
        for name in sel:
            j = idx[name]
            fig, ax = plt.subplots(figsize=(7.6,3.6))
            x = np.arange(T)
            for s in shells:
                m = (A_comp==s)
                mu = seq_sig[m, :, j].mean(axis=0)
                se = seq_sig[m, :, j].std(axis=0, ddof=1) / np.sqrt(max(m.sum(),1))
                ax.plot(x, mu, label=s)
                ax.fill_between(x, mu-se, mu+se, alpha=0.2)
            ax.set_xlabel(f"Window bin (0..{T-1})")
            ax.set_ylabel(name)
            ax.set_title(f"{name}: profile across bins (len={T})")
            ax.legend(frameon=True)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, f"profile_bins_{name}.png"), dpi=220)
            plt.close(fig)

def aggregate_and_plot_rollout(attn_pattern="attention_*_rollout_*.npy",
                               outdir="analysis",
                               shell_windows: Optional[Dict[str, Tuple[int,int]]] = None):
    """
    Aggregate attention rollout npy files and plot:
    - Mean±SEM rollout per shell.
    - Overall overlay.
    Also writes a CSV with the most-attended token per shell.
    If shell_windows is provided, also outputs:
        - absolute frequency bin index
        - approximate frequency (Hz) of the peak bin.
    """
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

    # Handle different sequence lengths per shell
    fig, ax = plt.subplots(figsize=(7.8,3.6))
    for shell, li in shell_curves.items():
        V = np.vstack(li)          # [Nshell, T_shell]
        T_shell = V.shape[1]
        x_shell = np.arange(T_shell)

        mu = V.mean(axis=0)
        se = V.std(axis=0, ddof=1) / np.sqrt(max(V.shape[0],1))

        ax.plot(x_shell, mu, label=shell)
        ax.fill_between(x_shell, mu-se, mu+se, alpha=0.2)

        # Per-shell individual plot
        fig2, ax2 = plt.subplots(figsize=(7.2,3.2))
        ax2.plot(x_shell, mu, lw=1.5)
        ax2.fill_between(x_shell, mu-se, mu+se, alpha=0.25)
        ax2.set_xlabel("Token index")
        ax2.set_ylabel("Rollout importance")
        ax2.set_title(f"Attention rollout — {shell} (len={T_shell})")
        fig2.tight_layout()
        fig2.savefig(os.path.join(outdir, f"rollout_profile_{shell}.png"), dpi=220)
        plt.close(fig2)

        # Peak attention token index
        peak_idx = int(np.argmax(mu))
        peak_val = float(mu[peak_idx])

        # Map token index -> absolute frequency bin and Hz (if possible)
        abs_bin_idx = None
        freq_hz = None
        if shell_windows is not None and shell in shell_windows:
            s_idx, e_idx = shell_windows[shell]
            abs_bin_idx = int(s_idx + peak_idx)
            if cfg.F0_HZ is not None and cfg.DF_HZ is not None:
                freq_hz = float(cfg.F0_HZ + abs_bin_idx * cfg.DF_HZ)

        peak_rows.append([
            shell,
            peak_idx,
            peak_val,
            abs_bin_idx if abs_bin_idx is not None else "",
            freq_hz if freq_hz is not None else ""
        ])

    ax.set_xlabel("Token index")
    ax.set_ylabel("Rollout importance")
    ax.set_title("Attention rollout (mean±SEM)")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "rollout_profiles_all.png"), dpi=220)
    plt.close(fig)

    # Save peak token table
    with open(os.path.join(outdir, "attention_peak_tokens.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["shell", "peak_token_index", "peak_attention_value",
                    "absolute_freq_bin_index", "approx_freq_hz"])
        for row in peak_rows:
            w.writerow(row)
    print("[rollout] Saved attention_peak_tokens.csv with per-shell most-attended bin.")

def plot_roc_curves_for_best_run(best_run_idx: int,
                                 all_runs_ytrue: List[np.ndarray],
                                 all_runs_yprob: List[np.ndarray],
                                 all_runs_shell_ytrue: Dict[str, List[np.ndarray]],
                                 all_runs_shell_yprob: Dict[str, List[np.ndarray]],
                                 outdir: str):
    """
    Plot ROC curves for the best run:
    - Overall ROC.
    - Per-shell ROC (A2, A3, A14, A16).
    """
    _ensure_dir(outdir)
    r = best_run_idx  # 0-based
    y_true = all_runs_ytrue[r]
    y_prob = all_runs_yprob[r]

    # Overall ROC
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

    # Per-shell ROC
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
    """
    Plot a single ROC figure for the best run, showing one ROC curve per shell
    (A2, A3, A14, A16) plus a random-classifier reference line.
    Axes are in percent (0–100%).
    """
    _ensure_dir(outdir)
    r = best_run_idx  # 0-based

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

    # Random classifier reference
    ax.plot([0, 100], [0, 100], linestyle="--", linewidth=1.2,
            label="Random classifier")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
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
    """Handle sklearn OneHotEncoder API changes across versions."""
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown="ignore")

def build_meta_fit_transform(F_comp: np.ndarray, numX: np.ndarray, years: np.ndarray,
                             train_idx: np.ndarray, apply_idx: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Fit metadata encoders/scalers on train_idx only, then transform apply_idx.
    Returns:
    - meta_feats: [len(apply_idx), Dmeta]
    - meta_info: dict with fitted encoders/scalers (not used further here)
    """
    ohe = _ohe_fit()
    ohe.fit(F_comp[train_idx].reshape(-1,1))
    F_ohe = ohe.transform(F_comp[apply_idx].reshape(-1,1)).astype(np.float32)

    num_scaler = StandardScaler().fit(numX[train_idx])
    num_scaled = num_scaler.transform(numX[apply_idx]).astype(np.float32)

    mats = [F_ohe, num_scaled]
    meta_info = {"ohe_F": ohe, "num_scaler": num_scaler}

    if cfg.INCLUDE_YEAR:
        year_ohe = _ohe_fit()
        year_ohe.fit(years[train_idx].reshape(-1,1))
        year_feats = year_ohe.transform(years[apply_idx].reshape(-1,1)).astype(np.float32)
        mats.append(year_feats)
        meta_info["ohe_year"] = year_ohe

    meta_feats = np.concatenate(mats, axis=1)
    return meta_feats, meta_info

def scale_signal_seq_per_fold(seq_sig: np.ndarray, train_idx: np.ndarray, apply_idx: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardize signal features per fold:
    - Fit scaler on all timepoints from train_idx.
    - Apply to apply_idx.
    """
    N,T,F = seq_sig.shape
    tr_flat = seq_sig[train_idx].reshape(-1, F)
    sig_scaler = StandardScaler().fit(tr_flat)
    out = seq_sig[apply_idx].reshape(-1, F)
    out = sig_scaler.transform(out).astype(np.float32)
    return out.reshape(len(apply_idx), T, F), sig_scaler

def fuse_seq_and_meta(seq_scaled, meta_feats):
    """Concatenate sequence features and (tiled) meta features along feature dimension."""
    B,T,F = seq_scaled.shape
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
                       save_rollout_prefix: str | None = None) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    Full LOGO pipeline for one hold-out shell:
    - Train on shells != shell, early stop by val loss.
    - Evaluate on held-out shell.
    - Returns:
        metrics dict (incl. timing),
        y_true (test),
        y_prob (test).
    """
    set_seed(seed)

    N,T,Fsig = seq_sig.shape
    F_comp = meta_struct["F_comp"]
    numX   = meta_struct["numX"]
    years  = meta_struct["year"]

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

    assert not np.any(A_comp[tr_idx] == shell)
    assert not np.any(A_comp[va_idx] == shell)
    assert np.all(A_comp[test_idx] == shell)

    # Scale signal per fold
    seq_tr_scaled, _ = scale_signal_seq_per_fold(seq_sig, tr_idx, tr_idx)
    seq_va_scaled, _ = scale_signal_seq_per_fold(seq_sig, tr_idx, va_idx)
    seq_te_scaled, _ = scale_signal_seq_per_fold(seq_sig, tr_idx, test_idx)

    # Metadata encoding
    meta_tr, _ = build_meta_fit_transform(F_comp, numX, years, tr_idx, tr_idx)
    meta_va, _ = build_meta_fit_transform(F_comp, numX, years, tr_idx, va_idx)
    meta_te, _ = build_meta_fit_transform(F_comp, numX, years, tr_idx, test_idx)

    # Fuse seq + meta
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
    print(
        f"[{shell}] Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M "
        f"(in_dim={in_dim}, use_rope={cfg.USE_ROPE})"
    )

    # Train with timing
    model, train_time_s, epochs_ran = train_one_fold(
        train_loader, val_loader, model, device,
        cfg.MAX_EPOCHS, cfg.PATIENCE, cfg.LR, cfg.WEIGHT_DECAY
    )

    # Evaluate with attention rollout + inference timing
    metrics, y_true, y_pred, y_prob, rollout = eval_loader(
        test_loader, model, device, return_attn=True
    )
    metrics["train_time_s"] = float(train_time_s)
    metrics["train_epochs"] = int(epochs_ran)

    print(
        f"[{shell}] ACC={metrics['acc']:.3f} PRE={metrics['pre']:.3f} REC={metrics['rec']:.3f} "
        f"F1={metrics['f1']:.3f} AUC={metrics['auc']:.4f} ECE={metrics['ece']:.4f}"
    )
    print(
        f"[{shell}] Confusion:\n{metrics['cm']}"
    )
    print(
        f"[{shell}] train_time={train_time_s:.2f}s over {epochs_ran} epochs | "
        f"infer={metrics['infer_time_ms_per_sample']:.3f} ms/sample"
    )

    # Save rollout vectors for later aggregation
    if save_rollout_prefix:
        np.save(f"{save_rollout_prefix}_rollout_{shell}.npy",
                np.array(rollout, dtype=object))
    return metrics, y_true, y_prob

# ========================== MAIN (5 runs) ==========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] USE_ROPE = {cfg.USE_ROPE}")

    # Create base analysis directory early
    _ensure_dir(cfg.OUT_DIR)

    # Load data + meta
    X = load_complex_data(cfg.DATA_PATH, cfg.IS_PICKLE)   # [N, T_full, 72]
    with open(cfg.METADATA_PATH, "rb") as f:
        meta = _silenced_pickle_load(f)
    if len(meta) != X.shape[0]:
        raise ValueError(f"Metadata length {len(meta)} != samples {X.shape[0]}")

    # Labels & metadata parse
    y = make_labels(meta)
    A_comp, meta_struct, phant_ids = extract_safe_metadata(meta)

    # --------- Compute optimal (s,e) per shell using TRAIN-only CV ----------
    shell_windows = compute_optimal_windows_per_shell(X, y, A_comp, seeds=cfg.RUN_SEEDS)

    # For visualization, pick one shell's window (e.g., A2) or the first available
    vis_shell = "A2" if "A2" in shell_windows else next(iter(shell_windows))
    s_idx_vis, e_idx_vis = shell_windows[vis_shell]
    outdir = cfg.OUT_DIR
    visualize_frequency_window_full_context(
        X, s_idx_vis, e_idx_vis, cfg.F0_HZ, cfg.DF_HZ, outdir=os.path.join(outdir, "freq")
    )

    # Prepare results aggregation
    rows = []
    per_run_overall = []
    per_run_shells = {s: [] for s in ADIPOSE_LIST}
    per_run_shells_acc = {s: [] for s in ADIPOSE_LIST}
    per_run_shells_f1  = {s: [] for s in ADIPOSE_LIST}
    per_run_shells_ece = {s: [] for s in ADIPOSE_LIST}

    # For ROC curves / best run selection
    all_runs_ytrue = []
    all_runs_yprob = []
    all_runs_shell_ytrue = {s: [] for s in ADIPOSE_LIST}
    all_runs_shell_yprob = {s: [] for s in ADIPOSE_LIST}

    # 5 runs with different seeds
    for run_idx, seed in enumerate(cfg.RUN_SEEDS, start=1):
        print(f"\n===== RUN {run_idx} (seed={seed}) — heads={cfg.N_HEAD}, layers={cfg.NUM_LAYERS}, use_rope={cfg.USE_ROPE} =====")
        set_seed(seed)
        per_shell = {}
        all_true = []; all_prob = []
        save_prefix = f"attention_{run_idx}"

        for shell in ADIPOSE_LIST:
            # Apply the chosen window for THIS shell
            s_idx, e_idx = shell_windows.get(shell, (s_idx_vis, e_idx_vis))
            Xw_shell = X[:, s_idx:e_idx, :]                     # [N, Tw, 72]
            Tw = Xw_shell.shape[1]
            print(
                f"[{shell}] Using window [{s_idx}:{e_idx}) len={Tw} bins "
                f"({s_idx/cfg.SAMPLE_RATE_HZ*1e9:.3f}–{e_idx/cfg.SAMPLE_RATE_HZ*1e9:.3f} ns)"
            )
            # Build raw (unscaled) sequence features from signal
            seq_sig, keys = build_signal_seq_features(Xw_shell)  # [N,Tw,Fsig]

            # Per-run feature analysis (do once per run, on first shell)
            if shell == ADIPOSE_LIST[0]:
                heatmaps_per_shell_examples(
                    Xw_shell, A_comp, ADIPOSE_LIST,
                    outdir=os.path.join(outdir, "heatmaps"),
                    per_shell=cfg.N_HEATMAP_SAMPLES_PER_SHELL
                )
                feature_stats_by_shell(
                    seq_sig, A_comp, keys,
                    outdir=os.path.join(outdir, "features")
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

        # Overall metrics for this run (all shells concatenated)
        y_true = np.concatenate(all_true)
        y_prob = np.concatenate(all_prob)
        all_runs_ytrue.append(y_true)
        all_runs_yprob.append(y_prob)
        overall_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true))==2 else float("nan")

        per_run_overall.append(overall_auc)
        for s in ADIPOSE_LIST:
            per_run_shells[s].append(per_shell[s]["auc"])
            per_run_shells_acc[s].append(per_shell[s]["acc"])
            per_run_shells_f1[s].append(per_shell[s]["f1"])
            per_run_shells_ece[s].append(per_shell[s]["ece"])

        # Row for CSV (per-run metrics); we also add timing info per shell.
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
        print(
            f"[RUN {run_idx}] overall AUC={overall_auc:.4f} | "
            f"A2={per_shell['A2']['auc']:.4f} A3={per_shell['A3']['auc']:.4f} "
            f"A14={per_shell['A14']['auc']:.4f} A16={per_shell['A16']['auc']:.4f}"
        )

    # Save per-run CSV (ensure directory exists to avoid PermissionError)
    results_dir = os.path.dirname(cfg.RESULTS_CSV)
    if results_dir:
        _ensure_dir(results_dir)
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

    # Determine best run (by overall AUC) and plot ROC curves
    best_run_idx = int(np.argmax(np.array(per_run_overall)))  # 0-based
    print(
        f"[ROC] Best run index (0-based) = {best_run_idx}, "
        f"overall_auc={per_run_overall[best_run_idx]:.4f}"
    )
    roc_outdir = os.path.join(cfg.OUT_DIR, "roc")
    plot_roc_curves_for_best_run(
        best_run_idx,
        all_runs_ytrue,
        all_runs_yprob,
        all_runs_shell_ytrue,
        all_runs_shell_yprob,
        outdir=roc_outdir
    )
    # New: single figure with all shells' ROC
    plot_roc_shells_single_figure(
        best_run_idx,
        all_runs_shell_ytrue,
        all_runs_shell_yprob,
        outdir=roc_outdir
    )

    # Mean ± std summary (for tables in the paper)
    def mstd(x):
        x=np.array(x, float)
        return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x)>1 else 0.0
    ov_m, ov_s = mstd(per_run_overall)
    print("\n=== FINAL (5 runs) mean ± std ===")
    print(f"Overall AUC : {ov_m:.4f} ± {ov_s:.4f}")
    for s in ADIPOSE_LIST:
        am,as_ = mstd(per_run_shells[s])
        acm,acs= mstd(per_run_shells_acc[s])
        fm,fs  = mstd(per_run_shells_f1[s])
        em,es  = mstd(per_run_shells_ece[s])
        print(
            f"{s}: AUC {am:.4f}±{as_:.4f} | ACC {acm:.4f}±{acs:.4f} "
            f"| F1 {fm:.4f}±{fs:.4f} | ECE {em:.4f}±{es:.4f}"
        )

    # Attention rollout summary for BEST RUN only
    best_run_pattern = f"attention_{best_run_idx+1}_rollout_*.npy"
    aggregate_and_plot_rollout(
        attn_pattern=best_run_pattern,
        outdir=os.path.join(cfg.OUT_DIR, "rollout_best_run"),
        shell_windows=shell_windows
    )

if __name__ == "__main__":
    # Optional CLI: --no_rope to disable RoPE (ablation)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no_rope", action="store_true",
        help="Disable RoPE (use standard MHA without rotary embeddings)."
    )
    args = parser.parse_args()
    cfg.USE_ROPE = not args.no_rope
    main()
