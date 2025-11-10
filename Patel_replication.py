import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, roc_auc_score, precision_score,
                             f1_score, confusion_matrix, matthews_corrcoef)
from datetime import datetime
import logging

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UMBMID_RF")

# ----------------------------
# Paper-aligned preprocessing
# ----------------------------
def idft_time_window_features(X_complex,
                              f_low_hz=1e9, f_high_hz=8e9,
                              start_time_ns=0.71, stop_time_ns=5.71):
    """
    X_complex: np.ndarray with shape (N_samples, N_freqs, N_pos)
               complex-valued frequency-domain S11 data (evenly spaced between f_low and f_high)
    Steps (paper):
      - IDFT along frequency axis
      - take real part
      - window to [0.71 ns, 5.71 ns] -> keep exactly 35 samples
      - flatten to 35*72 = 2520 features
    Returns:
      X_flat: (N_samples, 2520) float64
    """
    assert X_complex.ndim == 3, "Expected (samples, freqs, positions)"
    n_samples, n_freqs, n_pos = X_complex.shape

    # Frequency spacing (assume evenly spaced between f_low_hz and f_high_hz, inclusive of endpoints)
    df = (f_high_hz - f_low_hz) / (n_freqs - 1)  # Hz
    # IDFT time grid properties
    T = 1.0 / df                       # total period in time domain
    dt = T / n_freqs                   # time step
    # Compute time indices for the paper's window
    start_time_s = start_time_ns * 1e-9
    stop_time_s  = stop_time_ns  * 1e-9
    start_idx = int(np.round(start_time_s / dt))
    stop_idx  = int(np.round(stop_time_s  / dt))
    # Safety: enforce bounds and exact 35 samples if possible
    # If stop_idx - start_idx + 1 != 35 due to rounding, we adjust stop_idx to get 35 samples.
    desired_len = 35
    if stop_idx - start_idx + 1 != desired_len:
        stop_idx = start_idx + desired_len - 1

    start_idx = max(0, start_idx)
    stop_idx = min(n_freqs - 1, stop_idx)
    if stop_idx - start_idx + 1 != desired_len:
        raise ValueError(f"Window length is {stop_idx - start_idx + 1}, expected {desired_len}. "
                         f"Check frequency grid assumptions.")

    # IDFT: along axis=1 (frequency axis). Use np.fft.ifft with n=n_freqs (already n points).
    # Paper: use REAL PART ONLY after IDFT.
    X_time = np.fft.ifft(X_complex, axis=1)
    X_time_real = np.real(X_time)

    # Slice the time window [start_idx:stop_idx+1]
    X_win = X_time_real[:, start_idx:stop_idx+1, :]   # (N, 35, n_pos)

    # Expect n_pos = 72 (antenna positions). If not, we won't force it but will log.
    if n_pos != 72:
        logger.warning(f"Expected 72 positions; got {n_pos}. Proceeding anyway.")

    # Flatten (N, 35*positions) -> 2520 when positions=72
    X_flat = X_win.reshape(n_samples, -1).astype(np.float64)
    return X_flat

# ----------------------------
# Load helpers
# ----------------------------
def load_pickle_array(path):
    with open(path, 'rb') as f:
        arr = pickle.load(f)
    return arr

# ----------------------------
# Train/Eval for one run
# ----------------------------
def train_eval_one_run(train_complex, y_train, test_complex, y_test, seed, out_prefix):
    # Preprocess (paper): IDFT -> real -> time window -> flatten
    X_train = idft_time_window_features(train_complex)
    X_test  = idft_time_window_features(test_complex)

    # Standardize (fit on train, transform train & test)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std  = scaler.transform(X_test)

    # Random Forest (paper best): 100 trees; vary random_state across runs
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=seed,
        n_jobs=-1
    )
    clf.fit(X_train_std, y_train)

    # Predict
    y_pred = clf.predict(X_test_std)
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test_std)[:, 1]
    else:
        # Fallback if not available (should be available)
        y_prob = (y_pred == 1).astype(float)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    sens = recall_score(y_test, y_pred, zero_division=0)                 # sensitivity (tumor=1)
    spec = recall_score(y_test, y_pred, pos_label=0, zero_division=0)    # specificity (healthy=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) == 2 else float('nan')
    mcc = matthews_corrcoef(y_test, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    run_metrics_path = os.path.join("artifacts", f"{out_prefix}_metrics.txt")
    with open(run_metrics_path, "w") as f:
        f.write(f"Seed: {seed}\n")
        f.write(f"Accuracy: {acc:.6f}\n")
        f.write(f"Precision: {prec:.6f}\n")
        f.write(f"Sensitivity: {sens:.6f}\n")
        f.write(f"Specificity: {spec:.6f}\n")
        f.write(f"F1: {f1:.6f}\n")
        f.write(f"ROC AUC: {roc:.6f}\n")
        f.write(f"MCC: {mcc:.6f}\n")
        f.write(f"Confusion Matrix [labels 0,1]:\n{cm}\n")

    # Save scaler+model (optional but useful)
    with open(os.path.join("artifacts", f"{out_prefix}_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join("artifacts", f"{out_prefix}_rf.pkl"), "wb") as f:
        pickle.dump(clf, f)

    # Return for summary table
    return {
        "seed": seed,
        "accuracy": acc,
        "precision": prec,
        "recall_sensitivity": sens,
        "specificity": spec,
        "f1": f1,
        "roc_auc": roc,
        "mcc": mcc
    }, cm

# ----------------------------
# Main: 5 runs + summary
# ----------------------------
if __name__ == "__main__":
    # --- Paths (keep your existing structure) ---
    train_data_path = r'E:\Breast_Cancer\Ali Arshad\BMS\ML_in_BMS\umbmid\gen-one\clean\train_data.pickle'
    train_md_path   = r'E:\Breast_Cancer\Ali Arshad\BMS\ML_in_BMS\umbmid\gen-one\clean\train_md.pickle'  # labels

    test_data_path  = r'E:\Breast_Cancer\Ali Arshad\BMS\ML_in_BMS\umbmid\gen-one\clean\test_data.pickle'
    test_md_path    = r'E:\Breast_Cancer\Ali Arshad\BMS\ML_in_BMS\umbmid\gen-one\clean\test_md.pickle'   # labels

    # --- Load data ---
    logger.info("Loading pickled frequency-domain data...")
    X_train_complex = load_pickle_array(train_data_path)  # expected shape (N_train, N_freqs, N_pos), complex
    X_test_complex  = load_pickle_array(test_data_path)   # expected shape (N_test,  N_freqs, N_pos), complex

    logger.info(f"Train array shape: {X_train_complex.shape} | dtype: {X_train_complex.dtype}")
    logger.info(f"Test  array shape: {X_test_complex.shape} | dtype: {X_test_complex.dtype}")

    # --- Load labels from metadata ---
    # Expect a list of dicts; tumor if 'tum_rad' present and not NaN (same rule as your code)
    def labels_from_md(md_pickle):
        with open(md_pickle, 'rb') as f:
            md = pickle.load(f)
        y = np.array([1 if isinstance(entry, dict) and ('tum_rad' in entry) and pd.notna(entry['tum_rad']) else 0
                      for entry in md], dtype=np.int64)
        return y

    y_train = labels_from_md(train_md_path)
    y_test  = labels_from_md(test_md_path)

    assert len(y_train) == X_train_complex.shape[0], "Train labels length mismatch"
    assert len(y_test)  == X_test_complex.shape[0],  "Test labels length mismatch"

    # --- 5 runs (different seeds) ---
    seeds = [11, 22, 33, 44, 55]
    rows = []
    cms = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, seed in enumerate(seeds, start=1):
        print("\n==============================")
        print(f"🔁 Run {i}/5 (seed={seed}) starting...")
        print("==============================")
        out_prefix = f"run{i}_seed{seed}_{timestamp}"
        metrics, cm = train_eval_one_run(X_train_complex, y_train, X_test_complex, y_test, seed, out_prefix)
        rows.append(metrics)
        cms.append(cm)

        # Pretty print
        print("\nTest Set Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall/Sensitivity: {metrics['recall_sensitivity']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"MCC: {metrics['mcc']:.4f}")
        print(f"Confusion Matrix [0,1]:\n{cm}")

    # --- Save per-run table & summary ---
    df = pd.DataFrame(rows, index=[f"run{i}" for i in range(1, 6)])
    os.makedirs("artifacts", exist_ok=True)
    per_run_csv = os.path.join("artifacts", "metrics_per_run.csv")
    df.to_csv(per_run_csv, index=True)

    summary = df.agg(['mean', 'std'])
    summary_csv = os.path.join("artifacts", "metrics_summary.csv")
    summary.to_csv(summary_csv)

    # Confusion matrices
    cm_sum = np.sum(np.stack(cms, axis=0), axis=0)
    with open(os.path.join("artifacts", "confusions.txt"), "w") as f:
        for i, cm in enumerate(cms, start=1):
            f.write(f"Run {i} confusion (labels=[0,1]):\n{cm}\n\n")
        f.write("Element-wise sum across runs:\n")
        f.write(str(cm_sum) + "\n")

    print("\n================ SUMMARY (mean ± std over 5 runs) ================")
    for metric in ["accuracy", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc", "mcc"]:
        m = summary.loc['mean', metric]
        s = summary.loc['std', metric]
        print(f"{metric}: {m:.4f} ± {s:.4f}")
    print("==================================================================")
    print("\nArtifacts saved to ./artifacts:")
    print(f"- metrics_per_run.csv")
    print(f"- metrics_summary.csv")
    print(f"- confusions.txt")
    print(f"- runX_seedY_* (model + scaler per run)")
