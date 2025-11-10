import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import pickle
from torchvision.models import vgg16, VGG16_Weights
from sklearn.metrics import (accuracy_score, recall_score, roc_auc_score, precision_score,
                             f1_score, confusion_matrix, matthews_corrcoef)
import logging
import matplotlib.pyplot as plt

# ===========================
# Logging & Device
# ===========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ablation")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# ===========================
# Data processing (time/freq)
# ===========================
def process_signals(data):
    sample_rate = 7e9
    start_time = 1.00e-9
    stop_time  = 5.85e-9

    start_index = int(start_time * sample_rate)
    stop_index  = int(stop_time * sample_rate)

    time_domain_signals = []
    freq_domain_signals = []

    for sample in range(data.shape[0]):
        spectrogram_complex = data[sample, :, :]

        # Time-domain
        spectrogram_time_domain = np.fft.ifft(spectrogram_complex, axis=0)
        spectrogram_magnitude_time = np.abs(spectrogram_time_domain)
        cropped_time = spectrogram_magnitude_time[start_index:stop_index, :]
        max_time = np.max(cropped_time) if np.max(cropped_time) != 0 else 1.0
        cropped_time = cropped_time / max_time

        # Frequency-domain
        spectrogram_magnitude_freq = np.abs(spectrogram_complex)
        cropped_freq = spectrogram_magnitude_freq[start_index:stop_index, :]
        max_freq = np.max(cropped_freq) if np.max(cropped_freq) != 0 else 1.0
        cropped_freq = cropped_freq / max_freq

        time_domain_signals.append(cropped_time)
        freq_domain_signals.append(cropped_freq)

    return np.array(time_domain_signals), np.array(freq_domain_signals)

# ===========================
# Model (modular branches)
# ===========================
class ModularFusionModel(nn.Module):
    def __init__(self, time_shape, freq_shape, use_time, use_freq, use_sino, num_classes=2):
        super().__init__()
        self.use_time = use_time
        self.use_freq = use_freq
        self.use_sino = use_sino

        # Time branch
        if self.use_time:
            self.time_conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.time_flat = self._get_conv_output(time_shape, self.time_conv)
        else:
            self.time_conv = None
            self.time_flat = 0

        # Freq branch
        if self.use_freq:
            self.freq_conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.freq_flat = self._get_conv_output(freq_shape, self.freq_conv)
        else:
            self.freq_conv = None
            self.freq_flat = 0

        # Sinogram (image/VGG) branch → conv reduce + pool to 256×7×7 = 12544
        if self.use_sino:
            self.sino_conv = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.sino_flat = 256 * 7 * 7
        else:
            self.sino_conv = None
            self.sino_flat = 0

        in_dim = self.time_flat + self.freq_flat + self.sino_flat
        assert in_dim > 0, "At least one branch must be enabled."
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    @staticmethod
    def _get_conv_output(shape, conv):
        dummy = torch.rand(1, 1, *shape)  # CPU probe
        with torch.no_grad():
            out = conv(dummy)
        return out.view(1, -1).size(1)

    def forward(self, time_in, freq_in, sino_in):
        feats = []
        if self.use_time:
            t = self.time_conv(time_in)
            feats.append(t.view(t.size(0), -1))
        if self.use_freq:
            f = self.freq_conv(freq_in)
            feats.append(f.view(f.size(0), -1))
        if self.use_sino:
            s = self.sino_conv(sino_in)
            feats.append(s.view(s.size(0), -1))
        x = torch.cat(feats, dim=1)
        return self.classifier(x)

# ===========================
# Dataset (optionally compute sinogram features)
# ===========================
class FusionDataset(Dataset):
    def __init__(self, time_signals, freq_signals, folder_path, metadata_path, transform=None,
                 use_sino=False, vgg_feature_extractor=None):
        self.time_signals = time_signals
        self.freq_signals = freq_signals
        self.folder_path = folder_path
        self.use_sino = use_sino
        self.vgg_feature_extractor = vgg_feature_extractor
        self.transform = transform

        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        self.labels = self._extract_labels()
        self.image_names = self._get_image_names() if self.use_sino else [None]*len(self.time_signals)

        logger.info(f"Dataset init: N={len(self.time_signals)} | use_sino={self.use_sino}")

        if not (len(self.time_signals) == len(self.freq_signals) == len(self.image_names) == len(self.metadata)):
            raise ValueError("Mismatch between signals/images/metadata lengths")

    def _extract_labels(self):
        return [1 if isinstance(entry, dict) and 'tum_rad' in entry and pd.notna(entry['tum_rad']) else 0
                for entry in self.metadata]

    def _get_image_names(self):
        image_names = []
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_names.append(os.path.join(root, file))
        image_names = sorted(image_names)
        if len(image_names) != len(self.time_signals):
            logger.warning(f"Image count ({len(image_names)}) != signal count ({len(self.time_signals)}). "
                           f"Make sure ordering matches.")
        return image_names

    def __len__(self):
        return len(self.time_signals)

    def __getitem__(self, idx):
        time_signal = torch.tensor(self.time_signals[idx], dtype=torch.float32).unsqueeze(0)  # [1,H,W]
        freq_signal = torch.tensor(self.freq_signals[idx], dtype=torch.float32).unsqueeze(0)  # [1,H,W]
        label = torch.tensor( self.labels[idx], dtype=torch.long )

        if self.use_sino:
            img_path = self.image_names[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            # Extract VGG conv features once per sample on-device
            with torch.no_grad():
                vgg_in = image.unsqueeze(0).to(device)
                sino_feat = self.vgg_feature_extractor(vgg_in).squeeze(0).cpu()  # [512,14,14]
        else:
            sino_feat = torch.empty(0)  # placeholder

        return time_signal, freq_signal, sino_feat, label

# ===========================
# Train / Eval
# ===========================
def train_model(model, train_loader, criterion, optimizer, num_epochs=50, run_id=1, tag=""):
    losses, accs = [], []
    for ep in range(num_epochs):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for time_in, freq_in, sino_in, labels in train_loader:
            labels = labels.to(device)

            # Move only enabled streams
            time_in = time_in.to(device) if model.use_time else torch.empty(0)
            freq_in = freq_in.to(device) if model.use_freq else torch.empty(0)
            if model.use_sino:
                # sino_in is [C,H,W] on CPU; add batch dim if needed
                if sino_in.ndim == 3:
                    sino_in = sino_in.unsqueeze(0)
                sino_in = sino_in.to(device)
            else:
                sino_in = torch.empty(0)

            optimizer.zero_grad()
            outputs = model(time_in, freq_in, sino_in)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * labels.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

        ep_loss = run_loss / total
        ep_acc  = correct / total
        losses.append(ep_loss); accs.append(ep_acc)
        logger.info(f"[{tag} | Run {run_id}] Epoch {ep+1}/{num_epochs} - "
                    f"Loss {ep_loss:.4f} | Acc {ep_acc:.4f}")

    # Save curves
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1); plt.plot(losses); plt.title(f'Train Loss ({tag})'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.subplot(1,2,2); plt.plot(accs);  plt.title(f'Train Acc ({tag})');  plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.tight_layout(); plt.savefig(f'training_curves_{tag}_run{run_id}.png', dpi=200); plt.close()
    return losses, accs

@torch.no_grad()
def evaluate_model(model, test_loader):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for time_in, freq_in, sino_in, labels in test_loader:
        labels = labels.to(device)

        time_in = time_in.to(device) if model.use_time else torch.empty(0)
        freq_in = freq_in.to(device) if model.use_freq else torch.empty(0)
        if model.use_sino:
            if sino_in.ndim == 3:
                sino_in = sino_in.unsqueeze(0)
            sino_in = sino_in.to(device)
        else:
            sino_in = torch.empty(0)

        out = model(time_in, freq_in, sino_in)
        _, pred = out.max(1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
        # Assume class 1 is positive
        y_prob.extend(out.softmax(dim=1)[:,1].cpu().numpy())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)

# ===========================
# Main: Ablations (3 modes × 5 runs)
# ===========================
if __name__ == "__main__":
    # ---- Paths ----
    train_data_path     = r'E:\Breast_Cancer\Ali Arshad\BMS\ML_in_BMS\umbmid\gen-one\clean\train_data.pickle'
    train_folder_path   = r'E:\Breast_Cancer\Ali Arshad\BMS\ML_in_BMS\umbmid\gen-one\clean\figs_gen1_clean_train_data_final'
    train_metadata_path = r'E:\Breast_Cancer\Ali Arshad\BMS\ML_in_BMS\umbmid\gen-one\clean\train_md.pickle'

    test_data_path      = r'E:\Breast_Cancer\Ali Arshad\BMS\ML_in_BMS\umbmid\gen-one\clean\test_data.pickle'
    test_folder_path    = r'E:\Breast_Cancer\Ali Arshad\BMS\ML_in_BMS\umbmid\gen-one\clean\figs_gen1_clean_test_data_final'
    test_metadata_path  = r'E:\Breast_Cancer\Ali Arshad\BMS\ML_in_BMS\umbmid\gen-one\clean\test_md.pickle'

    # ---- Load + preprocess once ----
    with open(train_data_path, 'rb') as f: train_data = pickle.load(f)
    with open(test_data_path,  'rb') as f: test_data  = pickle.load(f)
    train_time, train_freq = process_signals(train_data)
    test_time,  test_freq  = process_signals(test_data)

    # ---- VGG16 conv feature extractor (shared) ----
    vgg16_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
    vgg16_model.eval()
    vgg_feature_extractor = nn.Sequential(*list(vgg16_model.features.children())[:-1]).to(device)
    for p in vgg_feature_extractor.parameters(): p.requires_grad = False

    # ---- Image transform ----
    img_tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

    # ---- Ablations: (use_time, use_freq, use_sino) ----
    ABLATIONS = [
        ("A1_freq_only",                 dict(use_time=False, use_freq=True,  use_sino=False)),
        ("A2_freq_plus_time",            dict(use_time=True,  use_freq=True,  use_sino=False)),
        ("A3_freq_time_plus_sinogram",   dict(use_time=True,  use_freq=True,  use_sino=True)),
    ]

    N_RUNS = 5
    combined_summary_rows = []

    for tag, flags in ABLATIONS:
        logger.info(f"\n========== Ablation {tag} : {flags} ==========")
        metrics_rows, conf_sum, all_cms = [], np.zeros((2,2), dtype=np.int64), []

        for run in range(1, N_RUNS+1):
            print("\n==============================")
            print(f"🔁 {tag} | Run {run}/{N_RUNS} starting...")
            print("==============================")

            train_ds = FusionDataset(
                train_time, train_freq, train_folder_path, train_metadata_path,
                transform=img_tf, use_sino=flags['use_sino'],
                vgg_feature_extractor=vgg_feature_extractor if flags['use_sino'] else None
            )
            test_ds = FusionDataset(
                test_time, test_freq, test_folder_path, test_metadata_path,
                transform=img_tf, use_sino=flags['use_sino'],
                vgg_feature_extractor=vgg_feature_extractor if flags['use_sino'] else None
            )
            train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
            test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False)

            # Shapes for model (time/freq always computed; selected by flags)
            time_shape = train_time.shape[1:]  # (H,W)
            freq_shape = train_freq.shape[1:]  # (H,W)

            model = ModularFusionModel(
                time_shape, freq_shape,
                use_time=flags['use_time'],
                use_freq=flags['use_freq'],
                use_sino=flags['use_sino'],
                num_classes=2
            ).to(device)

            crit = nn.CrossEntropyLoss()
            opt  = optim.Adam(model.parameters(), lr=0.001)

            # Train
            _ = train_model(model, train_loader, crit, opt, num_epochs=50, run_id=run, tag=tag)

            # Eval
            y_true, y_pred, y_prob = evaluate_model(model, test_loader)

            # Metrics
            acc  = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec  = recall_score(y_true, y_pred, zero_division=0)
            spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
            f1   = f1_score(y_true, y_pred, zero_division=0)
            roc  = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float('nan')
            mcc  = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
            cm   = confusion_matrix(y_true, y_pred, labels=[0,1])

            print("\nTest Set Metrics:")
            print(f"{tag} | Run {run}  Acc {acc:.4f}  Prec {prec:.4f}  Rec {rec:.4f}  Spec {spec:.4f}  "
                  f"F1 {f1:.4f}  ROC-AUC {roc:.4f}  MCC {mcc:.4f}")
            print(f"Confusion Matrix:\n{cm}")

            # Save per-run
            torch.save(model.state_dict(), f'{tag}_final_model_run{run}.pt')
            with open(f'{tag}_metrics_run{run}.txt', 'w') as f:
                f.write(f"{tag} - Run {run}\n")
                f.write(f"Accuracy: {acc:.6f}\nPrecision: {prec:.6f}\nRecall: {rec:.6f}\n")
                f.write(f"Specificity: {spec:.6f}\nF1: {f1:.6f}\nROC AUC: {roc:.6f}\nMCC: {mcc:.6f}\n")
                f.write(f"Confusion Matrix:\n{cm}\n")

            metrics_rows.append({
                "ablation": tag, "run": run,
                "accuracy": acc, "precision": prec, "recall_sensitivity": rec,
                "specificity": spec, "f1": f1, "roc_auc": roc, "mcc": mcc
            })
            conf_sum += cm
            all_cms.append(cm)

            # Free memory
            del model, train_loader, test_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- Summary for this ablation ---
        df = pd.DataFrame(metrics_rows).set_index("run").sort_index()
        df.to_csv(f'{tag}_metrics_per_run.csv', index=True)
        summary = df.drop(columns=['ablation']).agg(['mean', 'std'])
        summary.to_csv(f'{tag}_metrics_summary.csv')

        print(f"\n===== {tag} SUMMARY (mean ± std over {N_RUNS} runs) =====")
        for metric in ["accuracy","precision","recall_sensitivity","specificity","f1","roc_auc","mcc"]:
            m, s = summary.loc['mean', metric], summary.loc['std', metric]
            print(f"{metric}: {m:.4f} ± {s:.4f}")

        # Append to combined summary collector
        sm_row = {"ablation": tag}
        for metric in ["accuracy","precision","recall_sensitivity","specificity","f1","roc_auc","mcc"]:
            sm_row[f"{metric}_mean"] = summary.loc['mean', metric]
            sm_row[f"{metric}_std"]  = summary.loc['std',  metric]
        combined_summary_rows.append(sm_row)

        # Confusion matrices overview
        print("\nConfusion Matrices per run (labels=[0,1]):")
        for i, cm in enumerate(all_cms, 1):
            print(f"{tag} | Run {i}:\n{cm}\n")
        print(f"Element-wise sum across runs for {tag}:\n{conf_sum}")

    # ---- Combined ablation summary table ----
    comb_df = pd.DataFrame(combined_summary_rows)
    comb_df.to_csv('ablation_summary_overview.csv', index=False)
    print("\nSaved overall ablation summary → 'ablation_summary_overview.csv'")

    print("\nArtifacts saved per ablation:")
    print("- {tag}_metrics_per_run.csv")
    print("- {tag}_metrics_summary.csv")
    print("- training_curves_{tag}_runX.png")
    print("- {tag}_final_model_runX.pt")
