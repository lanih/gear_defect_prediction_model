import os
import glob
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from scipy.signal import welch
from scipy.stats import kurtosis, skew


# ============================================================
# USER SETTINGS
# ============================================================
DATA_DIR = r"C:\Users\balan\Desktop\ML for materials\project\data"

WINDOW_SIZE = 1024
OVERLAP = 0.75
RANDOM_STATE = 0
FS = 5000

BATCH_SIZE = 64
CLS_EPOCHS = 80
LEARNING_RATE = 1e-4

TRAIN_FRAC = 0.60
VAL_FRAC = 0.15
GAP_FRAC = 0.02

LABEL_SMOOTHING = 0.02
NO_FAULT_WEIGHT_BOOST = 1.25
BINARY_LOSS_WEIGHT = 1.0
MULTI_LOSS_WEIGHT = 1.0
NO_FAULT_MARGIN_WEIGHT = 0.20
AUGMENT_NOISE_STD = 0.005

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# REPRODUCIBILITY

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)



# HELPERS

def create_windows(signal, window_size=1024, overlap=0.5):
    signal = np.asarray(signal)
    step = int(window_size * (1 - overlap))

    if step <= 0:
        raise ValueError("Overlap too large. Step size became <= 0.")

    windows = []
    for start in range(0, len(signal) - window_size + 1, step):
        windows.append(signal[start:start + window_size])

    return np.array(windows)


def load_data_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    df.columns = df.columns.str.strip()

    required_cols = ["sensor1", "sensor2", "speedSet", "load_value"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {file_path}")

    return df


def safe_normalize(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    mu = np.mean(x)
    sigma = np.std(x)
    return (x - mu) / (sigma + eps)


def compute_fft_features(window, fs=5000):
    """
    Rich spectral + impulsiveness features for one 1D signal window.
    Designed to better separate subtle gear faults from no_fault.
    """
    window = np.asarray(window, dtype=np.float32)

    freqs, psd = welch(
        window,
        fs=fs,
        nperseg=min(len(window), 256),
        noverlap=min(len(window), 128),
        scaling="density"
    )

    # Compact spectral shape features
   
    psd = np.maximum(psd, 1e-12)
    total_power = np.sum(psd) + 1e-12

    # Basic spectral descriptors
    peak_idx = np.argmax(psd)
    dominant_freq = freqs[peak_idx]
    dominant_amp = psd[peak_idx]

    spectral_centroid = np.sum(freqs * psd) / total_power
    rms_freq = np.sqrt(np.sum((freqs ** 2) * psd) / total_power)
    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / total_power)
    spectral_flatness = np.exp(np.mean(np.log(psd))) / (np.mean(psd) + 1e-12)
    log_total_power = np.log10(total_power)

    # Narrower band powers
    bands = [
        (0, 50),
        (50, 100),
        (100, 150),
        (150, 200),
        (200, 300),
        (300, 450),
        (450, 600),
        (600, 900),
        (900, 1200),
        (1200, 1800),
        (1800, 2500),
    ]

    band_powers = []
    for f_low, f_high in bands:
        mask = (freqs >= f_low) & (freqs < f_high)
        band_powers.append(np.sum(psd[mask]))

    band_powers = np.array(band_powers, dtype=np.float32)
    rel_band_powers = band_powers / (np.sum(band_powers) + 1e-12)

    # Top-k spectral peaks
    k = 5
    peak_indices = np.argsort(psd)[-k:]
    peak_freqs = freqs[peak_indices]
    peak_amps = psd[peak_indices]

    order = np.argsort(peak_freqs)
    peak_freqs = peak_freqs[order]
    peak_amps = peak_amps[order]

    peak_ratios = peak_amps / total_power

    # Time-domain impulsiveness / shape
    rms = np.sqrt(np.mean(window ** 2))
    peak_abs = np.max(np.abs(window))
    crest_factor = peak_abs / (rms + 1e-8)
    kurt = kurtosis(window, fisher=False, bias=False)
    skw = skew(window, bias=False)
    mean_abs = np.mean(np.abs(window))
    std = np.std(window)

    feats = np.concatenate([
    np.array([
        dominant_freq,
        dominant_amp,
        spectral_centroid,
        rms_freq,
        spectral_spread,
        spectral_flatness,
        log_total_power,
        rms,
        crest_factor,
        kurt,
        skw,
        mean_abs,
        std
    ], dtype=np.float32),
    band_powers.astype(np.float32),
    rel_band_powers.astype(np.float32),
    peak_freqs.astype(np.float32),
    peak_amps.astype(np.float32),
    peak_ratios.astype(np.float32),
    ])

    return feats.astype(np.float32)

def compute_fft_vector(window, fs=5000, n_bins=128):
    window = window * np.hanning(len(window))
    fft = np.fft.rfft(window)
    mag = np.abs(fft)

    freqs = np.fft.rfftfreq(len(window), d=1/fs)

    # limit frequency range
    mask = freqs <= 2500
    mag = mag[mask]

    # downsample to fixed size
    idx = np.linspace(0, len(mag) - 1, n_bins).astype(int)
    mag = mag[idx]

    # normalize
    norm = np.linalg.norm(mag) + 1e-8
    mag_norm = mag / norm

    log_energy = np.log10(norm + 1e-8)

    return np.concatenate([
        mag_norm,
        np.array([log_energy], dtype=np.float32)
    ])

def build_meta_features(s1_window, s2_window, speed_mean, load_mean, fs=5000):    
    """
    Metadata = operating condition + spectral/statistical features from both sensors.
    """
    fft_summary_1 = compute_fft_features(s1_window, fs=fs)
    fft_summary_2 = compute_fft_features(s2_window, fs=fs)

    fft_vec_1 = compute_fft_vector(s1_window, fs=fs)
    fft_vec_2 = compute_fft_vector(s2_window, fs=fs)
    meta = np.concatenate([
    np.array([speed_mean, load_mean], dtype=np.float32),

    # summary features
    fft_summary_1,
    fft_summary_2,

    fft_vec_1,
    fft_vec_2
    ]).astype(np.float32)

    return meta


# ============================================================
# DATASET BUILDING
# ============================================================
def build_time_split_window_dataset(
    data_dir,
    window_size=1024,
    overlap=0.5,
    train_frac=0.6,
    val_frac=0.15,
    gap_frac=0.02
):
    """
    Split each file by time first, then create windows inside each split.
    This reduces leakage for time-series classification.
    """
    def make_windows_from_segment(s1_seg, s2_seg, speed_seg, load_seg, label):
        rows_signal, rows_meta, rows_y = [], [], []

        if len(s1_seg) < window_size or len(s2_seg) < window_size:
            return rows_signal, rows_meta, rows_y

        s1_windows = create_windows(s1_seg, window_size=window_size, overlap=overlap)
        s2_windows = create_windows(s2_seg, window_size=window_size, overlap=overlap)
        speed_windows = create_windows(speed_seg, window_size=window_size, overlap=overlap)
        load_windows = create_windows(load_seg, window_size=window_size, overlap=overlap)

        n_windows = min(len(s1_windows), len(s2_windows), len(speed_windows), len(load_windows))

        for i in range(n_windows):
            s1w = s1_windows[i].astype(np.float32)
            s2w = s2_windows[i].astype(np.float32)

            # Time-domain input to CNN
            s1w_norm = safe_normalize(s1w)
            s2w_norm = safe_normalize(s2w)
            window_2ch = np.stack([s1w_norm, s2w_norm], axis=-1).astype(np.float32)

            # Window-level operating condition metadata
            speed_mean = np.mean(speed_windows[i])
            load_mean = np.mean(load_windows[i])

            # FFT / PSD / impulsiveness features
            meta = build_meta_features(s1w, s2w, speed_mean, load_mean, fs=FS)

            rows_signal.append(window_2ch)
            rows_meta.append(meta)
            rows_y.append(label)

        return rows_signal, rows_meta, rows_y

    X_signal_tr, X_meta_tr, y_text_tr = [], [], []
    X_signal_val, X_meta_val, y_text_val = [], [], []
    X_signal_te, X_meta_te, y_text_te = [], [], []

    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    excel_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    xls_files = glob.glob(os.path.join(data_dir, "*.xls"))
    data_files = csv_files + excel_files + xls_files

    print("Data files found:")
    for f in data_files:
        print("  ", f)

    if len(data_files) == 0:
        raise ValueError(f"No data files found in: {data_dir}")

    for file_path in data_files:
        try:
            label = os.path.splitext(os.path.basename(file_path))[0]
            df = load_data_file(file_path)

            s1 = pd.to_numeric(df["sensor1"], errors="coerce").to_numpy()
            s2 = pd.to_numeric(df["sensor2"], errors="coerce").to_numpy()
            speed = pd.to_numeric(df["speedSet"], errors="coerce").to_numpy()
            load = pd.to_numeric(df["load_value"], errors="coerce").to_numpy()

            valid_mask = (
                np.isfinite(s1) &
                np.isfinite(s2) &
                np.isfinite(speed) &
                np.isfinite(load)
            )

            s1 = s1[valid_mask]
            s2 = s2[valid_mask]
            speed = speed[valid_mask]
            load = load[valid_mask]

            if len(s1) < window_size or len(s2) < window_size:
                print(f"Skipping {file_path}: signal shorter than WINDOW_SIZE")
                continue

            # Mean-center each whole file
            s1 = s1 - np.mean(s1)
            s2 = s2 - np.mean(s2)

            n = len(s1)

            train_end = int(train_frac * n)
            val_start = int((train_frac + gap_frac) * n)
            val_end = int((train_frac + gap_frac + val_frac) * n)
            test_start = int((train_frac + 2 * gap_frac + val_frac) * n)

            if test_start >= n or val_start >= val_end:
                print(f"Skipping {file_path}: not enough length after adding gaps")
                continue

            tr_sig, tr_meta, tr_y = make_windows_from_segment(
                s1[:train_end], s2[:train_end], speed[:train_end], load[:train_end], label
            )
            val_sig, val_meta, val_y = make_windows_from_segment(
                s1[val_start:val_end], s2[val_start:val_end],
                speed[val_start:val_end], load[val_start:val_end], label
            )
            te_sig, te_meta, te_y = make_windows_from_segment(
                s1[test_start:], s2[test_start:],
                speed[test_start:], load[test_start:], label
            )

            X_signal_tr.extend(tr_sig)
            X_meta_tr.extend(tr_meta)
            y_text_tr.extend(tr_y)

            X_signal_val.extend(val_sig)
            X_meta_val.extend(val_meta)
            y_text_val.extend(val_y)

            X_signal_te.extend(te_sig)
            X_meta_te.extend(te_meta)
            y_text_te.extend(te_y)

            print(f"\n{os.path.basename(file_path)}")
            print(f"  raw length    : {n}")
            print(f"  train windows : {len(tr_y)}")
            print(f"  val windows   : {len(val_y)}")
            print(f"  test windows  : {len(te_y)}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return (
        np.array(X_signal_tr, dtype=np.float32),
        np.array(X_meta_tr, dtype=np.float32),
        np.array(y_text_tr),
        np.array(X_signal_val, dtype=np.float32),
        np.array(X_meta_val, dtype=np.float32),
        np.array(y_text_val),
        np.array(X_signal_te, dtype=np.float32),
        np.array(X_meta_te, dtype=np.float32),
        np.array(y_text_te),
    )


# ============================================================
# PYTORCH DATASET
# ============================================================
class VibrationDataset(Dataset):
    def __init__(self, X_signal, X_meta, y=None, augment=False):
        """
        X_signal: (N, L, C) -> convert to (N, C, L) for Conv1d
        X_meta:   (N, M)
        y:        (N,)
        """
        self.X_signal = torch.tensor(np.transpose(X_signal, (0, 2, 1)), dtype=torch.float32)
        self.X_meta = torch.tensor(X_meta, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.X_signal)

    def __getitem__(self, idx):
        x_signal = self.X_signal[idx]
        x_meta = self.X_meta[idx]

        if self.augment:
            noise = AUGMENT_NOISE_STD * torch.randn_like(x_signal)
            x_signal = x_signal + noise

        if self.y is None:
            return x_signal, x_meta
        return x_signal, x_meta, self.y[idx]


# ============================================================
# MODEL
# ============================================================
class DirectCNNClassifier(nn.Module):
    def __init__(self, n_classes, meta_dim=2):
        super().__init__()

        self.signal_net = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.meta_net = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, n_classes)
        )

    def forward(self, x_signal, x_meta):
        x = self.signal_net(x_signal)
        x = x.squeeze(-1)

        m = self.meta_net(x_meta)

        features = torch.cat([x, m], dim=1)
        logits = self.classifier(features)
        return logits


# ============================================================
# TRAINING / EVALUATION
# ============================================================
def train_classifier(
    model,
    train_loader,
    val_loader,
    y_train,
    label_encoder,
    epochs=50,
    lr=1e-3,
    device="cpu"
):
    n_classes = model.classifier[-1].out_features

    class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
    print("Train class counts:", class_counts)

    class_weights = np.zeros(n_classes, dtype=np.float32)
    present_mask = class_counts > 0
    class_weights[present_mask] = np.sum(class_counts) / (n_classes * class_counts[present_mask])

    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_t,
        label_smoothing=0.0
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    patience = 12
    counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x_signal, x_meta, y in train_loader:
            x_signal = x_signal.to(device)
            x_meta = x_meta.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x_signal, x_meta)
            loss = criterion(logits, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x_signal, x_meta, y in val_loader:
                x_signal = x_signal.to(device)
                x_meta = x_meta.to(device)
                y = y.to(device)

                logits = model(x_signal, x_meta)
                loss = criterion(logits, y)

                running_val_loss += loss.item() * y.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_loss = running_val_loss / val_total
        val_acc = val_correct / val_total

        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Classifier early stopping triggered")
                break

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[CLS] Epoch {epoch+1:03d}/{epochs} | "
            f"lr={current_lr:.2e} | "
            f"train_loss={train_loss:.6f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.6f} | val_acc={val_acc:.4f}"
        )

    model.load_state_dict(best_state)
    return train_losses, val_losses, train_accs, val_accs

def evaluate_classifier(model, test_loader, label_encoder, device="cpu"):
    model.eval()

    all_y = []
    all_pred = []

    with torch.no_grad():
        for x_signal, x_meta, y in test_loader:
            x_signal = x_signal.to(device)
            x_meta = x_meta.to(device)

            logits = model(x_signal, x_meta)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_pred.extend(preds)
            all_y.extend(y.numpy())

    all_y = np.array(all_y)
    all_pred = np.array(all_pred)

    labels = np.arange(len(label_encoder.classes_))

    print("\nClassification Report:")
    print(classification_report(
        all_y,
        all_pred,
        labels=labels,
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    cm = confusion_matrix(all_y, all_pred, labels=labels)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_
    )
    disp.plot(xticks_rotation=45)
    plt.title("CNN Classifier Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return all_y, all_pred


def plot_loss_curves(train_losses, val_losses, title):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_accuracy_curves(train_accs, val_accs, title):
    plt.figure(figsize=(8, 4))
    plt.plot(train_accs, label="train")
    plt.plot(val_accs, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# PSD PLOTS
# ============================================================
def plot_psds_by_class(data_dir, fs=5000, sensor_col="sensor1", max_files_per_class=1):
    """
    Plot average Welch PSD for each class/file.
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    excel_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    xls_files = glob.glob(os.path.join(data_dir, "*.xls"))
    data_files = csv_files + excel_files + xls_files

    class_count = {}
    plt.figure(figsize=(10, 6))

    for file_path in sorted(data_files):
        label = os.path.splitext(os.path.basename(file_path))[0]
        class_count.setdefault(label, 0)

        if class_count[label] >= max_files_per_class:
            continue

        df = load_data_file(file_path)
        signal = pd.to_numeric(df[sensor_col], errors="coerce").to_numpy()
        signal = signal[np.isfinite(signal)]

        if len(signal) < 256:
            continue

        signal = signal - np.mean(signal)

        freqs, psd = welch(
            signal,
            fs=fs,
            nperseg=min(len(signal), 1024),
            noverlap=min(len(signal) // 2, 512),
            scaling="density"
        )

        plt.semilogy(freqs, psd, label=label)
        class_count[label] += 1

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.title(f"Welch PSD by Class ({sensor_col})")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_psd_comparison_for_classes(data_dir, target_classes, fs=5000, sensor_col="sensor1"):
    """
    Focused PSD comparison for selected classes.
    Example target_classes=["no_fault", "missing_tooth", "root_crack"]
    """
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    excel_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    xls_files = glob.glob(os.path.join(data_dir, "*.xls"))
    data_files = csv_files + excel_files + xls_files

    plt.figure(figsize=(10, 6))

    for file_path in sorted(data_files):
        label = os.path.splitext(os.path.basename(file_path))[0]
        if label not in target_classes:
            continue

        df = load_data_file(file_path)
        signal = pd.to_numeric(df[sensor_col], errors="coerce").to_numpy()
        signal = signal[np.isfinite(signal)]

        if len(signal) < 256:
            continue

        signal = signal - np.mean(signal)

        freqs, psd = welch(
            signal,
            fs=fs,
            nperseg=min(len(signal), 1024),
            noverlap=min(len(signal) // 2, 512),
            scaling="density"
        )

        plt.semilogy(freqs, psd, label=label)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.title(f"Focused PSD Comparison ({sensor_col})")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Using device:", DEVICE)

    # PSD diagnostics before training
    plot_psds_by_class(DATA_DIR, fs=FS, sensor_col="sensor1", max_files_per_class=1)
    plot_psd_comparison_for_classes(
        DATA_DIR,
        target_classes=["no_fault", "missing_tooth", "root_crack", "tooth_chipped_fault"],
        fs=FS,
        sensor_col="sensor1"
    )

    (
        X_signal_tr, X_meta_tr, y_text_tr,
        X_signal_val, X_meta_val, y_text_val,
        X_signal_test, X_meta_test, y_text_test
    ) = build_time_split_window_dataset(
        DATA_DIR,
        window_size=WINDOW_SIZE,
        overlap=OVERLAP,
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        gap_frac=GAP_FRAC
    )

    label_encoder = LabelEncoder()
    all_labels = np.concatenate([y_text_tr, y_text_val, y_text_test])
    label_encoder.fit(all_labels)

    y_tr = label_encoder.transform(y_text_tr)
    y_val = label_encoder.transform(y_text_val)
    y_test = label_encoder.transform(y_text_test)

    print("\nClass mapping:")
    for i, c in enumerate(label_encoder.classes_):
        print(f"{i}: {c}")

    print("\nTrain distribution:", np.bincount(y_tr, minlength=len(label_encoder.classes_)))
    print("Val distribution  :", np.bincount(y_val, minlength=len(label_encoder.classes_)))
    print("Test distribution :", np.bincount(y_test, minlength=len(label_encoder.classes_)))

    meta_scaler = StandardScaler()
    X_meta_tr = meta_scaler.fit_transform(X_meta_tr)
    X_meta_val = meta_scaler.transform(X_meta_val)
    X_meta_test = meta_scaler.transform(X_meta_test)

    print("\nSplit sizes:")
    print("Train:", X_signal_tr.shape, X_meta_tr.shape, y_tr.shape)
    print("Val  :", X_signal_val.shape, X_meta_val.shape, y_val.shape)
    print("Test :", X_signal_test.shape, X_meta_test.shape, y_test.shape)

    train_ds = VibrationDataset(X_signal_tr, X_meta_tr, y_tr, augment=False)
    val_ds = VibrationDataset(X_signal_val, X_meta_val, y_val, augment=False)
    test_ds = VibrationDataset(X_signal_test, X_meta_test, y_test, augment=False)

    class_counts = np.bincount(y_tr)
    sample_weights = 1.0 / np.sqrt(class_counts[y_tr])
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    classifier = DirectCNNClassifier(
        n_classes=len(label_encoder.classes_),
        meta_dim=X_meta_tr.shape[1]
    ).to(DEVICE)

    cls_train_losses, cls_val_losses, cls_train_accs, cls_val_accs = train_classifier(
        classifier,
        train_loader,
        val_loader,
        y_tr,
        label_encoder=label_encoder,
        epochs=CLS_EPOCHS,
        lr=LEARNING_RATE,
        device=DEVICE
    )

    plot_loss_curves(cls_train_losses, cls_val_losses, "Classifier Loss")
    plot_accuracy_curves(cls_train_accs, cls_val_accs, "Classifier Accuracy")

    evaluate_classifier(classifier, test_loader, label_encoder, device=DEVICE)