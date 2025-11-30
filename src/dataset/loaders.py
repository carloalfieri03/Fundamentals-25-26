import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import numpy as np

class HarDataset(Dataset):
    SIGNALS = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z",
    ]

    def __init__(self, data_dir: Path, split: str = 'train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        self.X = self.load_signals()
        self.y = self.load_labels()

        if self.transform:
            self.X = self.transform(self.X)
    
    # [batch, channels, seq_len], in pratica con quello stack unisco le 9 matrici in un tensore 3D
    def load_signals(self):
        signals_dir = self.data_dir / f'UCI HAR Dataset/{self.split}/Inertial Signals'
        X_signals = []
        for sig in self.SIGNALS:
            file_path = signals_dir / f"{sig}_{self.split}.txt"
            data = np.loadtxt(file_path)  # (N, 128)
            X_signals.append(data)
        X = np.stack(X_signals, axis=1)  # (N, 9, 128)
        return torch.tensor(X, dtype=torch.float32)

    def load_labels(self):
        labels_file = self.data_dir / f'UCI HAR Dataset/{self.split}/y_{self.split}.txt'
        y = np.loadtxt(labels_file).astype(int).squeeze() - 1 # Convert to 0-based indexing
        return torch.tensor(y, dtype=torch.long)
        self.split = split

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class WisdmDataset(Dataset):
    def __init__(self, data_dir: Path, window_size=128, 
    overlap=0.5, transform=None):

        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.step = int(window_size * (1 - overlap))
        self.transform = transform

        raw_file = self.data_dir / f"WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"
        self.X, self.y = self.load_and_window(raw_file)

        if self.transform:
            self.X = self.transform(self.X)

    def load_and_window(self, file_path: Path):
        raw_X = []
        raw_y = []

        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")

                if len(parts) < 6:
                    continue  

                user = parts[0]
                activity = parts[1].strip()
                x = float(parts[3])
                y = float(parts[4])
                z_str = parts[5].split(";")[0].strip() 
                z = float(z_str) if z_str else 0.0 
                raw_X.append([x, y, z])
                raw_y.append(activity)

        raw_X = np.array(raw_X)               # (N, 3)
        raw_y = np.array(raw_y)               # (N,)

        unique_labels = sorted(list(set(raw_y)))
        label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}
        y_int = np.array([label_to_id[l] for l in raw_y])

        windows_X = []
        windows_y = []

        N = len(raw_X)
        for start in range(0, N - self.window_size, self.step):
            end = start + self.window_size
            win_X = raw_X[start:end]                  # (128, 3)
            win_y = y_int[start:end]
            label = np.bincount(win_y).argmax()
            win_X = win_X.T                           # (3, 128)
            windows_X.append(win_X)
            windows_y.append(label)

        return (
            torch.tensor(np.stack(windows_X), dtype=torch.float32),
            torch.tensor(np.array(windows_y), dtype=torch.long)
        )
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_har(data_dir: Path, batch_size=32, val_split=0.2, num_workers=4):
    full_train_dataset = HarDataset(data_dir, split='train', transform=normalize_signals)

    val_size = int(val_split * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    test_dataset = HarDataset(data_dir, split='test', transform=normalize_signals)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def load_wisdm(data_dir: Path, batch_size=32,
               train_split=0.7, val_split=0.15, num_workers=4,
               window_size=128, overlap=0.5):

    full_dataset = WisdmDataset(
        data_dir=data_dir,
        window_size=window_size,
        overlap=overlap,
        transform=normalize_signals
    )

    total = len(full_dataset)
    train_size = int(train_split * total)
    val_size = int(val_split * total)
    test_size = total - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    train_loader =   DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader




# ----------------------------------------
# Esempio di trasformazione
# ----------------------------------------
def normalize_signals(X):
    # X: (N, 128, 9)...
    mean = X.mean(dim=(0,1), keepdim=True)
    std  = X.std(dim=(0,1), keepdim=True)
    return (X - mean) / (std + 1e-8)


"""
train_transforms = Compose([
    normalize,  # Prima normalizza
    noise,      # Poi aggiunge rumore
    scale       # Poi scala
])
"""
