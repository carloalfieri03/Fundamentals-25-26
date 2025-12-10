import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# ---------------------------------------------------------
# 1. Global Normalization Functions (The logic we discussed)
# ---------------------------------------------------------

def get_dataset_stats(X_train):
    """
    Calculates mean and std across the entire training set.
    Assumes X_train shape: (Batch, Channels, Time) -> (N, 9, 128)
    """
    # Average over Batch (0) and Time (2), keeping Channels (1) distinct
    mean = X_train.mean(dim=(0, 2), keepdim=True) 
    std  = X_train.std(dim=(0, 2), keepdim=True)
    return mean, std

def apply_normalization(X, mean, std):
    """
    Applies the calculated global stats to a tensor.
    """
    return (X - mean) / (std + 1e-8)

def split_by_subject(dataset, val_split=0.2, seed=42):
    """
    Splits a dataset into Train and Val based on Subject ID to prevent data leakage.
    """
    splits = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    
    # We pass 'dataset.subjects' as the groups
    train_idx, val_idx = next(splits.split(dataset.X, dataset.y, groups=dataset.subjects))

    # Create Subsets
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    print(f"--- Subject Split Created ---")
    print(f"Train Samples: {len(train_idx)}")
    print(f"Val Samples:   {len(val_idx)}")
    
    return train_subset, val_subset

# ---------------------------------------------------------
# 2. HAR Dataset (Cleaned)
# ---------------------------------------------------------

class HarDataset(Dataset):
    SIGNALS = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z",
    ]

    def __init__(self, data_dir: Path, split: str = 'train'):
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Load Raw Data
        self.X = self.load_signals()
        self.y = self.load_labels()
        self.subjects = self.load_subjects()
    
    def load_signals(self):
        signals_dir = self.data_dir / f'UCI HAR Dataset/{self.split}/Inertial Signals'
        X_signals = []
        for sig in self.SIGNALS:
            file_path = signals_dir / f"{sig}_{self.split}.txt"
            data = np.loadtxt(file_path)  # (N, 128)
            X_signals.append(data)
        X = np.stack(X_signals, axis=1)  # (N, 9, 128)
        return torch.tensor(X, dtype=torch.float32)
    
    def load_subjects(self):
        subject_file = self.data_dir / f'UCI HAR Dataset/{self.split}/subject_{self.split}.txt'
        subjects = np.loadtxt(subject_file).astype(int).squeeze()
        return subjects

    def load_labels(self):
        labels_file = self.data_dir / f'UCI HAR Dataset/{self.split}/y_{self.split}.txt'
        y = np.loadtxt(labels_file).astype(int).squeeze() - 1 
        return torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_har(data_dir: Path, batch_size=32, val_split=0.2, num_workers=4):
    print("Loading UCI HAR...")
    
    # 1. Load Raw Train and Test Data
    # Note: We do NOT normalize yet.
    full_train_dataset = HarDataset(data_dir, split='train')
    test_dataset = HarDataset(data_dir, split='test')

    # 2. Split Train into Train/Val by Subject
    # train_subset and val_subset are just list of INDICES pointing to full_train_dataset
    train_subset, val_subset = split_by_subject(full_train_dataset, val_split=val_split)

    # 3. Calculate Global Stats on TRAIN SUBSET ONLY
    print("Calculating Global Normalization Stats...")
    # We access the underlying data using the subset indices
    train_X_data = full_train_dataset.X[train_subset.indices]
    
    mean, std = get_dataset_stats(train_X_data)
    print(f"Global Mean Shape: {mean.shape}") # Should be [1, 9, 1]

    # 4. Apply Normalization to EVERYTHING
    # Modifying the tensors in-place is efficient
    full_train_dataset.X = apply_normalization(full_train_dataset.X, mean, std)
    test_dataset.X       = apply_normalization(test_dataset.X, mean, std)

    # 5. Create Loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,persistent_workers=True)
    val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,persistent_workers=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,persistent_workers=True)

    return train_loader, val_loader, test_loader

# ---------------------------------------------------------
# 3. WISDM Dataset (Cleaned)
# ---------------------------------------------------------

class WisdmDataset(Dataset):
    def __init__(self, data_dir: Path, window_size=128, overlap=0.5):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.step = int(window_size * (1 - overlap))

        raw_file = self.data_dir / f"WISDM_ar_latest/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt"
        
        # Load raw
        self.X, self.y = self.load_and_window(raw_file)

    def load_and_window(self, file_path: Path):
        raw_X = []
        raw_y = []
        # ... (Your existing file parsing logic is fine) ...
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6: continue  
                user = parts[0]
                activity = parts[1].strip()
                x = float(parts[3])
                y = float(parts[4])
                z_str = parts[5].split(";")[0].strip() 
                z = float(z_str) if z_str else 0.0 
                raw_X.append([x, y, z])
                raw_y.append(activity)

        raw_X = np.array(raw_X)
        raw_y = np.array(raw_y)

        unique_labels = sorted(list(set(raw_y)))
        label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}
        y_int = np.array([label_to_id[l] for l in raw_y])

        windows_X = []
        windows_y = []

        N = len(raw_X)
        for start in range(0, N - self.window_size, self.step):
            end = start + self.window_size
            win_X = raw_X[start:end]         # (128, 3)
            win_y = y_int[start:end]
            label = np.bincount(win_y).argmax()
            win_X = win_X.T                  # (3, 128) - Channels First
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


def load_wisdm(data_dir: Path, batch_size=32, train_split=0.7, val_split=0.15, num_workers=4):
    print("Loading WISDM...")
    
    # 1. Load Full Data
    full_dataset = WisdmDataset(data_dir=data_dir)
    total = len(full_dataset)
    
    # 2. Calculate Split Sizes
    train_size = int(train_split * total)
    val_size = int(val_split * total)
    test_size = total - train_size - val_size

    # 3. Create Random Splits (Indices)
    # Note: ideally for WISDM you should also split by subject ID using GroupShuffleSplit
    # but for now we keep your random_split logic.
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # 4. Calculate Global Stats on TRAIN SUBSET ONLY
    print("Calculating Global Normalization Stats for WISDM...")
    train_indices = train_dataset.indices
    train_X_data = full_dataset.X[train_indices]
    
    mean, std = get_dataset_stats(train_X_data)
    
    # 5. Apply to the entire underlying dataset
    # Since subsets reference the parent 'full_dataset', modifying full_dataset.X
    # automatically fixes data for train, val, and test loaders.
    full_dataset.X = apply_normalization(full_dataset.X, mean, std)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers,persistent_workers=True)
    val_loader   = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers,persistent_workers=True)
    test_loader  = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers,persistent_workers=True)

    return train_loader, val_loader, test_loader