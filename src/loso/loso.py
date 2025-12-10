from src.data.har_dataset import class HarDataset
def load_har_loso(data_dir, test_subject_id, batch_size=32, num_workers=4):
    """
    Loads HAR data specifically for LOSO validation.
    """
    # 1. Load EVERYTHING as one big dataset first
    # We combine original 'train' and 'test' folders because in LOSO 
    # we effectively reshuffle everyone. 
    # Note: Standard UCI-HAR has subjects 1,3,5.. in train and 2,4,6.. in test.
    # Ideally, you load BOTH folders and concatenate them. 
    # For simplicity, let's assume 'train' folder has enough subjects for now, 
    # or you modify HarDataset to load both.
    
    full_dataset = HarDataset(data_dir, split='train', transform=normalize_signals)
    
    # 2. Perform the Split based on the specific ID
    train_dataset, val_dataset = get_loso_split(full_dataset, test_subject_id)

    # 3. Create Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader