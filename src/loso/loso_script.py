##Pre-requisite: You must have a trained Autoencoder checkpoint saved as checkpoints/best_encoder.ckpt.
#  (Run your standard experiment.py once to generate this).
import torch
import lightning as lit
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from net.base import LSTMClassifier, ConvAE # Import your models
from dataset.loaders import load_har_loso, HarDataset
from hydra import compose, initialize
from omegaconf import OmegaConf
import numpy as np

def run_loso_evaluation():
    # 1. Load Hydra Config manually (since we are controlling the loop)
    with initialize(version_base=None, config_path="../cfg"):
        cfg = compose(config_name="base")

    # 2. Get list of all subjects
    # We load the dataset once just to read the subject IDs
    temp_dataset = HarDataset(cfg.dataset.data_dir, split='train')
    all_subjects = np.unique(temp_dataset.subjects)
    print(f"Found subjects: {all_subjects}")

    # Store results
    loso_accuracies = []
    loso_f1s = []

    # 3. THE LOOP: Iterate over every subject
    for subject_id in all_subjects:
        print(f"\n\n=== STARTING FOLD: Test Subject {subject_id} ===")
        
        # A. Data Loaders for this specific fold
        train_loader, val_loader = load_har_loso(
            cfg.dataset.data_dir, 
            test_subject_id=subject_id, 
            batch_size=cfg.dataset.batch_size
        )

        # B. Load Pre-trained Encoder (Frozen)
        # Note: In a rigorous paper, you should re-train the AE for every fold too.
        # But for efficiency, we often use one generic AE trained on everyone.
        # Ensure you have a 'best_encoder.ckpt' saved somewhere from a previous run.
        ae_path = "checkpoints/best_encoder.ckpt" 
        conv_ae = ConvAE.load_from_checkpoint(ae_path, cfg=cfg.net)
        
        # C. Initialize a Fresh LSTM
        model = LSTMClassifier(cfg.net, conv_ae.encoder)

        # D. Trainer (Fast training for each fold)
        trainer = Trainer(
            max_epochs=20, # LOSO folds are usually shorter
            accelerator="auto",
            devices=1,
            enable_checkpointing=False, # We just want the final score
            logger=False # Turn off wandb logging per fold to avoid spam
        )

        # E. Train
        trainer.fit(model, train_loader, val_loader)
        
        # F. Test (Validate on the left-out subject)
        results = trainer.validate(model, val_loader)
        
        # G. Save Metrics
        acc = results[0]['val_acc'] # Or test_acc
        print(f"Subject {subject_id} Accuracy: {acc:.4f}")
        loso_accuracies.append(acc)

    # 4. Final Report
    print("\n" + "="*30)
    print("LOSO RESULTS")
    print("="*30)
    print(f"Mean Accuracy: {np.mean(loso_accuracies):.4f}")
    print(f"Std Dev:       {np.std(loso_accuracies):.4f}")
    print(f"Min (Hardest): {np.min(loso_accuracies):.4f}")
    print(f"Max (Easiest): {np.max(loso_accuracies):.4f}")

if __name__ == "__main__":
    run_loso_evaluation()