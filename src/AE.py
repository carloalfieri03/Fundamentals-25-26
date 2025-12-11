import hydra
from omegaconf import DictConfig  
import lightning as lit
from net.base import ConvAE, LSTMClassifier
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import wandb
from lightning.pytorch.loggers import WandbLogger
from dataset.loaders import load_har, load_wisdm

import lightning.pytorch.callbacks as cb 

from omegaconf import OmegaConf

import os

@hydra.main(config_path="../cfg", config_name="base", version_base=None)
def main(cfg: DictConfig):
    lit.seed_everything(cfg.seed) 
    wandb_logger = WandbLogger(**cfg.wandb)
    
    # 1. Load Data
    train_loader, val_loader, test_loader = load_har(**cfg.dataset)
    
    # 2. Instantiate ConvAE Model
    conv_ae = ConvAE(cfg.net)

    # 3. Setup ConvAE Callbacks
    ae_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="convAE-{epoch:02d}-{val_loss:.4f}"
    )
    patience=cfg.callbacks.get("patience", 20)
    ae_earlystop = EarlyStopping(monitor="val_loss", patience=patience, mode="min")

    # 4. Setup ConvAE Trainer
    ae_trainer = Trainer(
        logger=wandb_logger,
        callbacks=[ae_checkpoint, ae_earlystop],
        **cfg.trainer
    )

    # 5. FIT (Train) the ConvAE
    print("--- Starting Convolutional Autoencoder (ConvAE) Training ---")
    ae_trainer.fit(conv_ae, train_loader, val_loader)
    print("--- ConvAE Training Complete ---")
    
    # Optional: Log the best ConvAE loss and the path to the best model
    best_loss = ae_checkpoint.best_model_score.item()
    best_ae_path = ae_checkpoint.best_model_path
    best_ae_model = ConvAE.load_from_checkpoint(ae_checkpoint.best_model_path,cfg=cfg.net)

    print(f"Best Validation Loss: {best_loss:.4f}")
    print(f"Best ConvAE Model saved at: {best_ae_path}")

    ae_trainer.test(best_ae_model, test_loader)

def get_num_params(module):
    """
    Returns the number of parameters in a Lightning module.
    
    Args:
        module (lightning.pytorch.LightningModule): The Lightning module to get the number of parameters for.
    
    Returns:
        int: The number of parameters in the module.
    """
    total_params = sum(p.numel() for p in module.parameters() )
    return total_params




if __name__ == "__main__":
    os.environ['HYDRA_FULL_ERROR'] = "1"
    main()
    

# The helper function 'get_num_params' and the '__main__' block should remain unchanged.