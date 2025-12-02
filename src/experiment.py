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
    train_loader, val_loader, test_loader = load_har(**cfg.dataset)

    conv_ae = ConvAE(cfg.net)

    ae_trainer = Trainer(
        logger=wandb_logger,
        callbacks=[    
            ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="convAE-{epoch:02d}-{val_loss:.4f}"
        ),
        EarlyStopping(monitor="val_loss", patience=5, mode="min")
        ],
        **cfg.trainer
    )

    ae_trainer.fit(conv_ae, train_loader, val_loader)

    best_encoder_path = ae_checkpoint.best_model_path
    conv_ae = ConvAE.load_from_checkpoint(best_encoder_path, cfg=cfg.net)
    pretrained_encoder = conv_ae.encoder

    lstm_model = LSTMClassifier(cfg.net, pretrained_encoder)


    lstm_trainer = Trainer(
        logger=wandb_logger,
        callbacks=[
        ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="LSTM-{epoch:02d}-{val_acc:.4f}"
        ),
        EarlyStopping(monitor="val_acc", patience=5, mode="max")],
        **cfg.trainer
    )

    lstm_trainer.fit(lstm_model, train_loader, val_loader)
    best_lstm_path = lstm_checkpoint.best_model_path
    best_lstm_model = LSTMClassifier.load_from_checkpoint(best_lstm_path, pretrained_encoder=pretrained_encoder, cfg=cfg.net)
    lstm_trainer.test(best_lstm_model, test_loader)
    
    hyperparams_dict = OmegaConf.to_container(cfg, resolve=True)
    hyperparams_dict["info"] = {  
        "num_params": get_num_params(best_lstm_model),
    }
    wandb_logger.log_hyperparams(hyperparams_dict)  

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