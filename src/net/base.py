import torch
import lightning as lit
import torch.nn.functional as F
import torch.nn as nn
from hydra.utils import instantiate
from torchmetrics import Accuracy


class ConvAE(lit.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.encoder = instantiate(cfg.embed)
        self.decoder = instantiate(cfg.decoder)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, _ = batch  
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return instantiate(self.cfg.optimizer, self.parameters())

class LSTMClassifier(lit.LightningModule):
    def __init__(self, cfg, pretrained_encoder):
        super().__init__()
        self.save_hyperparameters(ignore=["pretrained_encoder"])
        self.cfg = cfg

        self.encoder = pretrained_encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False   

        self.lstm = instantiate(cfg.rnn_block)

        self.fc = instantiate(cfg.unembed)

        self.train_acc = Accuracy(task='multiclass', num_classes=cfg.num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=cfg.num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=cfg.num_classes)

    def forward(self, x):
        with torch.no_grad():
            z = self.encoder(x)

        z = z.permute(0, 2, 1)
        z = self.lstm(z)
        z = z.reshape(z.size(0), -1)

        logits = self.fc(z)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc(y_hat, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc(y_hat, y), prog_bar=True)

    def configure_optimizers(self):
        return instantiate(self.cfg.optimizer, self.parameters())
