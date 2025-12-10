import torch
import lightning as lit
import torch.nn.functional as F
import torch.nn as nn
from hydra.utils import instantiate
from hydra.utils import get_class
from torchmetrics import Accuracy
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from torchmetrics.classification import ConfusionMatrix
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import wandb


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
        # --- 1. Create Mask ---
        # Create a mask of 0s and 1s with the same shape as x
        # Probability of masking (setting to 0) = 0.25
        mask_ratio = 0.2
        mask = torch.rand_like(x) > mask_ratio
        
        # --- 2. Mask the Input ---
        # x_masked has holes in it. x is the target (perfect signal).
        x_masked = x * mask.type_as(x)


        x_hat = self(x_masked)
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
       # 1. Metrics Setup
        self.conf_mat = ConfusionMatrix(task="multiclass", num_classes=cfg.num_classes)
        self.class_names = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]
        
        self.f1 = MulticlassF1Score(num_classes=cfg.num_classes, average='macro')
        self.precision = MulticlassPrecision(num_classes=cfg.num_classes, average='macro')
        self.recall = MulticlassRecall(num_classes=cfg.num_classes, average='macro')
        self.f1_per_class = MulticlassF1Score(num_classes=cfg.num_classes, average=None)

        self.train_acc = Accuracy(task='multiclass', num_classes=cfg.num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=cfg.num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=cfg.num_classes)
        
       
        self.save_hyperparameters(ignore=["pretrained_encoder"])
        self.cfg = cfg

        self.encoder = pretrained_encoder
        
        self.encoder.eval() 
        
        # FIX 2: Unfreeze weights   
# --- ADD THIS (UNFREEZE) ---
        for p in self.encoder.parameters():
            p.requires_grad = False 

       # 3. LSTM Setup (MISSING PART RESTORED)
        # -------------------------------------------------------
        # We detect the encoder output size safely (handling GPU/CPU)
        device = next(self.encoder.parameters()).device
        dummy = torch.zeros(1, 9, 128).to(device)
        
        with torch.no_grad():
            enc_out_dim = self.encoder(dummy).shape[1]
            
        # Create the LSTM with the correct input size
        self.lstm = instantiate(cfg.rnn_block, input_size=enc_out_dim)



# --- CHANGE 1: Calculate New Input Size ---
        # LSTM Output Size
        lstm_dim = cfg.rnn_block.hidden_size * (2 if cfg.rnn_block.bidirectional else 1)
        
        # Gravity Vector Size (We have 9 sensors, so 9 mean values)
        gravity_dim = 9 
        
        # Combined Size for the Classifier Head
        total_input_dim = lstm_dim + gravity_dim 

        # FIX 4: Restore Deep Head with Dropout (Fights 100% Overfitting)
        lstm_out_dim = cfg.rnn_block.hidden_size * (2 if cfg.rnn_block.bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Dropout(0.2), # Keep this to prevent 100% overfitting
            nn.Linear(lstm_out_dim, cfg.num_classes)
        )
        



       
        #self.lstm = instantiate(cfg.rnn_block)

        #self.fc = instantiate(cfg.unembed)

        self.train_acc = Accuracy(task='multiclass', num_classes=cfg.num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=cfg.num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=cfg.num_classes)

    def forward(self, x):
   
        z = self.encoder(x)

        z = z.permute(0, 2, 1)
        z = self.lstm(z)
        z = z.reshape(z.size(0), -1)

        logits = self.fc(z)
        return logits
    ########################### NEW METHOD #############################
    def configure_optimizers(self):
        OptimizerClass = get_class(self.cfg.optimizer._target_)
        base_lr = self.cfg.optimizer.lr 
        global_wd = self.cfg.optimizer.weight_decay

        # --- SMART FILTERING ---
        # We filter parameters explicitly for each group.
        # If requires_grad is False, the iterator is empty, and the optimizer skips it safely.
        
        encoder_params = filter(lambda p: p.requires_grad, self.encoder.parameters())
        head_params = filter(lambda p: p.requires_grad, list(self.lstm.parameters()) + list(self.fc.parameters()))

        optimizer = OptimizerClass([
            # Group 1: Encoder (Low LR)
            # If frozen, this group is empty and does nothing.
            # If unfrozen, this applies the tiny learning rate.
            {
                'params': encoder_params, 
                'lr': base_lr * 0.01 
            }, 
            
            # Group 2: LSTM & Head (Normal LR)
            {
                'params': head_params, 
                'lr': base_lr 
            }
        ], weight_decay=global_wd)

        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
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
        logits = self(x) # Your forward pass
        preds = torch.argmax(logits, dim=1)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc(y_hat, y), prog_bar=True)
        self.f1.update(y_hat, y)
        self.precision.update(y_hat, y)
        self.recall.update(y_hat, y)
        self.f1_per_class.update(y_hat, y)
        self.conf_mat.update(preds, y)
   
 #### OLD FOR GRAD FROZEN #######  
   # def configure_optimizers(self):
        # 1. Filter: Keep only parameters that are allowed to learn
       # trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        
        # 2. Pass filtered list to optimizer
       # return instantiate(self.cfg.optimizer, trainable_params)
    
    def on_test_epoch_end(self):
        
        self.log("test_f1_macro", self.f1.compute())
        self.log("test_precision_macro", self.precision.compute())
        self.log("test_recall_macro", self.recall.compute())
        
        # Log PER CLASS scores to console or WandB
        per_class_scores = self.f1_per_class.compute()
        
        # Print readable breakdown
        print("\n--- Per-Class F1 Scores ---")
        for i, score in enumerate(per_class_scores):
            class_name = self.class_names[i] if i < len(self.class_names) else str(i)
            print(f"{class_name}: {score:.4f}")
            
            # Log to WandB
            if self.logger:
                self.logger.experiment.log({f"f1_score/{class_name}": score})

        # Reset
        self.f1.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1_per_class.reset()

        # 1. Compute the matrix (on GPU if available)
        cm_tensor = self.conf_mat.compute()
        
        # 2. Move to CPU and convert to Numpy for plotting
        cm = cm_tensor.cpu().numpy()

        # 3. Create the Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 4. Draw Heatmap
        sns.heatmap(
            cm, 
            annot=True,       # Show numbers in boxes
            fmt='d',          # Integers (no decimals)
            cmap='Blues',     # Blue color scheme
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )
        
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.title('Test Confusion Matrix')

        # 5. Log to WandB
        # We check if the logger exists to avoid crashing in dry runs
        if self.logger:
            self.logger.experiment.log({
                "confusion_matrix": wandb.Image(fig),
                "global_step": self.global_step
            })
            
        # 6. Clean up
        plt.close(fig)
        self.conf_mat.reset()

