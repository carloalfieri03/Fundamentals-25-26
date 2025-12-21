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
import random

class ConvAE(lit.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.beta= cfg.get("beta", 0.08)
        self.pooling=cfg.get("pooling_type","avg")
        self.encoder = instantiate(cfg.embed)
        latent_dim = cfg.embed.out_channels 
        
        # Safety check for decoder input
        self.decoder = instantiate(cfg.decoder, in_channels=latent_dim)
        self.test_acc = Accuracy(task='multiclass', num_classes=cfg.num_classes)

    def forward(self, x):

        ''' The commented code in this part was used for running a sweep for the AE hyperparameters 
            to avoid dimensions mismatch due to kernel size, padding combinations '''
        
        #original_length = x.shape[-1]  
        z = self.encoder(x)
        x_hat = self.decoder(z)
        
        #if x_hat.shape[-1] != original_length:
            # Interpolate (resize) the output to exactly match the input's length (128)
            # This uses the linear interpolation mode appropriate for 1D time series data
            #x_hat = F.interpolate(x_hat, 
                                  #size=original_length, 
                                  #mode='linear', 
                                 # align_corners=False)
        return x_hat


    def training_step(self, batch, batch_idx):

        ''' Training using Huber Loss with the beta parameter configured in the base.yaml 
        The beta value was set out via bayesian search with a wandb agent sweep. 
        
        N.B. We extract x and and the label y in this step only for visualization purposes since in log visualization we compare AE reconstruction with its label. 
        If we want to perform only the AE training we don't need the label yy since the loss is computed between x-hat and x, which is the main benefit of this architecture '''

        x, y = batch 
        z= self.encoder(x) 
        x_hat = self(x)
        loss = F.smooth_l1_loss(x_hat, x, beta=self.beta)
        #mse_loss=F.mse_loss(x_hat,x)
        self.log("train_loss", loss, prog_bar=True)
       
        if batch_idx == 0:
            self.log_visualizations(x, x_hat,y,stage='train')
        return loss
    
    def log_visualizations(self, x, x_hat, y, stage="train"):
        """
        Finds one Sitting and Standing sample in the batch and plots them to check the reconstruction signals. This is done both for test and validation.
        """

        activity_map = {4: "Sitting", 5: "Standing"}
        targets = {}

        # Search for the samples
        for i in range(len(y)):
            label = y[i].item()
            if label in activity_map and label not in targets:
                targets[label] = i
            if len(targets) == 2: break 

        if not targets:
            return  # if none in the batch

        num_cols = len(targets)
        fig, axes = plt.subplots(3, num_cols, figsize=(6 * num_cols, 10), sharex=True)

        if num_cols == 1:
            axes = axes.reshape(3, 1)

        axis_labels = ['Acc X', 'Acc Y', 'Acc Z']

        # 4. Loop through the activities 
        for col_idx, (act_id, batch_idx) in enumerate(targets.items()):
            orig = x[batch_idx].cpu().numpy()
            recon = x_hat[batch_idx].detach().cpu().numpy()
            act_name = activity_map[act_id]

            for row_idx in range(3):
                ax = axes[row_idx, col_idx]
                ax.plot(orig[row_idx], label='Original', color='blue', alpha=0.5)
                ax.plot(recon[row_idx], label='Recon', color='red', linestyle='--', alpha=0.8)
                
                if row_idx == 0:
                    ax.set_title(f"Activity: {act_name}", fontsize=14, fontweight='bold')
                if col_idx == 0:
                    ax.set_ylabel(axis_labels[row_idx])
                ax.grid(True, alpha=0.3)
                if row_idx == 2:
                    ax.legend(loc='lower right', fontsize='small')

        plt.tight_layout()
        self.logger.experiment.log({f"comparison/{stage}_sit_vs_stand": wandb.Image(fig)})
        plt.close(fig)
        
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self(x)
        loss = F.smooth_l1_loss(x_hat, x, beta=self.beta)
        #mse_loss=F.mse_loss(x_hat,x)
        self.log("val_loss", loss, prog_bar=True)
        
        if batch_idx == 0:
            self.log_visualizations(x, x_hat, y, stage="val")

    def configure_optimizers(self):
        return instantiate(self.cfg.optimizer, self.parameters())
    
    def test_step(self, batch, batch_idx):
        x,_ = batch
        x_hat = self(x)
        loss = F.smooth_l1_loss(x_hat, x, beta=self.beta)
        mse_loss=F.mse_loss(x_hat,x)
        self.log("test_loss", loss, prog_bar=True)
        
       # print(f'Test MSE Loss: {mse_loss.item():.4f}')
        self.log("test/mse_loss", mse_loss, prog_bar=False)
    
        return loss 

class LSTMClassifier(lit.LightningModule):
    def __init__(self, cfg, pretrained_encoder):
        super().__init__()
        
       # 1. Metrics 
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
        
        # Freezing Encoder to pass it to the LSTM
        self.encoder.eval() 
        for p in self.encoder.parameters():
            p.requires_grad = False


    #  Atuomatic input size calculation for LSTM according to the convolution bottleneck dimension ---

        device = next(self.encoder.parameters()).device ## To solve GPU/CPU mismatch errors

        dummy_input = torch.randn(1, 9, 128).to(device) 

        ## Automatic input size check from the encoder latent space
        with torch.no_grad():
            z_dummy = self.encoder(dummy_input) # Runs the AE one time to see the shape
            
        lstm_input_dim = z_dummy.shape[1] ## Getting the input size 

        self.lstm = instantiate(cfg.rnn_block, input_size=lstm_input_dim) ## Passing the automatic input size
        # Check for bidirectional
        is_bidirectional = getattr(self.lstm, 'bidirectional', False)
        num_directions = 2 if is_bidirectional else 1
       
        self.fc= nn.LazyLinear(cfg.num_classes)
        
       
    def forward(self, x):
        z = self.encoder(x)
        z = z.permute(0, 2, 1)
        lstm_out = self.lstm(z)
        logits = self.fc(lstm_out)
      
       
        return logits

    def configure_optimizers(self):
        # training only the LSTM parameters
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        return instantiate(self.cfg.optimizer, params=trainable_params)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        w_list = self.cfg.get("lstm_weigts", [1.0]*self.cfg.num_classes) 
        ## used whan weighted cross entropy loss was used if the parameter is not specified in the yaml file the weights are set all to 1 which is equal to not having them
        
        class_weights = torch.tensor(w_list, dtype=torch.float).to(self.device)
        loss = F.cross_entropy(y_hat, y,weight=class_weights)
        #loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        w_list = self.cfg.get("lstm_weigts", [1.0]*self.cfg.num_classes)
        class_weights = torch.tensor(w_list, dtype=torch.float).to(self.device)
        loss = F.cross_entropy(y_hat, y,weight=class_weights)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc(y_hat, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x) # forward pass
        
        probs = F.softmax(logits, dim=1)
        
        max_probs, preds = torch.max(probs, dim=1)
        
        # To check for the type of error in our most critical classes: Sitting=4, Standing=5
        watch_classes = [4, 5] 
        
        # Iterate through the batch to find relevant cases
        for i in range(len(y)):
            true_label = y[i].item()
            pred_label = preds[i].item()
            confidence = max_probs[i].item()
            
            # Select only sitting and standing
            is_problem_class = (true_label in watch_classes) or (pred_label in watch_classes)
            
            # Check for outcome 
            is_wrong = true_label != pred_label
            
            if is_problem_class and is_wrong:
                print(f"\n  MISCLASSIFICATION DETECTED (Batch {batch_idx}, Sample {i})")
                print(f"   True: {self.class_names[true_label]}  -->  Pred: {self.class_names[pred_label]}")
                print(f"   Confidence: {confidence:.4f} (Uncertainty: {1.0 - confidence:.4f})")
                
                print("   --- Scores ---")
                print(f"   Sitting (3):  Logit={logits[i][3]:.2f} | Prob={probs[i][3]:.4f}")
                print(f"   Standing (4): Logit={logits[i][4]:.2f} | Prob={probs[i][4]:.4f}")

    ## Final test loss and accuracy
        preds = torch.argmax(logits, dim=1)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc(y_hat, y), prog_bar=True)

    ## Metrics update
        self.f1.update(y_hat, y)
        self.precision.update(y_hat, y)
        self.recall.update(y_hat, y)
        self.f1_per_class.update(y_hat, y)
        self.conf_mat.update(preds, y)
     
    def on_test_epoch_end(self):
        
        ''' Computation of macro and per class scores '''

        self.log("test_f1_macro", self.f1.compute())
        self.log("test_precision_macro", self.precision.compute())
        self.log("test_recall_macro", self.recall.compute())
        
        per_class_scores = self.f1_per_class.compute()
        
        
        print("\n--- Per-Class F1 Scores ---")
        for i, score in enumerate(per_class_scores):
            class_name = self.class_names[i] if i < len(self.class_names) else str(i)
            print(f"{class_name}: {score:.4f}")
            
            if self.logger:
                self.logger.experiment.log({f"f1_score/{class_name}": score})

        # Reset scores
        self.f1.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1_per_class.reset()

        #  Confusion matrix computation (still on GPU)
        cm_tensor = self.conf_mat.compute()
        
        # Move the confusion matrix to cpu for plotting with numpy and matplotlib
        cm = cm_tensor.cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Heatmap
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

        if self.logger:
            self.logger.experiment.log({
                "confusion_matrix": wandb.Image(fig),
                "global_step": self.global_step
            })
            

        plt.close(fig)
        self.conf_mat.reset()

