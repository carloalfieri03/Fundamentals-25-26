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
        self.beta= cfg.get("beta", 0.08)
        self.pooling=cfg.get("pooling_type","avg")
        self.encoder = instantiate(cfg.embed)
        # 2. Get the output channels from the config (or the object itself)
        latent_dim = cfg.embed.out_channels 
        
        # 3. Instantiate Decoder, FORCING in_channels to match the encoder
        # This overrides 'in_channels: 128' from the yaml file
        self.decoder = instantiate(cfg.decoder, in_channels=latent_dim)
        self.test_acc = Accuracy(task='multiclass', num_classes=cfg.num_classes)

    def forward(self, x):
        original_length = x.shape[-1]
        z = self.encoder(x)
        x_hat = self.decoder(z)
        # 3. CRASH PREVENTION / LENGTH CORRECTION
        
        if x_hat.shape[-1] != original_length:
            # Interpolate (resize) the output to exactly match the input's length (128)
            # This uses the linear interpolation mode appropriate for 1D time series data
            x_hat = F.interpolate(x_hat, 
                                  size=original_length, 
                                  mode='linear', 
                                  align_corners=False)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, _ = batch 
        z= self.encoder(x) 
        x_hat = self(x)
        loss = F.smooth_l1_loss(x_hat, x, beta=self.beta)
        #mse_loss=F.mse_loss(x_hat,x)
        self.log("train_loss", loss, prog_bar=True)
        if batch_idx == 0:
            self.log_visualizations(x, x_hat, z)
        return loss

    def log_visualizations(self, x, x_hat, z):
        """
        Plots:
        1. Reconstruction comparison (Line Plot)
        2. Bottleneck Activity (Heatmap)
        """
        # Move to CPU/Numpy
        orig = x[0].cpu().numpy()       # Shape: [9, 128]
        recon = x_hat[0].detach().cpu().numpy() # Shape: [9, 128]
        latent = z[0].detach().cpu().numpy()    # Shape: [64, 32] (Example)

        # --- PLOT 1: Reconstruction (Input vs Output) ---
        fig_recon, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axis_labels = ['Acc X', 'Acc Y', 'Acc Z'] # Assuming first 3 are Acc
        
        for i in range(3):
            axes[i].plot(orig[i], label='Original', color='blue', alpha=0.6)
            axes[i].plot(recon[i], label='Reconstruction', color='red', linestyle='--', alpha=0.8)
            axes[i].set_ylabel(axis_labels[i])
            axes[i].legend(loc='upper right')
            axes[i].grid(True, alpha=0.3)
            
        plt.xlabel("Time Steps")
        plt.suptitle(f"Reconstruction (Val Loss: {self.trainer.callback_metrics.get('val_loss', 0):.4f})")
        plt.tight_layout()
        
        # --- PLOT 2: Bottleneck Heatmap ---
        # This shows the "Compressed Signal"
        fig_latent, ax_lat = plt.subplots(figsize=(10, 4))
        # Use imshow to visualize the 2D matrix of features [Channels, Time]
        im = ax_lat.imshow(latent, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(im, ax=ax_lat, label="Activation")
        ax_lat.set_title(f"Bottleneck Features (Shape: {latent.shape})")
        ax_lat.set_ylabel("Feature Channels")
        ax_lat.set_xlabel("Compressed Time Steps")
        plt.tight_layout()

        # Log both to WandB
        self.logger.experiment.log({
            "Reconstruction Check": wandb.Image(fig_recon),
            "Bottleneck Heatmap": wandb.Image(fig_latent)
        })
        
        # Cleanup
        plt.close(fig_recon)
        plt.close(fig_latent)
        
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = F.smooth_l1_loss(x_hat, x, beta=self.beta)
        #mse_loss=F.mse_loss(x_hat,x)
        self.log("val_loss", loss, prog_bar=True)

        # 3. VISUALIZATION (Log only for the first batch to save data)
        if batch_idx == 0:
            self.log_reconstruction_images(x, x_hat)
            
    def log_reconstruction_images(self, x, z):
        # Pick the first sample in the batch
        # Convert to CPU numpy: [Channels, Time]
        orig = x[0].cpu().numpy()
        recon = z[0].detach().cpu().numpy()
        
        # Create a Plot (3 rows: X, Y, Z axes of Accelerometer)
        # Assuming channels 0,1,2 are Acc_X, Acc_Y, Acc_Z
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axis_labels = ['Acc X', 'Acc Y', 'Acc Z']
        
        for i in range(3):
            # Plot Original (Blue)
            axes[i].plot(orig[i], label='Original (Input)', color='blue', alpha=0.7)
            # Plot Reconstruction (Red -- Dashed)
            axes[i].plot(recon[i], label='Reconstruction', color='red', linestyle='--', alpha=0.8)
            axes[i].set_ylabel(axis_labels[i])
            axes[i].legend(loc='upper right')
            axes[i].grid(True, alpha=0.3)
            
        plt.xlabel("Time Steps")
        plt.suptitle(f"Reconstruction Check (Val Loss: {self.trainer.callback_metrics.get('val_loss', 0):.4f})")
        plt.tight_layout()
        
        # Log to W&B
        # This will appear in the "Media" or "Images" section of your dashboard
        self.logger.experiment.log({"Reconstruction Analysis": [wandb.Image(fig)]})
        
        # Close plot to save memory
        plt.close(fig)

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


    # --- 3. Atuomatic input size calculation for LSTM according to the convolution bottleneck dimension ---

        device = next(self.encoder.parameters()).device ## To solve GPU/CPU mismatch errors

        dummy_input = torch.randn(1, 9, 128).to(device) ## Update if you change time window
        
        with torch.no_grad():
            z_dummy = self.encoder(dummy_input) # Runs the AE one time to see the shape
            
        lstm_input_dim = z_dummy.shape[1] ## Getting the input size 

        self.lstm = instantiate(cfg.rnn_block, input_size=lstm_input_dim) ## Passing the automatic input size
        # Check for bidirectional
        is_bidirectional = getattr(self.lstm, 'bidirectional', False)
        num_directions = 2 if is_bidirectional else 1
       # 
        lstm_out_dim = self.lstm.hidden_size * num_directions
       
        #self.bn=nn.BatchNorm1d(lstm_input_dim)
        self.fc= nn.LazyLinear(cfg.num_classes)
        #self.bn_lstm = nn.BatchNorm1d(lstm_out_dim)
        
       
    def forward(self, x):
        z = self.encoder(x)
       # 2. Calculate Gravity Features (Global Average Pooling)
        # We use keepdim=True so it stays 3D: (Batch, Input_Channels, 1)
        #gravity_features = torch.mean(x, dim=2, keepdim=True) 
        #z= self.bn(z)
        # 3. Expand Gravity Features to match z's sequence length
        # Shape becomes: (Batch, Input_Channels, Seq_Len_Reduced)
       # gravity_features = gravity_features.expand(-1, -1, z.shape[2])
        
        # 4. Concatenate along the Feature dimension (dim=1)
        # New Shape: (Batch, Enc_Channels + Input_Channels, Seq_Len_Reduced)
       # z = torch.cat((z, gravity_features), dim=1)
       # gravity_features = torch.mean(x, dim=2)
        z = z.permute(0, 2, 1)
        lstm_out = self.lstm(z)
        #lstm_out= self.bn_lstm(lstm_out)
        
        # Classify
        logits = self.fc(lstm_out)
      
        # 3. LSTM
        # LSTM returns tuple: (output, (h_n, c_n))
        # output shape: [Batch, Time, Hidden_Size]
        
        # lstm_out= output at each step (output, (hn,cn)) ## hn and cn are final output and cell state ( last long term memo)

 

        
        #logits = self.fc(lstm_out)
        #print ( f'Logits {logits}') 
        ## tensor for the whole batch with shape [Batch, Time, Num_Classes] useful to see class confusion before softmax
        ## NB with this we can check for class confusion. #### FOCAL LOSS -> think about it 
        return logits

    def configure_optimizers(self):
        # Filter: Keep only parameters that are allowed to learn (LSTM + FC)
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        return instantiate(self.cfg.optimizer, params=trainable_params)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        w_list = self.cfg.get("lstm_weigts", [1.0]*self.cfg.num_classes)
        
        # 2. Create Tensor AND move to device immediately
        class_weights = torch.tensor(w_list, dtype=torch.float).to(self.device)


    
# Update loss
        loss = F.cross_entropy(y_hat, y,weight=class_weights)
        #loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        w_list = self.cfg.get("lstm_weigts", [1.0]*self.cfg.num_classes)
        
        # 2. Create Tensor AND move to device immediately
        class_weights = torch.tensor(w_list, dtype=torch.float).to(self.device)
        loss = F.cross_entropy(y_hat, y,weight=class_weights)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc(y_hat, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
     
        
        logits = self(x) # Your forward pass
        
        # 1. Convert Logits to Probabilities
        probs = F.softmax(logits, dim=1)
        
        # 2. Get the winner and its confidence score
        max_probs, preds = torch.max(probs, dim=1)
        
        # 3. Define your "Watchlist" (Sitting=3, Standing=4)
        watch_classes = [3, 4] 
        
        # 4. Iterate through the batch to find interesting cases
        for i in range(len(y)):
            true_label = y[i].item()
            pred_label = preds[i].item()
            confidence = max_probs[i].item()
            
            # --- FILTER CONDITIONS ---
            # A. Is the model uncertain? (e.g., prediction confidence < 70%)
            is_uncertain = confidence < 0.70
            
            # B. Is it one of our problem classes? (Truth OR Pred is Sitting/Standing)
            is_problem_class = (true_label in watch_classes) or (pred_label in watch_classes)
            
            # C. Is it actually wrong?
            is_wrong = true_label != pred_label
            
            # COMBINE: Show me cases that are WRONG involving SITTING/STANDING
            # (You can change 'and' to 'or' depending on what you want to see)
            if is_problem_class and is_wrong:
                print(f"\n⚠️  MISCLASSIFICATION DETECTED (Batch {batch_idx}, Sample {i})")
                print(f"   True: {self.class_names[true_label]}  -->  Pred: {self.class_names[pred_label]}")
                print(f"   Confidence: {confidence:.4f} (Uncertainty: {1.0 - confidence:.4f})")
                
                # Print the "Hedge": The scores for the specific classes you care about
                print("   --- Scores ---")
                print(f"   Sitting (3):  Logit={logits[i][3]:.2f} | Prob={probs[i][3]:.4f}")
                print(f"   Standing (4): Logit={logits[i][4]:.2f} | Prob={probs[i][4]:.4f}")
                
                # Check the Margin (Distance between the two)
                margin = abs(probs[i][3] - probs[i][4])
                print(f"   Margin: {margin:.4f}")
                print("-" * 50)

    
        
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

