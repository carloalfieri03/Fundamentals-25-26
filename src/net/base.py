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

## DA FARE:
# Unfreeze dell'encoder durante training LSTM
# Lazy linear optimization
# dataset

class ConvAE(lit.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.beta= cfg.get("beta", 0.08)
        self.encoder = instantiate(cfg.embed)
        # 2. Get the output channels from the config (or the object itself)
        # This assumes your encoder config has 'out_channels'
        latent_dim = cfg.embed.out_channels 
        
        # 3. Instantiate Decoder, FORCING in_channels to match the encoder
        # This overrides 'in_channels: 128' from the yaml file
        self.decoder = instantiate(cfg.decoder, in_channels=latent_dim)
        self.test_acc = Accuracy(task='multiclass', num_classes=cfg.num_classes)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        x, _ = batch  
        x_hat = self(x)
        loss = F.smooth_l1_loss(x_hat, x, beta=self.beta)
        #mse_loss=F.mse_loss(x_hat,x)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = F.smooth_l1_loss(x_hat, x, beta=self.beta)
        #mse_loss=F.mse_loss(x_hat,x)
        self.log("val_loss", loss, prog_bar=True)

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

        #self.encoder.train() 

        #self.loss_fn = FocalLoss(gamma=2) ### FOCAL LOSS init

    # --- 3. Atuomatic input size calculation for LSTM according to the convolution bottleneck dimension ---

        device = next(self.encoder.parameters()).device ## To solve GPU/CPU mismatch errors

        dummy_input = torch.randn(1, 9, 128).to(device) ## Update if you change time window
        
        with torch.no_grad():
            z_dummy = self.encoder(dummy_input) # Runs the AE one time to see the shape
            
        lstm_input_dim = z_dummy.shape[1] ## Getting the input size 

        self.lstm = instantiate(cfg.rnn_block, input_size=lstm_input_dim) ## Passing the automatic input size
        self.fc = instantiate(cfg.fc_block) ## Fully connected layer ### CHECK HOW THIS CAN AFFECT PERFORMANCE OF THE MODEL

    def forward(self, x):
        z = self.encoder(x) 
       
        z = z.permute(0, 2, 1) ## reshaping for Lstm


        # 3. LSTM
        # LSTM returns tuple: (output, (h_n, c_n))
        # output shape: [Batch, Time, Hidden_Size]
        lstm_out = self.lstm(z) 
        # lstm_out= output at each step (output, (hn,cn)) ## hn and cn are final output and cell state ( last long term memo)

        # 4. CLASSIFICATION HEAD
     
        
        logits = self.fc(lstm_out)
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
        # If you forget .to(self.device), this will CRASH on GPU
        class_weights = torch.tensor(w_list, dtype=torch.float).to(self.device)
# We give higher weights (2.0) to class 3 and 4 (Sitting/Standing)

    
# Update loss
        loss = F.cross_entropy(y_hat, y,weight=class_weights)
        #loss = self.loss_fn(y_hat, y)
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

