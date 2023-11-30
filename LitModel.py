# from dino_trunc import dino_trunc
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

class LitModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.save_hyperparameters()
        # self.model = dino_trunc()
        self.model = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
        # only train linear layer
        for p in self.model.parameters():
            p.requires_grad = False
        self.linear = nn.Linear(384, num_classes)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        self.log("val_acc", self.accuracy(y_hat, y), prog_bar=True, sync_dist=True)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=3e-4)