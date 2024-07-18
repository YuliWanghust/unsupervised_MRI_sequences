import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


class SimSiamLM(pl.LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=100):

        super().__init__()

        # Save constructor parameters to self.hparams
        self.save_hyperparameters()

        # Base encoder
        self.convnet = torchvision.models.resnet18(weights=None, num_classes=4 * self.hparams.hidden_dim)
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,
            nn.ReLU(inplace=True),
            nn.Linear(4 * self.hparams.hidden_dim, self.hparams.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim)
        )

        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.hparams.hidden_dim // 4, self.hparams.hidden_dim)
        )

    def configure_optimizers(self):
        """
        Lightning Module utility method. Using AdamW optimiser with
        CosineAnnealingLR scheduler. Do not call this method. 
        """
        # AdamW decouples weight decay from gradient updates
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Set learning rate using a cosine annealing schedule
        # See https://pytorch.org/docs/stable/optim.html
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.lr / 50
        )

        return [optimizer], [lr_scheduler]

    def forward(self, x):
        """
        Performs forward pass on the input data.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output data.
        """
        x = self.convnet(x)
        x = self.prediction_head(x)
        return x

    def loss_function(self, p, z):
        z = z.detach()  # Stop gradient
        p = nn.functional.normalize(p, dim=-1)
        z = nn.functional.normalize(z, dim=-1)
        return -2 * (p * z).sum(dim=1).mean()  # Negative cosine similarity

    def step(self, batch, mode="train"):
        """
        Performs a forward pass for a given batch. This method should not be
        called. Use fit() instead.
        """
        (x1, x2), _ = batch
        # p1, p2 = self(x1), self(x2)
        # loss = (self.loss_function(p1, p2) + self.loss_function(p2, p1)) / 2
        # self.log(f"{mode}_loss", loss)
        z1, z2 = self.convnet(x1), self.convnet(x2)
        p1, p2 = self.prediction_head(z1), self.prediction_head(z2)
        loss = (self.loss_function(p1, z2) + self.loss_function(p2, z1)) / 2
        self.log(f"{mode}_loss", loss)

        return loss

    def training_step(self, batch, batch_index):
        """
        Performs a forward training pass for a given batch. Lightning Module
        utility method. This method should not be called. Use fit() instead.
        """
        return self.step(batch, mode="train")

    def validation_step(self, batch, batch_index):
        """
        Performs a forward validation pass for a given batch. Lightning Module
        utility method. This method should not be called. Use fit() instead.
        """
        self.step(batch, mode="val")
