import os
from src.models import _PROCESSED_DATA, _RAW_DATA, _MODEL_DIR

import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision import transforms
from src.libs.utils import load_raw_data
from tests.test_data import test_all_labels_represented

from mnist_data_loader import MNISTData

# TODO: hydra + test + prettify + make for fixing script using (black, isort)
# 1e-2
class MNISTmodel(pl.LightningModule):

    def __init__(self, lr):
        self.lr = lr
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):

        batch_size, width, height = x.size()
        x = x.view(batch_size, -1)

        return self.classifier(x)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        # calculate acc
        labels_hat = torch.argmax(logits, dim=1)
        test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

        # log the outputs
        self.log_dict({'test_loss': loss, 'test_acc': test_acc})

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
        return optimizer
    
    def safe_model(self, output_filepath_model):
        torch.save(self.classifier.state_dict(), os.path.join(output_filepath_model, "model_saved.pth"))

    def load_existing_model(self, input_filepath_model):
        state_dict = torch.load(input_filepath_model)
        self.classifier.load_state_dict(state_dict)

        return self.classifier

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch

        logits = self.forward(x)
        return logits




data_module = MNISTData()
model = MNISTmodel()

model.safe_model(_MODEL_DIR)
trainer = pl.Trainer(max_epochs = 2, gpus=0)

trainer.fit(model, data_module)
trainer.test(model, data_module)
