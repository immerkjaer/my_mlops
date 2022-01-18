from src.models.model import MNISTmodel, MNISTData
from tests import _MODEL_DIR
import pytorch_lightning as pl
import torch

import numpy as np

import os

data_module = MNISTData()
model = MNISTmodel()

model.load_existing_model(os.path.join(_MODEL_DIR, "model_saved.pth") )

trainer = pl.Trainer(max_epochs = 2, gpus=1)

test = trainer.predict(model, data_module)
# print(type(test))
output = [t.numpy() for t in test]
print(np.shape(np.array(output)))

# output is list of tensor (size 64 because of batch size) and 78 of these + a single with 8
