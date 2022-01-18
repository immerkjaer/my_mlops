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

class MNISTData(pl.LightningDataModule):
    
    def __init__(self, loader_batch_size, normalize_mean, normalize_std):
        self.loader_batch_size = loader_batch_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def process_and_save_raw_data(self):
        loaded_data = load_raw_data(_RAW_DATA)

        norm = transforms.Normalize((self.normalize_mean,), (self.normalize_std,))

        tensor_train = norm(torch.from_numpy(loaded_data['images_train']).float())
        tensor_train_labels = torch.from_numpy(loaded_data['labels_train'])
        tensor_test = norm(torch.from_numpy(loaded_data['images_test']).float())
        tensor_test_labels = torch.from_numpy(loaded_data['labels_test'])

        torch.save(tensor_train, os.path.join(_PROCESSED_DATA, 'tensor_train.pt'))
        torch.save(tensor_train_labels, os.path.join(_PROCESSED_DATA, 'tensor_train_labels.pt'))
        torch.save(tensor_test, os.path.join(_PROCESSED_DATA, 'tensor_test.pt'))
        torch.save(tensor_test_labels, os.path.join(_PROCESSED_DATA, 'tensor_test_labels.pt'))

    def setup(self, stage):
        if stage == "fit" or stage is None:
            data_path = os.path.join(_PROCESSED_DATA, 'tensor_train.pt')
            label_path = os.path.join(_PROCESSED_DATA, 'tensor_train_labels.pt')
            if not os.path.exists(data_path):
                self.process_and_save_raw_data()
            tensor = torch.load(data_path)
            labels = torch.load(label_path)
            self.trainloader_data = torch.utils.data.TensorDataset(tensor, labels)

        if stage == "validate" or stage is None:
            data_path = os.path.join(_PROCESSED_DATA, 'tensor_test.pt')
            label_path = os.path.join(_PROCESSED_DATA, 'tensor_test_labels.pt')
            if not os.path.exists(data_path):
                self.process_and_save_raw_data()    
            tensor = torch.load(data_path)
            labels = torch.load(label_path)
            self.valloader_data = torch.utils.data.TensorDataset(tensor, labels) 

        if stage == "test" or stage is None:
            data_path = os.path.join(_PROCESSED_DATA, 'tensor_test.pt')
            label_path = os.path.join(_PROCESSED_DATA, 'tensor_test_labels.pt')
            if not os.path.exists(data_path):
                self.process_and_save_raw_data()   
            tensor = torch.load(data_path)
            labels = torch.load(label_path)
            self.testloader_data = torch.utils.data.TensorDataset(tensor, labels) 

        if stage == "predict" or stage is None:
            data_path = os.path.join(_PROCESSED_DATA, 'tensor_test.pt')
            label_path = os.path.join(_PROCESSED_DATA, 'tensor_test_labels.pt')
            if not os.path.exists(data_path):
                self.process_and_save_raw_data()  
            tensor = torch.load(data_path)
            labels = torch.load(label_path)     
            self.predictloader_data = torch.utils.data.TensorDataset(tensor, labels)                      

    def train_dataloader(self):
        return DataLoader(self.trainloader_data, batch_size=self.loader_batch_size)

    def val_dataloader(self):
        return DataLoader(self.valloader_data, batch_size=self.loader_batch_size)

    def test_dataloader(self):
            return DataLoader(self.testloader_data, batch_size=self.loader_batch_size)

    def predict_dataloader(self):
            return DataLoader(self.predictloader_data, batch_size=self.loader_batch_size)