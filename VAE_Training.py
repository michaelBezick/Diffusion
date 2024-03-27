import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
from LDM_Classes import VAE_GAN
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

num_devices = 1
num_nodes = 1
num_workers = 1
accelerator = "gpu"
batch_size = 1
epochs = -1

reconstruction_term = 0
perceptual_term = 1
kl_term = 1
adv_term = .3
lr = 1e-3


checkpoint_path = "/home/mbezick/Desktop/logs/VAE_V6/version_25/checkpoints/good-v1.ckpt"
checkpoint_callback = ModelCheckpoint(filename = "good", every_n_train_steps = 300)

torch.set_float32_matmul_precision('high')

vae = VAE_GAN(reconstruction_term, perceptual_term, kl_term, adv_term, 1, 64, lr = lr)
dataset = np.expand_dims(np.load("top_0.npy"), 1)
normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))
    
normalizedDataset = np.multiply(normalizedDataset, 2) - 1

normalizedDataset = normalizedDataset.astype(np.float32)

dataset = torch.from_numpy(normalizedDataset)

train_loader = DataLoader(dataset, num_workers = num_workers, batch_size = batch_size, shuffle = True, drop_last = True)

logger = TensorBoardLogger(save_dir='logs/', name='VAE_V6')

lr_monitor = LearningRateMonitor(logging_interval = 'step')
trainer = pl.Trainer(logger = logger, devices = num_devices, num_nodes = num_nodes, accelerator = "gpu", detect_anomaly=True)


trainer.fit(model = vae, train_dataloaders = train_loader) #, ckpt_path = checkpoint_path)

