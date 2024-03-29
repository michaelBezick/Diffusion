import os

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from LDM_Classes import VAE, LabeledDataset

"""
GOAL: Make a BVAE that compresses 32x32 space to 8x8. No energy correlation.
"""

resume_from_checkpoint = False
num_devices = 1
num_nodes = 1
num_workers = 1
accelerator = "gpu"
batch_size = 100
epochs = 10_000
lr = 1e-3

##################################################

checkpoint_path = None
if resume_from_checkpoint:
    checkpoint_path1 = f"./logs/"
    checkpoint_path2 = os.listdir(checkpoint_path1)[-1]
    checkpoint_path = os.path.join(checkpoint_path1, checkpoint_path2)
    checkpoint_path = checkpoint_path + "/checkpoints/"
    file_checkpoint = os.listdir(checkpoint_path)[0]
    checkpoint_path = os.path.join(checkpoint_path, file_checkpoint)

checkpoint_callback = ModelCheckpoint(filename="good", every_n_train_steps=300)

torch.set_float32_matmul_precision("high")

vae = VAE(in_channels=1, h_dim=128, batch_size=batch_size, lr=lr)

dataset = np.expand_dims(np.load("./Files/TPV_dataset.npy"), 1)
normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

normalizedDataset = np.multiply(normalizedDataset, 2) - 1

normalizedDataset = normalizedDataset.astype(np.float32)

dataset = torch.from_numpy(normalizedDataset)

labels = torch.load("./Files/FOM_labels.pt")

labeled_dataset = LabeledDataset(dataset, labels)

train_loader = DataLoader(
    labeled_dataset,
    num_workers=num_workers,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

logger = TensorBoardLogger(save_dir="logs/", name="VAE")

lr_monitor = LearningRateMonitor(logging_interval="step")

trainer = pl.Trainer(
    logger=logger,
    devices=num_devices,
    num_nodes=num_nodes,
    accelerator="gpu",
    log_every_n_steps=2,
    max_epochs=epochs,
)

if resume_from_checkpoint:
    trainer.fit(model=vae, train_dataloaders=train_loader, ckpt_path=checkpoint_path)
else:
    trainer.fit(model=vae, train_dataloaders=train_loader)
