import numpy as np
import pytorch_lightning as pl
import torch
from Classes import VAE
from LDM_Classes import LDM, AttentionUNet
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

num_devices = 3
num_nodes = 4
num_workers = 1
accelerator = "gpu"
batch_size = 100
epochs = -1

# terms VAE
reconstruction_term = 0
perceptual_term = 1
kl_term = 0.3
adv_term = 0
lr_VAE = 1e-3
lr_DDPM = 1e-3


checkpoint_path_VAE = (
    "/home/mbezick/Desktop/logs/VAE_V6/version_60/checkpoints/good-v1.ckpt"
)
checkpoint_path_LDM = "/home/mbezick/Desktop/logs/LDM/version_0/checkpoints/good.ckpt"

checkpoint_callback = ModelCheckpoint(filename="good", every_n_train_steps=300)

torch.set_float32_matmul_precision("high")

vae = VAE(reconstruction_term, perceptual_term, kl_term, adv_term, 1, 64, lr=lr_VAE)
vae = vae.load_from_checkpoint(
    checkpoint_path_VAE,
    a=reconstruction_term,
    b=perceptual_term,
    c=kl_term,
    d=adv_term,
    in_channels=1,
    h_dim=64,
    lr=lr_VAE,
)
vae.eval()

DDPM = AttentionUNet(
    in_channels=1,
    out_channels=1,
    UNet_channel=64,
    timeEmbeddingLength=16,
    batch_size=batch_size,
    height=8,
    width=8,
    num_steps=1000,
    condition_vector_size=16,
)

ldm = LDM(
    DDPM,
    vae,
    in_channels=1,
    batch_size=batch_size,
    num_steps=1000,
    height=8,
    width=8,
    lr=lr_DDPM,
)

# loading dataset
dataset = np.expand_dims(np.load("top_0.npy"), 1)
normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

normalizedDataset = np.multiply(normalizedDataset, 2) - 1

normalizedDataset = normalizedDataset.astype(np.float32)

dataset = torch.from_numpy(normalizedDataset)


# defining training classes
train_loader = DataLoader(
    dataset,
    num_workers=num_workers,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

logger = TensorBoardLogger(save_dir="logs/", name="LDM")

lr_monitor = LearningRateMonitor(logging_interval="step")
trainer = pl.Trainer(
    logger=logger,
    devices=num_devices,
    num_nodes=num_nodes,
    accelerator="gpu",
    log_every_n_steps=1,
    callbacks=[checkpoint_callback],
    max_epochs=epochs,
    strategy="ddp_find_unused_parameters_true",
    detect_anomaly=True,
)


trainer.fit(
    model=ldm, train_dataloaders=train_loader
)  # , ckpt_path = checkpoint_path_LDM)
