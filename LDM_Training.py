import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from LDM_Classes import LDM, VAE, AttentionUNet, LabeledDataset

num_devices = 2
num_nodes = 2
num_workers = 1
accelerator = "gpu"
batch_size = 100
epochs = 10_000
in_channels = 1
out_channels = 1

# terms VAE
lr_VAE = 1e-3
lr_DDPM = 1e-3
perceptual_loss_scale = 1
kl_divergence_scale = 0.3

resume_from_checkpoint = False

checkpoint_path_VAE = "./logs/VAE/version_0/checkpoints/epoch=1115-step=33480.ckpt"

#checkpoint_path_LDM = "/home/mbezick/Desktop/logs/LDM/version_0/checkpoints/good.ckpt"

###########################################################################################

checkpoint_callback = ModelCheckpoint(filename="good", every_n_train_steps=300)

torch.set_float32_matmul_precision("high")

vae = VAE(
    in_channels=in_channels,
    h_dim=128,
    lr=lr_VAE,
    batch_size=batch_size,
    perceptual_loss_scale=perceptual_loss_scale,
    kl_divergence_scale=kl_divergence_scale,
)
vae = vae.load_from_checkpoint(
    checkpoint_path=checkpoint_path_VAE,
    in_channels=in_channels,
    h_dim=128,
    lr=lr_VAE,
    batch_size=batch_size,
    perceptual_loss_scale=perceptual_loss_scale,
    kl_divergence_scale=kl_divergence_scale,
)
vae.eval()

DDPM = AttentionUNet(
    in_channels=in_channels,
    out_channels=out_channels,
    UNet_channel=64,
    timeEmbeddingLength=16,
    batch_size=batch_size,
    latent_height=8,
    latent_width=8,
    num_steps=1000,
    FOM_condition_vector_size=16,
)

ldm = LDM(
    DDPM,
    vae,
    in_channels=in_channels,
    batch_size=batch_size,
    num_steps=1000,
    latent_height=8,
    latent_width=8,
    lr=lr_DDPM,
)

# loading dataset
dataset = np.expand_dims(np.load("./Files/TPV_dataset.npy"), 1)
normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

normalizedDataset = np.multiply(normalizedDataset, 2) - 1

normalizedDataset = normalizedDataset.astype(np.float32)

dataset = torch.from_numpy(normalizedDataset)

labels = torch.load("./Files/FOM_labels.pt")

labeled_dataset = LabeledDataset(dataset, labels)


# defining training classes
train_loader = DataLoader(
    labeled_dataset,
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
    log_every_n_steps=2,
    max_epochs=epochs,
    strategy="ddp_find_unused_parameters_true",
)

if resume_from_checkpoint:
    trainer.fit(
        model=ldm, train_dataloaders=train_loader, ckpt_path=checkpoint_path_LDM
    )
else:
    trainer.fit(model=ldm, train_dataloaders=train_loader)
