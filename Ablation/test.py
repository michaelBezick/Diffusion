from tensorflow.python.eager.context import num_gpus
import torch
import numpy as np
from LDM_Ablation_Classes import VAE, LabeledDataset, LDM, AttentionUNet
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt
import time

num_devices = 1
num_nodes = 1
num_workers = 1
accelerator = "gpu"
batch_size = 10
epochs = 10_000
in_channels = 1
out_channels = 1

lr_VAE = 1e-3
lr_DDPM = 1e-3
perceptual_loss_scale = 1
kl_divergence_scale = 0.3

def clamp_output(tensor: torch.Tensor, threshold):
    return torch.where(tensor > threshold, torch.tensor(1.0), torch.tensor(0.0))

checkpoint_path_LDM = "../logs/LDM/version_6/checkpoints/epoch=6989-step=209700.ckpt"

ddpm = AttentionUNet(
    UNet_channel=64,
)
vae = VAE(h_dim=128)

ldm = LDM(ddpm, vae, in_channels, batch_size, num_steps=1000, latent_height=1, latent_width=1, lr=1)

ldm = LDM.load_from_checkpoint(
    checkpoint_path_LDM,
    DDPM=ddpm,
    VAE=vae,
    in_channels=in_channels,
    batch_size=batch_size,
    num_steps=1000,
    latent_height=8,
    latent_width=8,
    lr=lr_DDPM,
)

vae = ldm.VAE


vae = VAE.load_from_checkpoint(
    checkpoint_path="../logs/VAE/version_0/checkpoints/epoch=1115-step=33480.ckpt",
    in_channels=in_channels,
    h_dim=128,
    lr=lr_VAE,
    batch_size=batch_size,
    perceptual_loss_scale=perceptual_loss_scale,
    kl_divergence_scale=kl_divergence_scale,
)


vae = vae.eval().to('cuda')

dataset = np.expand_dims(np.load("../Files/TPV_dataset.npy"), 1)
normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

# dataset = clamp_output(normalizedDataset, 0.5)

normalizedDataset = np.multiply(normalizedDataset, 2) - 1

normalizedDataset = normalizedDataset.astype(np.float32)

dataset = torch.from_numpy(normalizedDataset)


labels = torch.load("../Files/FOM_labels_new.pt")

labeled_dataset = LabeledDataset(dataset, labels)


# defining training classes
train_loader = DataLoader(
    labeled_dataset,
    num_workers=num_workers,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)


for batch in train_loader:
    images, labels = batch
    x = images
    x = x.cuda()

    with torch.no_grad():
        mu, sigma = vae.encode(x)

    epsilon = torch.randn_like(sigma)

    z_reparameterized = mu + torch.multiply(sigma, epsilon) 
    images = vae.decode(z_reparameterized)

    images = images.cpu().detach()
    print(torch.min(images))
    print(torch.max(images))
    images = (images - torch.min(images)) / (torch.max(images) - torch.min(images)) * 255
    images = images.to(torch.int16)
    print(torch.min(images))
    print(torch.max(images))

    grid = torchvision.utils.make_grid(images)
    grid = grid.numpy()
    grid = np.transpose(grid, (1, 2, 0))
    plt.imshow(grid)
    plt.savefig("test.jpg")
    exit()
