from tensorflow import data
import matplotlib.pyplot as plt
import cv2
from LDM_Classes import VAE
import torch
import torchvision
import numpy as np
from PIL import Image
import random

num_devices = 2
num_nodes = 2
num_workers = 1
accelerator = "gpu"
batch_size = 100
epochs = 10_000
in_channels = 1
out_channels = 1

checkpoint_path_VAE = "./logs/VAE/version_0/checkpoints/epoch=1115-step=33480.ckpt"

checkpoint_path_LDM = "./logs/LDM/version_1/checkpoints/epoch=3552-step=106590.ckpt"

# terms VAE
lr_VAE = 1e-3
lr_DDPM = 1e-3
perceptual_loss_scale = 1
kl_divergence_scale = 0.3
dataset = np.expand_dims(np.load("./Files/TPV_dataset.npy"), 1)
normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

normalizedDataset = normalizedDataset * 2 - 1

normalizedDataset = normalizedDataset.astype(np.float32)

dataset = torch.from_numpy(normalizedDataset)

vae = VAE(
    in_channels=in_channels,
    h_dim=128,
    lr=lr_VAE,
    batch_size=batch_size,
    perceptual_loss_scale=perceptual_loss_scale,
    kl_divergence_scale=kl_divergence_scale,
)

vae = VAE.load_from_checkpoint(
    checkpoint_path=checkpoint_path_VAE,
    in_channels=in_channels,
    h_dim=128,
    lr=lr_VAE,
    batch_size=batch_size,
    perceptual_loss_scale=perceptual_loss_scale,
    kl_divergence_scale=kl_divergence_scale,
)
vae = vae.to("cuda")
vae.eval()

def expand_output(tensor: torch.Tensor, num_samples):
    x = torch.zeros([num_samples, 1, 16, 16])
    x[:, :, 0:8, 0:8] = tensor
    x[:, :, 8:16, 0:8] = torch.flip(tensor, dims=[2])
    x[:, :, 0:8, 8:16] = torch.flip(tensor, dims=[3])
    x[:, :, 8:16, 8:16] = torch.flip(tensor, dims=[2, 3])

    return x

def expand_output_2(tensor: torch.Tensor, num_samples):
    x = torch.zeros([num_samples, 1, 64, 64])
    x[:, :, 0:32, 0:32] = tensor
    x[:, :, 32:64, 0:32] = torch.flip(tensor, dims=[2])
    x[:, :, 0:32, 32:64] = torch.flip(tensor, dims=[3])
    x[:, :, 32:64, 32:64] = torch.flip(tensor, dims=[2, 3])

    return x

def save_image_grid(tensor: torch.Tensor, filename, nrow=8, padding=2, grid=False):
    # Make a grid from batch tensor

    print(tensor.size())

    if (grid):
        grid_image = torchvision.utils.make_grid(
                tensor, nrow=nrow, padding=padding, normalize=True
        )
        grid_image = (
            grid_image.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        )
    else:
        grid_image = (
            tensor.permute(1, 2, 0).expand(64, 64, 3).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        )

    plt.imshow(grid_image)
    plt.axis("off")
    plt.savefig(filename, dpi=300)
    # plt.plot()
    # pil_image = Image.fromarray(grid_image)
    #
    # # Save as PNG
    # pil_image.save(filename, bitmap_format="png", optimize=False, dpi=(1000,1000), compress_level=0)

def save_image_grid_2(tensor: torch.Tensor, filename, nrow=8, padding=2, grid=False):
    # Make a grid from batch tensor

    tensor = tensor.squeeze(1)
    print(tensor.size())

    if (grid):
        grid_image = torchvision.utils.make_grid(
                tensor, nrow=nrow, padding=padding, normalize=True
        )
        grid_image = (
            grid_image.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        )
    else:
        grid_image = (
            tensor.permute(1, 2, 0).expand(16, 16, 3).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        )

    plt.imshow(grid_image)
    plt.axis("off")
    plt.savefig(filename, dpi=300)

index = random.randint(0, 10_000)
save_image_grid(dataset[index, :, :, :], "sample_from_dataset.png")
mu, sigma = vae.encode(dataset[index, :, 0:32, 0:32].cuda().unsqueeze(0))
z = torch.randn(1, device="cuda") * sigma + mu
decoded = vae.decode(z.cuda())
print(decoded.size())
z = expand_output(z, 1)
decoded = expand_output_2(decoded, 1).squeeze(1)
save_image_grid_2(z, "latent.png")
save_image_grid(decoded, "decoded.png")
