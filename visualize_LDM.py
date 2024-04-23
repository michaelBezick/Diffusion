import numpy as np
import matplotlib.pyplot as plt
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

resume_from_checkpoint = True

checkpoint_path_VAE = "./logs/VAE/version_0/checkpoints/epoch=1115-step=33480.ckpt"

checkpoint_path_LDM = "./logs/LDM/version_1/checkpoints/epoch=3552-step=106590.ckpt"

def expand_output_latent(tensor: torch.Tensor, num_samples):
    print(tensor.size())
    x = torch.zeros([num_samples, 1, 16, 16])
    x[:, :, 0:8, 0:8] = tensor
    x[:, :, 8:16, 0:8] = torch.flip(tensor, dims=[2])
    x[:, :, 0:8, 8:16] = torch.flip(tensor, dims=[3])
    x[:, :, 8:16, 8:16] = torch.flip(tensor, dims=[2, 3])

    return x

def save_image_grid_latent(tensor: torch.Tensor, filename):
    grid_image = (
        tensor.permute(1, 2, 0).expand(16, 16, 3).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    )
    plt.imshow(grid_image)
    plt.axis("off")
    plt.savefig(filename, dpi=300, bbox_inches='tight')

def expand_output(tensor: torch.Tensor, num_samples):
    x = torch.zeros([num_samples, 1, 64, 64])
    x[:, :, 0:32, 0:32] = tensor
    x[:, :, 32:64, 0:32] = torch.flip(tensor, dims=[2])
    x[:, :, 0:32, 32:64] = torch.flip(tensor, dims=[3])
    x[:, :, 32:64, 32:64] = torch.flip(tensor, dims=[2, 3])

    return x
def save_image_grid(tensor: torch.Tensor, filename):
    grid_image = (
        tensor.permute(1, 2, 0).expand(64, 64, 3).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    )
    plt.imshow(grid_image)
    plt.axis("off")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
###########################################################################################

checkpoint_callback = ModelCheckpoint(filename="good", every_n_train_steps=300)

torch.set_float32_matmul_precision("high")

vae = VAE.load_from_checkpoint(
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

DDPM = DDPM.cuda()
vae = vae.cuda()

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

ldm = ldm.cuda()

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
for batch in train_loader:
    images, FOMs = batch
    images = images.cuda()
    FOMs = FOMs.cuda()

    save_image_grid(expand_output(images[0, :, :, :].unsqueeze(0), 1).squeeze(0), "LDM_original_image.png")

# random timestep
    t = torch.randint(0, ldm.num_steps - 1, (ldm.batch_size,), device="cpu")
    t[0] = 200

# generating epsilon_0
    epsilon_sample = ldm.random_generator.sample((ldm.batch_size,))
    epsilon_0 = epsilon_sample.view(
        ldm.batch_size, ldm.in_channels, ldm.height, ldm.width
    )

    print(epsilon_0.size())
    print(epsilon_0[0, :, :, :].unsqueeze(0).size())
    save_image_grid_latent(expand_output_latent(epsilon_0[0, :, :, :].unsqueeze(0), 1).squeeze(0), "original_noise.png")

# broadcasting alpha_bar_vector
    alpha_bar_vector = ldm.alpha_bar_schedule[t].float()
    alpha_bar_vector = alpha_bar_vector.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    alpha_bar_vector = alpha_bar_vector.expand(
        -1, ldm.in_channels, ldm.height, ldm.width
    ).cuda()

# encoding to latent space
    x = images
    with torch.no_grad():
        mu, sigma = ldm.VAE.encode(x)

    z_reparameterized = mu + torch.multiply(
        sigma, torch.randn_like(sigma, device=ldm.device)
    )

    save_image_grid_latent(expand_output_latent(z_reparameterized[0, :, :, :].unsqueeze(0), 1).squeeze(0), "LDM_x0.png")


### Creation of x_t, algorithm 1 of Ho et al.###
    mu = torch.mul(torch.sqrt(alpha_bar_vector), z_reparameterized)
    epsilon_0 = epsilon_0.cuda()
    variance = torch.mul(
        torch.sqrt(torch.ones_like(alpha_bar_vector, device="cuda") - alpha_bar_vector), epsilon_0
    )
    x_t = torch.add(mu, variance)

    save_image_grid_latent(expand_output_latent(x_t[0, :, :, :].unsqueeze(0), 1).squeeze(0), "LDM_xt.png")

# experiment with this
    tNotNormalized = t.float().cuda()
    FOMs = FOMs.float().cuda()
    x_t = x_t.cuda()

# predicted true epsilon in latent space
    epsilon_theta_latent = ldm.DDPM.forward(x_t, FOMs, tNotNormalized)
    save_image_grid_latent(expand_output_latent(epsilon_theta_latent[0, :, :, :].unsqueeze(0), 1).squeeze(0), "Predicted_noise.png")
    exit()
