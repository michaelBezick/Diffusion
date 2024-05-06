import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader, Dataset
import torchvision

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from LDM_Classes import LDM, VAE, AttentionUNet

dataset = np.expand_dims(np.load("../Files/TPV_dataset.npy"), 1)
normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

normalizedDataset = normalizedDataset * 2 - 1

normalizedDataset = normalizedDataset.astype(np.float32)

dataset = torch.from_numpy(normalizedDataset)


def save_image_grid(tensor, filename, nrow=6, padding=2):
    # Make a grid from batch tensor
    grid_image = torchvision.utils.make_grid(
        tensor, nrow=nrow, padding=padding, normalize=True
    )

    # Convert to numpy array and then to PIL image
    grid_image = (
        grid_image.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    )
    pil_image = Image.fromarray(grid_image)

    # Save as PNG
    pil_image.save(filename, bitmap_format="png")


class LabeledDataset(Dataset):
    def __init__(self, images, labels, size=32, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image[:, 0 : self.size, 0 : self.size], label


def make_beta_schedule(num_steps, beta_1, beta_T):
    return np.linspace(beta_1, beta_T, num=num_steps)


def get_alpha_schedule(beta_schedule):
    return np.ones_like(beta_schedule) - beta_schedule


def calculate_alpha_bar(alpha_schedule):
    return np.cumprod(alpha_schedule)


num_steps = 1000
device = "cuda"

in_channels = 1
batch_size = 100
height = 8
width = 8

checkpoint_path_LDM = (
    "../logs/LDM/fixed_conditioning/checkpoints/epoch=3568-step=107070.ckpt"
)

ddpm = AttentionUNet(UNet_channel=64, batch_size=batch_size).to("cuda")
vae = VAE(h_dim=128, batch_size=batch_size).to("cuda")
ldm = (
    LDM.load_from_checkpoint(
        checkpoint_path_LDM,
        DDPM=ddpm,
        VAE=vae,
        in_channels=1,
        batch_size=batch_size,
        num_steps=1000,
        latent_height=8,
        latent_width=8,
        lr=1e-3,
    )
    .to("cuda")
    .eval()
)

random_generator = MultivariateNormal(
    torch.zeros(height * width * in_channels, device=device),
    torch.eye(height * width * in_channels, device=device),
)

labels = torch.load("../Files/FOM_labels.pt")
labeled_dataset = LabeledDataset(dataset, labels)

train_loader = DataLoader(
    labeled_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

data = None
images = None

for data in train_loader:
    images, FOMs = data

    break

for t in [0, 100, 300, 600, 999]:
    images = images.cuda()
    beta_schedule = make_beta_schedule(num_steps, 1e-5, 0.02)
    alpha_schedule = get_alpha_schedule(beta_schedule)
    alpha_bar_schedule = calculate_alpha_bar(alpha_schedule)
 
    epsilon_sample = random_generator.sample((batch_size,))
    epsilon_0 = epsilon_sample.view(batch_size, in_channels, height, width)
    alpha_bar_vector = alpha_bar_schedule[t]
    alpha_bar_vector = torch.tensor(alpha_bar_vector).float().unsqueeze(-1).cuda()
    # alpha_bar_vector = torch.from_numpy(alpha_bar_vector).float()
    alpha_bar_vector = alpha_bar_vector.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    print(alpha_bar_vector.size())
    alpha_bar_vector = alpha_bar_vector.expand(-1, in_channels, height, width)

    with torch.no_grad():
        mu, sigma = ldm.VAE.encode(images)

    z_reparameterized = mu + torch.multiply(
        sigma, torch.randn_like(sigma, device=device)
    )
    mu = torch.mul(torch.sqrt(alpha_bar_vector), z_reparameterized)
    variance = torch.mul(
        torch.sqrt(torch.ones_like(alpha_bar_vector) - alpha_bar_vector), epsilon_0
    )
    x_t = torch.add(mu, variance)
    f_string = f"{t}_noise.png"
    f_string2 = f"{t}_original.png"
    save_image_grid(x_t[0, :, :, :], f_string)
    save_image_grid(z_reparameterized[0, :, :, :], f_string2)
