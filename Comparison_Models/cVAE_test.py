import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import tensorflow as tf
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from Models import cVAE

batch_size = 100
lr = 1e-3
perceptual_loss_scale = 1
kl_divergence_scale = 0.1
checkpoint_path = "./logs/cVAE/version_0/checkpoints/epoch=1098-step=32970.ckpt" 
vae = cVAE.load_from_checkpoint(checkpoint_path,n_channels=1, h_dim=128, batch_size=batch_size, lr=lr, kl_divergence_scale=kl_divergence_scale, perceptual_loss_scale=perceptual_loss_scale)
vae.eval()

dataset = np.expand_dims(np.load("../Files/TPV_dataset.npy"), 1)
normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

normalizedDataset = np.multiply(normalizedDataset, 2) - 1

normalizedDataset = normalizedDataset.astype(np.float32)

dataset = torch.from_numpy(normalizedDataset)

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
def save_image_grid(tensor, filename, nrow=8, padding=2):
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

def load_FOM_model(model_path, weights_path):
    with open(model_path, "r") as file:
        data = file.read()

    FOM_calculator = tf.keras.models.model_from_json(data)
    FOM_calculator.load_weights(weights_path)

    return FOM_calculator


def expand_output(tensor: torch.Tensor, num_samples):
    x = torch.zeros([num_samples, 1, 64, 64])
    x[:, :, 0:32, 0:32] = tensor
    x[:, :, 32:64, 0:32] = torch.flip(tensor, dims=[2])
    x[:, :, 0:32, 32:64] = torch.flip(tensor, dims=[3])
    x[:, :, 32:64, 32:64] = torch.flip(tensor, dims=[2, 3])

    return x


num_samples = 20_000

plot = True
mean = 1.8
variance = 0.1

FOM_calculator = load_FOM_model("../Files/VGGnet.json", "../Files/VGGnet_weights.h5")

labels_list = []
FOMs_list = []

vae = vae.cuda()

labels = torch.load("../Files/FOM_labels.pt")

labeled_dataset = LabeledDataset(dataset, labels)

train_loader = DataLoader(
    labeled_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

with torch.no_grad():
    for batch in train_loader:
        images, FOMs = batch
        images = images.cuda()
        FOMs = FOMs.cuda()
        save_image_grid(expand_output(images, batch_size), "cVAE_test/original.png")
        FOMs = FOMs.unsqueeze(1)
        images = images.float()
        FOMs = FOMs.float()
        FOMs_before = FOMs


        FOMs = vae.FOM_Conditioner(FOMs_before)
        FOMs_latent = vae.FOM_Conditioner_latent(FOMs_before)
        FOMs = FOMs.view(-1, 1, 32, 32)
        FOMs_latent = FOMs_latent.view(-1, 1, 8, 8)

        x = torch.cat([images, FOMs], dim = 1)

        mu, sigma = vae.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparameterized = mu + torch.multiply(sigma, epsilon)

        z_reparameterized = torch.cat([z_reparameterized, FOMs_latent], dim=1)

        x_hat = vae.decode(z_reparameterized)
        save_image_grid(expand_output(x_hat, batch_size), "cVAE_test/generated.png")

        #sample test now
        labels = variance * torch.randn((batch_size, 1), device="cuda") + mean
        noise = torch.randn((batch_size, 1, 8, 8), device="cuda")
        proj = vae.FOM_Conditioner_latent(labels).view(-1, 1, 8, 8)
        cat = torch.cat([noise, proj], dim = 1)
        images = vae.decode(cat)
        images = expand_output(images, batch_size)
        save_image_grid(images, "cVAE_test/sampled.png")
