import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import Dataset, dataloader
from torch.utils.data import DataLoader
import torchvision
from Models import Generator
from PIL import Image
from tqdm import tqdm
from Models import cVAE

"""
NEW STRATEGY: Latent point generated through randomly choosing from dataset
"""

"I need to investigate what ^ means."

num_samples = 20_000


clamp = True
RGB = False
batch_size = 1000
lr = 1e-3
perceptual_loss_scale = 1
kl_divergence_scale = 0.1
checkpoint_path = "./logs/cVAE/version_0/checkpoints/epoch=1098-step=32970.ckpt" 
vae = cVAE.load_from_checkpoint(checkpoint_path,n_channels=1, h_dim=128, batch_size=batch_size, lr=lr, kl_divergence_scale=kl_divergence_scale, perceptual_loss_scale=perceptual_loss_scale)
vae.eval()

def clamp_output(tensor: torch.Tensor, threshold):
    return torch.where(tensor > threshold, torch.tensor(1.0), torch.tensor(0.0))

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



plot = True
mean = 1.8
variance = 0.1

FOM_calculator = load_FOM_model("../Files/VGGnet.json", "../Files/VGGnet_weights.h5")

labels = torch.load("../Files/FOM_labels.pt")

labeled_dataset = LabeledDataset(dataset, labels)

train_loader = DataLoader(
    labeled_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

labels_list = []
FOMs_list = []

vae = vae.cuda()

dataset = []
with torch.no_grad():
    i = 0
    loop = True
    while loop:
        for batch in train_loader:
            print(i)
            if (i >= num_samples // batch_size) :
                loop = False
                break
            i += 1
            images, FOMs = batch
            FOMs = FOMs.unsqueeze(1)
            images = images.cuda().float()
            FOMs = FOMs.cuda().float()

            labels = variance * torch.randn((batch_size, 1), device="cuda") + mean

            labels1 = vae.FOM_Conditioner(labels).view(-1, 1, 32, 32)
            labels2 = vae.FOM_Conditioner_latent(labels).view(-1, 1, 8, 8)

            FOMs_projection = vae.FOM_Conditioner(FOMs).view(-1, 1, 32, 32)

            mu, sigma = vae.encode(torch.cat([images, labels1], dim=1)) #experiment with labels or FOMs

            epsilon = torch.randn_like(sigma)
            z_reparameterized = mu + torch.multiply(sigma, epsilon)

            labels_list.extend(labels.cpu().detach().numpy())

            cat = torch.cat([z_reparameterized, labels2], dim = 1)
            images = vae.decode(cat)
            dataset.extend(images.detach().cpu().numpy())
            images = expand_output(images, batch_size)
            if RGB:
                images = images * 255
            if clamp:
                images = clamp_output(images, 0.5)
            if i == 1:
                save_image_grid(images, "cVAE_Sample.png")
            FOMs = FOM_calculator(
                torch.permute(images.repeat(1, 3, 1, 1), (0, 2, 3, 1)).cpu().numpy()
            )

            FOMs_list.extend(FOMs.numpy().flatten().tolist())


dataset = torch.from_numpy(np.array(dataset))
dataset = (dataset - torch.min(dataset)) / (torch.max(dataset) - torch.min(dataset))
dataset = dataset.to(torch.half)
dataset = dataset.numpy()
np.save("cVAE_dataset.npy", dataset)

plt.figure()
plt.scatter(labels_list, FOMs_list)
plt.title("cVAE conditioning FOM values versus generated FOM values")
plt.xlabel("Conditioning FOM values")
plt.ylabel("Generated FOM values")
plt.plot()
plt.savefig("cVAE_Scatter.png", dpi=300)

with open("cVAE_Evaluation.txt", "w") as file:
    file.write(f"FOM max: {max(FOMs_list)}\n")
    file.write(f"FOM mean: {sum(FOMs_list) / len(FOMs_list)}")
