import os
import sys

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
import numpy as np

parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from LDM_Classes import LDM, VAE, AttentionUNet

dataset = np.expand_dims(np.load("../Files/TPV_dataset.npy"), 1)
normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

normalizedDataset = np.multiply(normalizedDataset, 2) - 1

normalizedDataset = normalizedDataset.astype(np.float32)

dataset = torch.from_numpy(normalizedDataset)


def mean_normalize(x: torch.Tensor):
    return (x - torch.min(x)) / (torch.max(x) - torch.min(x))


batch_size = 1
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
torch.set_float32_matmul_precision("high")

FOM_labels = torch.randn((batch_size), device="cuda") * 0.1 + 1.8
FOM_labels = (FOM_labels - 0.713) * 2000
positional_encodings = ldm.DDPM.FOM_embedder(FOM_labels)
first = ldm.DDPM.FOM_embedder1(positional_encodings)
first = mean_normalize(first)
first = first.squeeze().detach().cpu().numpy()
second = ldm.DDPM.FOM_embedder2(positional_encodings)
second = mean_normalize(second)
second = second.squeeze().detach().cpu().numpy()
third = ldm.DDPM.FOM_embedder3(positional_encodings)
third = mean_normalize(third)
third = third.squeeze().detach().cpu().numpy()

# plotting 8x8
plt.figure()
cax = plt.imshow(first, interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig("8x8.png", bbox_inches="tight", dpi=300)
# plotting 4x4
plt.figure()
cax = plt.imshow(second, interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig("4x4.png", bbox_inches="tight", dpi=300)
# plotting 2x2
plt.figure()
cax = plt.imshow(third, interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.savefig("2x2.png", bbox_inches="tight", dpi=300)
