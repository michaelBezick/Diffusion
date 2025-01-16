import os

import numpy as np
import torch

from LDM_Ablation_Classes import VAE, Ablation_LDM, AblationAttentionUNet

experiment_name = "Experiment_1"
experiment_notes = "No conditioning"
num_samples = 20_000
batch_size = 5000

checkpoint_path_LDM = "./logs/LDM/version_1/checkpoints/epoch=9999-step=1200000.ckpt"
############################################################

FOM_values = None

ddpm = AblationAttentionUNet(UNet_channel=64, batch_size=batch_size).to("cuda")
vae = VAE(h_dim=128, batch_size=batch_size).to("cuda")
ldm = (
    Ablation_LDM.load_from_checkpoint(
        checkpoint_path_LDM,
        DDPM=ddpm,
        vae=vae,
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

dataset = []

for i in range(num_samples // batch_size):
    samples = ldm.create_dataset(num_samples=batch_size)
    dataset.extend(samples)

dataset = torch.from_numpy(np.array(dataset))
dir_path = "./Generated_Datasets/" + experiment_name + "/"

if not os.path.isdir(dir_path):
    os.mkdir(dir_path)

torch.save(dataset, dir_path + "generated_dataset.pt")

with open(dir_path + "Experiment_Notes.txt", "w") as file:
    file.write(experiment_notes)
