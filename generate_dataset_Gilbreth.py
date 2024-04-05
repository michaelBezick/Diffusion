import torch
import os

from LDM_Classes import LDM, VAE, AttentionUNet

experiment_name = ""
experiment_notes = ""
num_samples = 20_000
batch_size = 1000
mean = 1.7
variance = 0.1

checkpoint_path_LDM = "./logs/LDM/version_1/checkpoints/epoch=3552-step=106590.ckpt"
############################################################

FOM_values = variance * torch.randn(batch_size) + mean
FOM_values = FOM_values.float()
FOM_values = FOM_values.to("cuda")
ddpm = AttentionUNet(UNet_channel=64, batch_size=batch_size).to("cuda")
vae = VAE(h_dim=128, batch_size=batch_size).to("cuda")
ldm = LDM.load_from_checkpoint(
    checkpoint_path_LDM,
    DDPM=ddpm,
    VAE=vae,
    in_channels=1,
    batch_size=batch_size,
    num_steps=1000,
    latent_height=8,
    latent_width=8,
    lr=1e-3,
).to("cuda").eval()
torch.set_float32_matmul_precision("high")

dataset = ldm.create_dataset(num_samples=num_samples, FOM_values=FOM_values)
dataset = ldm.create_dataset(num_samples=num_samples, FOM_values=FOM_values)
dataset = torch.from_numpy(dataset)

dir_path = "./Generated_Datasets/" + experiment_name + "/"

if not os.path.isdir(dir_path):
    os.mkdir(dir_path)

torch.save(dataset, dir_path + "generated_dataset.pt")
torch.save(FOM_values, dir_path + "FOM_values")
with open("Experiment_Notes.txt", "w") as file:
    file.write(experiment_notes)
