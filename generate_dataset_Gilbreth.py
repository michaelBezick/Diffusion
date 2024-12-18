import torch
import os
import numpy as np
import time

from LDM_Classes import LDM, VAE, AttentionUNet

experiment_name = "Experiment_12"
experiment_notes = "Fixed conditioning mean 1.8, variance 0.1."
num_samples = 20_000
batch_size = 5000
mean = 1.8
variance = 0.1
variable_conditioning = False

checkpoint_path_LDM = "./logs/LDM/version_6/checkpoints/epoch=6989-step=209700.ckpt"
############################################################

FOM_values = None
if variable_conditioning == False:
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

dataset = []
FOM_values_list = []
if variable_conditioning == False:
    for i in range(num_samples // batch_size):
        FOM_values = variance * torch.randn(batch_size) + mean
        FOM_values_list.extend(FOM_values.numpy())
        FOM_values = FOM_values.float()
        FOM_values = FOM_values.to("cuda")
        initial = time.time()
        samples = ldm.create_dataset(num_samples=batch_size, FOM_values=FOM_values)
        final = time.time()
        print(final - initial)
        print((final - initial) / 60)
        print(np.shape(samples))
        exit()
        dataset.extend(samples)
else:
    samples = ldm.create_dataset_variable_FOM(num_samples=num_samples, start_mean=1.4, end_mean=1.8, variance=0.1)

dataset = torch.from_numpy(np.array(dataset))
FOM_values_list = torch.from_numpy(np.array(FOM_values_list))
print(FOM_values_list.size())

dir_path = "./Generated_Datasets/" + experiment_name + "/"

if not os.path.isdir(dir_path):
    os.mkdir(dir_path)

torch.save(dataset, dir_path + "generated_dataset.pt")
if variable_conditioning == False:
    torch.save(FOM_values_list, dir_path + "FOM_values.pt")
with open(dir_path + "Experiment_Notes.txt", "w") as file:
    file.write(experiment_notes)
