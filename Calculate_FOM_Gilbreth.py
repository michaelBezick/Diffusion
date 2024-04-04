import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from LDM_Classes import LDM, VAE, AttentionUNet

checkpoint_path_LDM = "./logs/LDM/version_1/checkpoints/epoch=3552-step=106590.ckpt"

num_samples = 20_000
batch_size = 1000
generate_new_dataset = True
compiled = False
mean = 1.7
variance = 0.1
############################################################

if generate_new_dataset:
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

    #FOM_calculator = load_FOM_model("Files/VGGnet.json", "Files/VGGnet_weights.h5")

    dataset = ldm.create_dataset(num_samples=num_samples, FOM_values=FOM_values)
    if compiled:
        optimized_create_dataset = torch.compile(ldm.create_dataset, mode="reduce-overhead")
        print("Compiled")
        dataset = optimized_create_dataset(num_samples=num_samples, FOM_values=FOM_values)
    else:
        dataset = ldm.create_dataset(num_samples=num_samples, FOM_values=FOM_values)

    dataset = torch.from_numpy(dataset)
    torch.save(dataset, "generated_dataset.pt")
    exit()
    dataset = expand_output(dataset, num_samples)
else:
    dataset = torch.load("generated_dataset.pt")
    dataset = expand_output(dataset, num_samples)
#
#train_loader = DataLoader(
#    dataset,
#    batch_size=batch_size,
#    shuffle=False,
#    drop_last=False,
#)
#
#FOM_measurements = []
#
#for batch in train_loader:
#    grid = torchvision.utils.make_grid(batch)
#    FOM = FOM_calculator(torch.permute(batch.repeat(1, 3, 1, 1), (0, 2, 3, 1)).numpy())
#    FOM_measurements.extend(FOM.numpy().flatten().tolist())
#
#print(len(FOM_measurements))
#print(max(FOM_measurements))
