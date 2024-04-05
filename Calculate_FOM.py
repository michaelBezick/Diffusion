import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import statistics
import matplotlib.pyplot as plt
from tqdm import tqdm

from LDM_Classes import LDM, VAE, AttentionUNet, load_FOM_model, expand_output

checkpoint_path_LDM = "./logs/LDM/version_1/checkpoints/epoch=3552-step=106590.ckpt"
experiment_name = "Experiment_3"

num_samples = 20_000
batch_size = 1000
generate_new_dataset = False
mean = 1.8
variance = 0.2
############################################################

if generate_new_dataset:
    FOM_conditioning_values = variance * torch.randn(batch_size) + mean
    FOM_conditioning_values = FOM_conditioning_values.float()
    FOM_conditioning_values = FOM_conditioning_values.to("cuda")
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

    dataset = ldm.create_dataset(num_samples=num_samples, FOM_conditioning_values=FOM_conditioning_values)

    dataset = torch.from_numpy(dataset)
    torch.save(dataset, "generated_dataset.pt")
    dataset = expand_output(dataset, num_samples)
else:
    dataset = torch.load("./Generated_Datasets/" + experiment_name + "/generated_dataset.pt")
    FOM_conditioning_values = torch.load("./Generated_Datasets/" + experiment_name + "/FOM_values.pt")
    dataset = expand_output(dataset, num_samples)

FOM_calculator = load_FOM_model("Files/VGGnet.json", "Files/VGGnet_weights.h5")

train_loader = DataLoader(
   dataset,
   batch_size=batch_size,
   shuffle=False,
   drop_last=False,
)

FOM_measurements = []

for batch in tqdm(train_loader):
   grid = torchvision.utils.make_grid(batch)
   FOM = FOM_calculator(torch.permute(batch.repeat(1, 3, 1, 1), (0, 2, 3, 1)).numpy())
   FOM_measurements.extend(FOM.numpy().flatten().tolist())

print(len(FOM_measurements))
print(f"Max generated: {max(FOM_measurements):2f}")
print(f"Mean generated: {statistics.mean(FOM_measurements):2f}")
print(f"Min generated: {min(FOM_measurements):2f}")

dataset_labels = torch.load("./Files/FOM_labels.pt")
print(f"Max dataset: {max(dataset_labels):2f}")
print(f"Mean dataset: {statistics.mean(dataset_labels):2f}")
print(f"Min dataset: {min(dataset_labels):2f}")

max_generated =  max(FOM_measurements)
max_dataset = max(dataset_labels)
percent_improvement = max_generated / max_dataset

print(f"Percent improvement: {percent_improvement}")

#plotting correlation
num_duplicates = num_samples // batch_size

FOM_conditioning_values = FOM_conditioning_values.unsqueeze(0)
FOM_conditioning_values = FOM_conditioning_values.expand(num_duplicates, -1)
FOM_conditioning_values = FOM_conditioning_values.reshape(-1)

plt.figure()
plt.scatter(FOM_conditioning_values.cpu().numpy(), FOM_measurements)
plt.title("Conditioning FOM Values versus Generated FOM Values")
plt.xlabel("Conditioning FOM Values")
plt.ylabel("Generated FOM Values")
plt.savefig("./Generated_Datasets/" + experiment_name + "/Correlation.jpg", dpi=300)
