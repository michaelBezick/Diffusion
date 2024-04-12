import statistics

import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from LDM_Classes import LDM, VAE, AttentionUNet, expand_output, load_FOM_model

checkpoint_path_LDM = "./logs/LDM/version_1/checkpoints/epoch=3552-step=106590.ckpt"
experiment_name = "Experiment_3"

num_samples = 20_000
batch_size = 1000
plot = True
mean = 1.8
variance = 0.2
############################################################

dataset = torch.load(
    "./Generated_Datasets/" + experiment_name + "/generated_dataset.pt"
)

FOM_conditioning_values = torch.randn(100)  # to avoid linter errors

if plot:
    FOM_conditioning_values = torch.load(
        "./Generated_Datasets/" + experiment_name + "/FOM_values.pt"
    )
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

with open(
    "./Generated_Datasets/" + experiment_name + "/Experiment_Summary.txt", "w"
) as file:
    file.write(f"Max generated: {max(FOM_measurements):2f}\n")
    file.write(f"Max generated: {max(FOM_measurements):2f}\n")
    file.write(f"Mean generated: {statistics.mean(FOM_measurements):2f}\n")
    file.write(f"Min generated: {min(FOM_measurements):2f}\n")

    dataset_labels = torch.load("./Files/FOM_labels.pt")
    file.write(f"Max dataset: {max(dataset_labels):2f}\n")
    file.write(f"Mean dataset: {statistics.mean(dataset_labels):2f}\n")
    file.write(f"Min dataset: {min(dataset_labels):2f}\n")

    max_generated = max(FOM_measurements)
    max_dataset = max(dataset_labels)
    percent_improvement = max_generated / max_dataset

    file.write(f"Percent improvement: {percent_improvement}\n")

# plotting correlation

if plot:
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
