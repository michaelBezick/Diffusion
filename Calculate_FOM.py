import statistics

import matplotlib.pyplot as plt
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from LDM_Classes import LDM, VAE, AttentionUNet, expand_output, load_FOM_model

checkpoint_path_LDM = "./logs/LDM/version_2/checkpoints/epoch=6999-step=210000.ckpt"
experiment_name = "Experiment_8"

num_samples = 20_000
batch_size = 1000
plot = True
mean = 1.8
variance = 0.2
############################################################


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


dataset = torch.load(
    "./Generated_Datasets/" + experiment_name + "/generated_dataset.pt"
)

FOM_conditioning_values = torch.randn(100)  # to avoid linter errors

if plot:
    FOM_conditioning_values = torch.load(
        "./Generated_Datasets/" + experiment_name + "/FOM_values.pt"
    )
    print(FOM_conditioning_values.size())
dataset = expand_output(dataset, num_samples)

FOM_calculator = load_FOM_model("Files/VGGnet.json", "Files/VGGnet_weights.h5")

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
)

FOM_measurements = []

i = 0
for batch in tqdm(train_loader):
    if i == 0:
        save_image_grid(batch, "Diffusion_Grid.png")
        grid = torchvision.utils.make_grid(batch)
        i = 1
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
    # num_duplicates = num_samples // batch_size
    # FOM_conditioning_values = FOM_conditioning_values.unsqueeze(0)
    # FOM_conditioning_values = FOM_conditioning_values.expand(num_duplicates, -1)
    # FOM_conditioning_values = FOM_conditioning_values.reshape(-1)
    print(FOM_conditioning_values.size())
    plt.figure()
    plt.scatter(FOM_conditioning_values.cpu().numpy(), FOM_measurements)
    plt.title("Conditioning FOM Values versus Generated FOM Values")
    plt.xlabel("Conditioning FOM Values")
    plt.ylabel("Generated FOM Values")
    plt.savefig("./Generated_Datasets/" + experiment_name + "/Correlation.jpg", dpi=300)
