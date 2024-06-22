import torchvision
import numpy as np

import torch
from PIL import Image


def expand_output(tensor: torch.Tensor, num_samples):
    x = torch.zeros([num_samples, 1, 64, 64])
    x[:, :, 0:32, 0:32] = tensor
    x[:, :, 32:64, 0:32] = torch.flip(tensor, dims=[2])
    x[:, :, 0:32, 32:64] = torch.flip(tensor, dims=[3])
    x[:, :, 32:64, 32:64] = torch.flip(tensor, dims=[2, 3])

    return x
def save_image_grid(tensor, filename, nrow=4, padding=2):
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

def total_save(load_path, save_path):
    dataset = np.load(load_path)
    dataset = torch.from_numpy(dataset).float()
    batch = expand_output(dataset[0:25, :, :, :], 25)
    save_image_grid(batch, save_path)

total_save("./AAE_dataset.npy", "AAE_sample.png")
total_save("./WGAN_dataset.npy", "WGAN_sample.png")
total_save("./cVAE_dataset.npy", "cVAE_Sample.png")

ldm_dataset = torch.load("../Generated_Datasets/Experiment_11/generated_dataset.pt")
batch = expand_output(ldm_dataset[0:25, :, :, :], 25)
save_image_grid(batch, "LDM_sample.png")
