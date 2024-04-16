import torchvision
from PIL import Image
import torch
import numpy as np

def save_image_grid(tensor, filename, nrow=6, padding=2):
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

dataset = np.load("../Files/TPV_dataset.npy")

normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

normalizedDataset = np.multiply(normalizedDataset, 2) - 1

normalizedDataset = normalizedDataset.astype(np.float32)

dataset = torch.from_numpy(normalizedDataset)
dataset = dataset.unsqueeze(1)

dataset = dataset[0:36, :, :, :]
save_image_grid(dataset, "original_grid.png")
