"""
New labeled dataset will be only one quadrant.
Need to update VAE and DDPM
"""

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader

def clamp_output(tensor: torch.Tensor, threshold):
    return torch.where(tensor > threshold, torch.tensor(1.0), torch.tensor(0.0))

def load_FOM_model(model_path, weights_path):
    with open(model_path, "r") as file:
        data = file.read()

    FOM_calculator = tf.keras.models.model_from_json(data)
    FOM_calculator.load_weights(weights_path)

    return FOM_calculator


def expand_output(tensor: torch.Tensor):
    x = torch.zeros([100, 1, 64, 64])
    x[:, :, 0:32, 0:32] = tensor
    x[:, :, 32:64, 0:32] = torch.flip(tensor, dims=[2])
    x[:, :, 0:32, 32:64] = torch.flip(tensor, dims=[3])
    x[:, :, 32:64, 32:64] = torch.flip(tensor, dims=[2, 3])

    return x


dataset = np.load("./TPV_dataset.npy")

normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

# normalizedDataset = np.multiply(normalizedDataset, 2) - 1

normalizedDataset = normalizedDataset.astype(np.float32)

print(np.max(normalizedDataset))
print(np.min(normalizedDataset))

torch_dataset = torch.from_numpy(normalizedDataset)
torch_dataset = clamp_output(torch_dataset, 0.5)
batch_size = 100

train_loader = DataLoader(
    torch_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
)

FOM_measurements = []
FOM_calculator = load_FOM_model("VGGnet.json", "VGGnet_weights.h5")

for batch in train_loader:
    batch = batch.unsqueeze(1)
    FOM = FOM_calculator(torch.permute(batch.repeat(1, 3, 1, 1), (0, 2, 3, 1)).numpy())
    FOM_measurements.extend(FOM.numpy().flatten().tolist())

print(len(FOM_measurements))
print(max(FOM_measurements))
print(min(FOM_measurements))
print(sum(FOM_measurements) / len(FOM_measurements))

torch.save(FOM_measurements, "FOM_labels_new.pt")
