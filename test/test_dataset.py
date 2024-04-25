import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

dataset = np.load("../Files/TPV_dataset.npy")
dataset = torch.from_numpy(dataset)
print(dataset.size())
clamp = True
normalize = True
batch_size = 100


def mean_normalize(tensor: torch.Tensor):
    return (tensor - torch.min(tensor)) / (torch.max(tensor) - torch.min(tensor))


def clamp_output(tensor: torch.Tensor, threshold):
    return torch.where(tensor > threshold, torch.tensor(1.0), torch.tensor(0.0))


def load_FOM_model(model_path, weights_path):
    with open(model_path, "r") as file:
        data = file.read()

    FOM_calculator = tf.keras.models.model_from_json(data)
    FOM_calculator.load_weights(weights_path)

    return FOM_calculator


train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
)

FOM_calculator = load_FOM_model("../Files/VGGnet.json", "../Files/VGGnet_weights.h5")
FOM_measurements = []

i = 0
for batch in tqdm(train_loader):
    if normalize:
        batch = mean_normalize(batch)
    batch = batch.unsqueeze(1)
    if clamp:
        batch = clamp_output(batch, 0.5)

    FOM = FOM_calculator(torch.permute(batch.repeat(1, 3, 1, 1), (0, 2, 3, 1)).numpy())
    FOM_measurements.extend(FOM.numpy().flatten().tolist())

with open("Dataset_Summary.txt", "w") as file:
    file.write(f"Dataset FOM max: {max(FOM_measurements)}\n")
    file.write(f"Dataset FOM min: {min(FOM_measurements)}\n")
    file.write(f"Clamped: {clamp}\n")
    file.write(f"Normalized: {normalize}\n")
    file.write("Values in \{0, 1 \}")
