import torch
import numpy as np

"""
Have to preprocess to be B, C, H, W with 3 RGB channels with dtype uint8
Just saving dataset now for repo to do it
"""

def expand_output(tensor: torch.Tensor, num_samples):
    x = torch.zeros([num_samples, 1, 64, 64])
    x[:, :, 0:32, 0:32] = tensor
    x[:, :, 32:64, 0:32] = torch.flip(tensor, dims=[2])
    x[:, :, 0:32, 32:64] = torch.flip(tensor, dims=[3])
    x[:, :, 32:64, 32:64] = torch.flip(tensor, dims=[2, 3])

    return x
dataset = np.load("../Files/TPV_dataset.npy")

normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

normalizedDataset = np.multiply(normalizedDataset, 2) - 1

normalizedDataset = normalizedDataset.astype(np.float32)

original_dataset = torch.from_numpy(normalizedDataset)
generated_dataset = torch.load("../Generated_Datasets/Experiment_8/generated_dataset.pt")

generated_dataset = (generated_dataset - torch.min(generated_dataset)) / (torch.max(generated_dataset) - torch.min(generated_dataset))
original_dataset = (original_dataset - torch.min(original_dataset)) / (torch.max(original_dataset) - torch.min(original_dataset))

generated_dataset = expand_output(generated_dataset, 20_000)

generated_dataset = generated_dataset.expand(20_000, 3, 64, 64)
original_dataset = original_dataset.unsqueeze(1).expand(12_000, 3, 64, 64)

generated_dataset = generated_dataset.to(torch.half)
original_dataset = original_dataset.to(torch.half)

generated_dataset = generated_dataset.numpy()
original_dataset = original_dataset.numpy()

np.save("generated_dataset_LDM.npy", generated_dataset)
np.save("original_dataset.npy", original_dataset)
