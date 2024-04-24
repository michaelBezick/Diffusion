import torch
import numpy as np

def expand_output(tensor: torch.Tensor, num_samples):
    x = torch.zeros([num_samples, 1, 64, 64])
    x[:, :, 0:32, 0:32] = tensor
    x[:, :, 32:64, 0:32] = torch.flip(tensor, dims=[2])
    x[:, :, 0:32, 32:64] = torch.flip(tensor, dims=[3])
    x[:, :, 32:64, 32:64] = torch.flip(tensor, dims=[2, 3])

    return x

path = "../Diffusion/Generated_Datasets/Experiment_10/generated_dataset.pt"
save_path = "experiment_10_dataset.npy"
dataset = torch.load(path)
dataset = dataset * 255 #IDK WAHHHHH
dataset = expand_output(dataset, dataset.size[0])
dataset = dataset.numpy()
np.save(save_path, dataset)
