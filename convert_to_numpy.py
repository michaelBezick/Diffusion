import torch
import numpy as np

path = ""
save_path = ""
dataset = torch.load(path)
dataset = dataset * 255 #IDK WAHHHHH
dataset = dataset.numpy()
np.save(save_path, dataset)
