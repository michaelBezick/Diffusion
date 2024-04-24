import numpy as np
import torch
from tqdm import tqdm
dataset = np.load("../Files/TPV_dataset.npy")
dataset = torch.from_numpy(dataset)
print(torch.min(dataset))
print(torch.max(dataset))
exit()
