import torch

FOM_values = torch.randn(100).unsqueeze(0).expand(10, -1).reshape(-1)
print(FOM_values.size())
print(FOM_values[0] == FOM_values[100])
print(FOM_values[0:10])
print(FOM_values[100:110])
