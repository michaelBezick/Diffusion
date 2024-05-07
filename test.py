import torch

data = torch.randn((100, 1, 32, 32))
labels = torch.randint(low=0, high=10, size=(100, 1))

labels = labels.unsqueeze(2).unsqueeze(3)
labels = labels.expand(100, 1, 32, 32)

print(labels)
print(labels.size())
final = torch.cat((data, labels), dim=1)
print(final)
print(final.size())
print(final[1, 1, :, :])
print(final[0, 1, :, :])
