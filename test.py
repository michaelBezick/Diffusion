import torch

labels = torch.randint(low=0, high=10, size=(2, 1))
print(labels)
# labels = labels.expand(2, 3)
# labels = labels.unsqueeze(2)
# labels = labels.expand(2, 3, 3)

labels = labels.unsqueeze(2)
labels = labels.expand(2, 3, 3)

print(labels)
print(labels.size())
