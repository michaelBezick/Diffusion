import numpy as np
import torch
import torch.optim as optim
from Models import Discriminator, Generator, LabeledDataset
from torch.utils.data import DataLoader
from Training import Trainer

batch_size = 32

dataset = np.expand_dims(np.load("../Files/TPV_dataset.npy"), 1)
normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

normalizedDataset = np.multiply(normalizedDataset, 2) - 1

normalizedDataset = normalizedDataset.astype(np.float32)

dataset = torch.from_numpy(normalizedDataset)

labels = torch.load("../Files/FOM_labels.pt")

labeled_dataset = LabeledDataset(dataset, labels)

data_loader = DataLoader(
    labeled_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

img_size = (32, 32, 1)

generator = Generator(img_size=img_size, latent_dim=100, dim=68).cuda()
discriminator = Discriminator(img_size=img_size, dim=68).cuda()

model_parameters = filter(lambda p: p.requires_grad, generator.parameters())
params1 = sum([np.prod(p.size()) for p in model_parameters])

print(f" Generator parameter count: {params1}")

model_parameters = filter(lambda p: p.requires_grad, discriminator.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print(f" Discriminator parameter count: {params}")
print(f"Total {params1 + params}")


print(generator)
print(discriminator)

# Initialize optimizers
lr = 1e-5
betas = (0.9, 0.99)
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

# Train model
epochs = 200
trainer = Trainer(
    generator,
    discriminator,
    G_optimizer,
    D_optimizer,
    use_cuda=torch.cuda.is_available(),
)
trainer.train(data_loader, epochs)

# Save models
name = "mnist_model"
torch.save(trainer.G.state_dict(), "./gen_" + name + ".pt")
torch.save(trainer.D.state_dict(), "./dis_" + name + ".pt")