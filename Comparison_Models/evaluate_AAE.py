import matplotlib.pyplot as plt
import numpy as np
from numpy.core.multiarray import where
import tensorflow as tf
import torch
from torch.utils.data import Dataset, dataloader
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
from PIL import Image
from tqdm import tqdm
from Models import cVAE
import torch.nn as nn
import time

clamp = False
batch_size = 1000
lr = 1e-3
perceptual_loss_scale = 1
kl_divergence_scale = 0.1
checkpoint_path_encoder = "./AAE_Encoder"
checkpoint_path_decoder = "./AAE_Decoder" 
img_shape = (1, 32, 32)
latent_dim = 15

def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(torch.Tensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
    sampled_z = sampled_z.cuda()
    z = sampled_z * std + mu
    return z

class Encoder(nn.Module):
    def __init__(self, model_size):
        super(Encoder, self).__init__()
        self.model_size = model_size

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), self.model_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.model_size, self.model_size),
            nn.BatchNorm1d(self.model_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.mu = nn.Linear(self.model_size, latent_dim)
        self.logvar = nn.Linear(self.model_size, latent_dim)

    def forward(self, img, labels):
        img_flat = img.view(img.shape[0], -1)
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        z_label = torch.cat((z,labels), -1)
        return z_label

class Decoder(nn.Module):
    def __init__(self, model_size):
        super(Decoder, self).__init__()
        self.model_size = model_size

        self.model = nn.Sequential(
            nn.Linear(latent_dim+2, self.model_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.model_size, self.model_size),
            nn.BatchNorm1d(self.model_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.model_size, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        # Concatenate label embedding and image to produce input
#        gen_input = torch.cat((z,labels), -1)
        img_flat = self.model(z)
        img = img_flat.view(img_flat.shape[0], *img_shape)
        return img

encoder = Encoder(1500)
encoder.load_state_dict(torch.load(checkpoint_path_encoder))
decoder = Decoder(1500)
decoder.load_state_dict(torch.load(checkpoint_path_decoder))
encoder = encoder.cuda()
decoder = decoder.cuda()
def clamp_output(tensor: torch.Tensor, threshold):
    return torch.where(tensor > threshold, torch.tensor(1.0), torch.tensor(0.0))

dataset = np.expand_dims(np.load("../Files/TPV_dataset.npy"), 1)
normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

normalizedDataset = normalizedDataset.astype(np.float32)

dataset = torch.from_numpy(normalizedDataset)

class LabeledDataset(Dataset):
    def __init__(self, images, labels, size=32, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image[:, 0 : self.size, 0 : self.size], label

def save_image_grid(tensor, filename, nrow=8, padding=2):
    # Make a grid from batch tensor
    grid_image = torchvision.utils.make_grid(
            tensor, nrow=nrow, padding=padding, normalize=True
    )

    # Convert to numpy array and then to PIL image
    grid_image = (
        grid_image.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    )
    pil_image = Image.fromarray(grid_image)

    # Save as PNG
    pil_image.save(filename, bitmap_format="png")

def load_FOM_model(model_path, weights_path):
    with open(model_path, "r") as file:
        data = file.read()

    FOM_calculator = tf.keras.models.model_from_json(data)
    FOM_calculator.load_weights(weights_path)

    return FOM_calculator


def expand_output(tensor: torch.Tensor, num_samples):
    x = torch.zeros([num_samples, 1, 64, 64])
    x[:, :, 0:32, 0:32] = tensor
    x[:, :, 32:64, 0:32] = torch.flip(tensor, dims=[2])
    x[:, :, 0:32, 32:64] = torch.flip(tensor, dims=[3])
    x[:, :, 32:64, 32:64] = torch.flip(tensor, dims=[2, 3])

    return x


num_samples = 20_000

plot = True
mean = 1.8
variance = 0.1

FOM_calculator = load_FOM_model("../Files/VGGnet.json", "../Files/VGGnet_weights.h5")

labels = torch.load("../Files/FOM_labels.pt")

labeled_dataset = LabeledDataset(dataset, labels)

train_loader = DataLoader(
    labeled_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

labels_list = []
FOMs_list = []

dataset = []
time1 = time.time()
with torch.no_grad():
    i = 0
    loop = True
    while loop:
        for batch in tqdm(train_loader):
            print(i)
            if (i >= num_samples // batch_size) :
                loop = False
                break
            i += 1
            imgs, labels = batch
            labels = labels.unsqueeze(1)
            imgs = imgs.cuda().float()
            labels = labels.cuda().float()
            zz1 = np.zeros((imgs.shape[0], 2))
            # Sample noise as generator input
            for jj in range(imgs.shape[0]):
                if labels[jj] == 0:
                    zz1[jj,:]=[0.25,0.03]
                    pass
                if labels[jj] == 1:
                    zz1[jj,:]=[0.25,0.05]
                    pass
                if labels[jj] == 2:
                    zz1[jj,:]=[0.28,0.03]
                    pass
                if labels[jj] == 3:
                    zz1[jj,:]=[0.28,0.05]
                    pass
                if labels[jj] == 4:
                    zz1[jj,:]=[0.3,0.03]
                    pass
                if labels[jj] == 5:
                    zz1[jj,:]=[0.3,0.05]
                    pass
            # Sample noise and labels as generator input
            labelVect = Variable(torch.Tensor(zz1)).cuda()

            # labels = variance * torch.randn((batch_size, 1), device="cuda") + mean

            encoded_imgs = encoder(imgs, labelVect) #experiment with labels or FOMs
            decoded_imgs = decoder(encoded_imgs)

            dataset.extend(decoded_imgs.detach().cpu().numpy())
            images = expand_output(decoded_imgs, batch_size)
            if clamp:
                images = clamp_output(images, 0.5)
            # if i == 1:
            #     save_image_grid(images, "cVAE_Sample.png")

            FOMs = FOM_calculator(
                torch.permute(images.repeat(1, 3, 1, 1), (0, 2, 3, 1)).cpu().numpy()
            )

            FOMs_list.extend(FOMs.numpy().flatten().tolist())
time2 = time.time()
print(f"Experiment time: {time2 - time1}")
print(f"Minutes: {(time2 - time1) / 60}")
exit()
dataset = torch.from_numpy(np.array(dataset))
dataset = (dataset - torch.min(dataset)) / (torch.max(dataset) - torch.min(dataset))
dataset = dataset.to(torch.half)
dataset = dataset.numpy()
np.save("AAE_dataset.npy", dataset)

with open("AAE_Evaluation.txt", "w") as file:
    file.write(f"FOM max: {max(FOMs_list)}\n")
    file.write(f"FOM mean: {sum(FOMs_list) / len(FOMs_list)}")
