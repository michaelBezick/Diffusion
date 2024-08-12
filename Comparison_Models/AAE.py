import argparse
import os
import numpy as np
import itertools
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset, dataloader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F 
import torch
import time

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=4000, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=256, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=15, help='dimensionality of the latent code')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=200, help='interval between image sampling')
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels,opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(torch.Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
    sampled_z = sampled_z.cuda()
    z = sampled_z * std + mu
    return z

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

        self.mu = nn.Linear(self.model_size, opt.latent_dim)
        self.logvar = nn.Linear(self.model_size, opt.latent_dim)

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
            nn.Linear(opt.latent_dim+2, self.model_size),
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

class Discriminator(nn.Module):
    def __init__(self, model_size):
        super(Discriminator, self).__init__()
        self.model_size = model_size

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim+2, model_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(model_size, model_size // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(model_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
#        z_label=torch.cat((z,labels), -1)
        validity = self.model(z)
        return validity

# Use binary cross-entropy loss
adversarial_loss = torch.nn.BCELoss()
pixelwise_loss = torch.nn.L1Loss()

# Initialize generator and discriminator
model_size = 1500
encoder = Encoder(model_size)
decoder = Decoder(model_size)
discriminator = Discriminator(model_size)

encoder_parameters = filter(lambda p: p.requires_grad, encoder.parameters())
params1 = sum([np.prod(p.size()) for p in encoder_parameters])

print(f" encoder parameter count: {params1}")

decoder_parameters = filter(lambda p: p.requires_grad, decoder.parameters())
params2 = sum([np.prod(p.size()) for p in decoder_parameters])

print(f" decoder parameter count: {params1}")

disc_parameters = filter(lambda p: p.requires_grad, discriminator.parameters())
params3 = sum([np.prod(p.size()) for p in disc_parameters])

print(f" discriminator parameter count: {params1}")
total = params1 + params2 + params3
print(f"Total parameter count: {total}")

if cuda:
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    pixelwise_loss.cuda()

transform = transforms.Compose([
                    transforms.Resize(opt.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

#dataset in [0, 1]
dataset = np.expand_dims(np.load("../Files/TPV_dataset.npy"), 1)
normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

normalizedDataset = normalizedDataset.astype(np.float32)

dataset = torch.from_numpy(normalizedDataset)

labels = torch.load("../Files/FOM_labels.pt")
labeled_dataset = LabeledDataset(dataset, labels)

dataloader = DataLoader(
    labeled_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    drop_last=True,
)

# train_set = datasets.ImageFolder(root=r'D:\Dropbox\Purdue_work\Optimization\c_AAE\c_AAE_network\Data_cAAE',transform=transform)
# dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)
# Optimizers
optimizer_G = torch.optim.Adam( itertools.chain(encoder.parameters(), decoder.parameters()),
                                lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

def sample_image(n_row, z, batches_done):
    """Saves a grid of generated digits"""
    # Sample noise
    gen_imgs = decoder(z)
    save_image(gen_imgs.data, 'images/%d.png' % batches_done, nrow=n_row, normalize=True)
#%%
# ----------
#  Training
# ----------
DlossMeanTot = np.zeros(opt.n_epochs)
DlossMeanReal = np.zeros(opt.n_epochs)
DlossMeanFake = np.zeros(opt.n_epochs)
GlossMean = np.zeros(opt.n_epochs)

for epoch in range(opt.n_epochs):
    time1 = time.time()
    dlossmeanTot = 0
    dlossmeanReal = 0
    dlossmeanFake = 0
    glossmean = 0
    for i, (imgs, labels) in enumerate(dataloader):
                
        imgs = imgs.cuda()
        labels = labels.cuda()
       # Adversarial ground truths
        valid = Variable(torch.Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        valid = valid.cuda()
        fake = Variable(torch.Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
        fake = fake.cuda()

        # Configure input
        real_imgs = Variable(imgs.type(torch.Tensor))

        # -----------------
        #  Train Generator
        # -----------------
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
        labelVect = Variable(torch.Tensor(zz1))       
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
#        labelVect = Variable(LongTensor(zz1))
        zReal = torch.cat((z,labelVect), -1)
            
        
        optimizer_G.zero_grad()

        real_imgs = real_imgs.cuda()
        labelVect = labelVect.cuda()
#
        encoded_imgs = encoder(real_imgs, labelVect)        
        decoded_imgs = decoder(encoded_imgs)
        
        # Loss measures generator's ability to fool the discriminator
        g_loss =    0.001 * adversarial_loss(discriminator(encoded_imgs), valid) + \
                    0.999 * pixelwise_loss(decoded_imgs, real_imgs)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()
        
        zReal = zReal.cuda()
        valid = valid.cuda()
        real_loss = adversarial_loss(discriminator(zReal), valid)
        fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()
        time2 = time.time()
        print(f"time per epoch:{time2 - time1}")
        print(f"time per experiment {(time2 - time1) * 4000}")
        exit()

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                            d_loss.item(), g_loss.item()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            zrand = np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))
            zrand_label = np.zeros((imgs.shape[0], opt.latent_dim+2))
            for jj in range(imgs.shape[0]):
                zrand_label[jj,:] = np.append(zrand[jj,:],[0.25,0.03])
                pass
            
            z = Variable(torch.Tensor(zrand_label))
            z = z.cuda()
            gen_imgs = decoder(z)
            save_image(gen_imgs.data[0], 'AAE_images/%d.png' % batches_done, nrow=1, normalize=True)            

        dlossmeanTot = dlossmeanTot+d_loss.item()
        dlossmeanReal = dlossmeanReal+real_loss.item()
        dlossmeanFake = dlossmeanFake+fake_loss.item()
        glossmean = glossmean+g_loss.item()
        pass
    DlossMeanTot[epoch] = dlossmeanTot/imgs.shape[0]
    DlossMeanFake[epoch] = dlossmeanFake/imgs.shape[0]
    DlossMeanReal[epoch] = dlossmeanReal/imgs.shape[0]
    GlossMean[epoch] = glossmean/imgs.shape[0]
    pass

#%% Saving model
torch.save(decoder.state_dict(),'AAE_Decoder')
torch.save(encoder.state_dict(),'AAE_Encoder')
