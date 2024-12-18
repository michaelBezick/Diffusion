import math as m
import lightning as pl

import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image
from metalayers import *
from torch import nn, optim
from torchvision.models import VGG16_Weights

import math as m
from enum import Enum

import numpy as np
import pytorch_lightning as pl
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import nn, optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import Dataset
from torchvision.models import VGG16_Weights
from tqdm import tqdm

class cVAE(pl.LightningModule):
    """
    Variational autoencoder with UNet structure.
    """

    def __init__(
        self,
        in_channels=1,
        h_dim=32,
        lr=1e-3,
        batch_size=100,
        perceptual_loss_scale=1.0,
        kl_divergence_scale=1.0,
    ):
        super().__init__()

        self.batch_size = batch_size

        self.lr = lr

        self.perceptual_loss_scale = perceptual_loss_scale
        self.kl_divergence_scale = kl_divergence_scale

        self.attention1E = AttnBlock(h_dim)
        self.attention2E = AttnBlock(h_dim)
        self.resnet1E = ResnetBlockVAE(h_dim, h_dim, (3, 3), h_dim)
        self.resnet2E = ResnetBlockVAE(h_dim, h_dim, (3, 3), h_dim)
        self.resnet3E = ResnetBlockVAE(h_dim, h_dim, (3, 3), h_dim)
        self.resnet4E = ResnetBlockVAE(h_dim, h_dim, (3, 3), h_dim)
        self.resnet5E = ResnetBlockVAE(h_dim, h_dim, (3, 3), h_dim)
        self.resnet6E = ResnetBlockVAE(h_dim, h_dim, (3, 3), h_dim)
        self.maxPool = nn.MaxPool2d((2, 2), 2)

        self.encoder = nn.Sequential(
            nn.Conv2d(2, h_dim, kernel_size=(3, 3), padding="same"),
            nn.SiLU(),
            self.resnet1E,
            self.resnet2E,
            self.maxPool,  # 16 x 16
            self.resnet3E,
            self.attention1E,
            self.resnet4E,
            self.maxPool,  # 8 x 8
            self.resnet5E,
            self.attention2E,
            self.resnet6E,
        )

        self.to_mu = nn.Conv2d(h_dim, 1, (1, 1))
        self.to_sigma = nn.Conv2d(h_dim, 1, (1, 1))

        self.attention1D = AttnBlock(h_dim)
        self.attention2D = AttnBlock(h_dim)
        self.resnet1D = ResnetBlockVAE(h_dim, h_dim, (3, 3), h_dim)
        self.resnet2D = ResnetBlockVAE(h_dim, h_dim, (3, 3), h_dim)
        self.resnet3D = ResnetBlockVAE(h_dim, h_dim, (3, 3), h_dim)
        self.resnet4D = ResnetBlockVAE(h_dim, h_dim, (3, 3), h_dim)
        self.resnet5D = ResnetBlockVAE(h_dim, h_dim, (3, 3), h_dim)
        self.resnet6D = ResnetBlockVAE(h_dim, h_dim, (3, 3), h_dim)

        self.decoder = nn.Sequential(
            nn.Conv2d(2, h_dim, (1, 1)),
            self.resnet1D,
            self.attention1D,
            self.resnet2D,
            nn.ConvTranspose2d(h_dim, h_dim, (2, 2), 2),  # 16 x 16
            self.resnet3D,
            self.attention2D,
            self.resnet4D,
            nn.ConvTranspose2d(h_dim, h_dim, (2, 2), 2),  # 32 x 32
            self.resnet5D,
            self.resnet6D,
            nn.Conv2d(h_dim, 1, (1, 1)),
        )

        self.perceptual_loss = VGGPerceptualLoss()
        self.FOM_Conditioner = nn.Linear(1, 32 * 32)
        self.FOM_Conditioner_latent = nn.Linear(1, 8 * 8)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.to_mu(h)
        sigma = self.to_sigma(h)
        return mu, sigma

    def decode(self, z):
        return self.decoder(z)

    def training_step(self, batch, batch_idx):
        images, FOMs = batch
        FOMs = FOMs.unsqueeze(1)
        images = images.float()
        FOMs = FOMs.float()
        FOMs_before = FOMs


        FOMs = self.FOM_Conditioner(FOMs_before)
        FOMs_latent = self.FOM_Conditioner_latent(FOMs_before)
        FOMs = FOMs.view(-1, 1, 32, 32)
        FOMs_latent = FOMs_latent.view(-1, 1, 8, 8)

        x = torch.cat([images, FOMs], dim = 1)

        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparameterized = mu + torch.multiply(sigma, epsilon)

        z_reparameterized = torch.cat([z_reparameterized, FOMs_latent], dim=1)

        x_hat = self.decode(z_reparameterized)

        kl_divergence = -0.5 * torch.mean(
            1 + torch.log(sigma.pow(2) + 1e-12) - mu.pow(2) - sigma.pow(2)
        )
        perceptual_loss = self.perceptual_loss(images, x_hat)

        loss = (
            perceptual_loss * self.perceptual_loss_scale
            + kl_divergence * self.kl_divergence_scale
        )

        self.log("Perceptual Loss", perceptual_loss)
        self.log("kl_divergence", kl_divergence)
        self.log("Total loss", loss)

        # logging generated images
        sample_imgs_generated = x_hat[:30]
        sample_imgs_original = x[:30]
        gridGenerated = torchvision.utils.make_grid(sample_imgs_generated)
        gridOriginal = torchvision.utils.make_grid(sample_imgs_original)

        """Tensorboard logging"""

        if self.global_step % 1000 == 0:
            self.logger.experiment.add_image(
                "Generated_images", gridGenerated, self.global_step
            )
            self.logger.experiment.add_image(
                "Original_images", gridOriginal, self.global_step
            )

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

class VGGPerceptualLoss(torch.nn.Module):
    """
    Returns perceptual loss of two batches of images
    """

    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        """
        You can adjust the indices of the appended blocks to change
        capacity and size of loss model.
        """
        blocks = []
        blocks.append(
            torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).features[:4].eval()
        )
        blocks.append(
            torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT).features[4:8].eval()
        )
        blocks.append(
            torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
            .features[8:14]
            .eval()
        )
        blocks.append(
            torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
            .features[14:20]
            .eval()
        )
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(
                input, mode="bilinear", size=(224, 224), align_corners=False
            )
            target = self.transform(
                target, mode="bilinear", size=(224, 224), align_corners=False
            )
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)

        return loss

class ResnetBlockVAE(nn.Module):
    # Employs intra block skip connection, which needs a (1, 1) convolution to scale to out_channels

    def __init__(self, in_channels, out_channels, kernel_size, in_channel_image):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer1 = Block(in_channels, out_channels, kernel_size)
        self.SiLU = nn.SiLU()
        self.layer2 = Block(out_channels, out_channels, kernel_size)
        self.resizeInput = nn.Conv2d(in_channel_image, out_channels, (1, 1))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        xCopy = x

        x = self.layer1(x)
        x = self.SiLU(x)
        x = self.layer2(x)
        xCopy = self.resizeInput(xCopy)
        x = x + xCopy
        x = self.SiLU(x)

        return x


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layer = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, padding="same", bias=False
            ),
            nn.GroupNorm(8, out_channels),
        )

    def forward(self, x):
        x = self.layer(x)

        return x


class FOM_Conditioner(nn.Module):
    def __init__(self, batch_size=0, height=0, width=0, embedding_length=0, channels=1):
        super().__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.channels = channels
        self.layer = nn.Linear(embedding_length, height * width)
        self.SiLU = nn.SiLU()

    def forward(self, FOM_embeddings):
        x = self.layer(FOM_embeddings)
        x = self.SiLU(x)
        return x.view(self.batch_size, self.channels, self.height, self.width)


class ResnetBlock(nn.Module):
    # Employs intra block skip connection, which needs a (1, 1) convolution to scale to out_channels

    def __init__(self, in_channels, out_channels, kernel_size, in_channel_image):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer1 = Block(in_channels, out_channels, kernel_size)
        self.SiLU = nn.SiLU()
        self.layer2 = Block(out_channels, out_channels, kernel_size)
        self.resizeInput = nn.Conv2d(in_channel_image, out_channels, (1, 1))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, time_step_embeddings, FOM_embeddings):
        xCopy = x

        x = torch.cat((x, time_step_embeddings, FOM_embeddings), dim=1)

        x = self.layer1(x)
        x = self.SiLU(x)
        x = self.layer2(x)
        xCopy = self.resizeInput(xCopy)
        x = x + xCopy
        x = self.SiLU(x)

        return x


class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = m.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=8, num_channels=self.in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


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

class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, dim, labels_dim=1, batch_size=32):
        super(Generator, self).__init__()

        self.dim = dim
        h_dim = dim
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.feature_sizes = (int(self.img_size[0] // 16), int(self.img_size[1] // 16))

        self.FC = nn.Sequential(
                nn.Linear(latent_dim + labels_dim, 512), 
                nn.ReLU(),
                nn.Linear(512, 4 * 4 * 64, bias=False),
                nn.BatchNorm1d(4 * 4 * 64),
                nn.ReLU(),

        )

        self.decoder = nn.Sequential(
                ConvTranspose2d_meta(64, 64, 5, stride=2, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                ConvTranspose2d_meta(64, 32, 5, stride=2, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                ConvTranspose2d_meta(32, 16, 5, stride=2, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                ConvTranspose2d_meta(16, 1, 5),
                )

        self.gkernel = gkern(3, 2)
        self.tanh = nn.Tanh()

    def forward(self, input_data: torch.Tensor, labels: torch.Tensor):
        x = torch.cat([input_data, labels], 1)
        x = self.FC(x)
        x = x.view(-1, 64, 4, 4)
        x = self.decoder(x)
        x = conv2d_meta(x, self.gkernel)

        return self.tanh(x)

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))


class Discriminator(nn.Module):
    def __init__(self, img_size, dim, label_dim=1, batch_size=32):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(Discriminator, self).__init__()

        self.img_size = img_size

        self.CONV = nn.Sequential(
                Conv2d_meta(1, 64, 5, stride=2),
                nn.LeakyReLU(0.2),
                )
        self.FC = nn.Sequential(
                nn.Linear(16385, 512),
                nn.LayerNorm(512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 1),
                )
        self.batch_size = batch_size

    def forward(self, input_data, labels):

        input_data = input_data + torch.randn_like(input_data, device="cuda") * 0.05

        # self.save_image_grid(input_data, "noised_data.png")

        x = self.CONV(input_data)
        x = x.view(self.batch_size, -1)
        x = torch.cat([x, labels], 1)
        x = self.FC(x)

        return x


    def save_image_grid(self, tensor, filename, nrow=8, padding=2):
        # Make a grid from batch tensor
        grid_image = torchvision.utils.make_grid(
            tensor, nrow=nrow, padding=padding, normalize=True
        )

        # Convert to numpy array and then to PIL image
        grid_image = (
            grid_image.permute(1, 2, 0)
            .mul(255)
            .clamp(0, 255)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        pil_image = Image.fromarray(grid_image)

        # Save as PNG
        pil_image.save(filename, bitmap_format="png")
