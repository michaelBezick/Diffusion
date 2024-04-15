import math as m

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from PIL import Image
from metalayers import *

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
