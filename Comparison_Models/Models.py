import math as m

import torch
import torch.nn as nn
from torch.utils.data import Dataset


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
    def __init__(self, img_size, latent_dim, dim, labels_dim=1):
        super(Generator, self).__init__()

        self.dim = dim
        h_dim = dim
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.feature_sizes = (int(self.img_size[0] // 16), int(self.img_size[1] // 16))

        self.latent_to_features = nn.Sequential(
            nn.Linear(
                latent_dim + labels_dim,
                64,
            ),
            nn.ReLU(),
        )

        self.attention1D = AttnBlock(h_dim)
        self.attention2D = AttnBlock(h_dim)
        self.resnet1D = ResnetBlockVAE(h_dim, h_dim, (3, 3), h_dim)
        self.resnet2D = ResnetBlockVAE(h_dim, h_dim, (3, 3), h_dim)
        self.resnet3D = ResnetBlockVAE(h_dim, h_dim, (3, 3), h_dim)
        self.resnet4D = ResnetBlockVAE(h_dim, h_dim, (3, 3), h_dim)
        self.resnet5D = ResnetBlockVAE(h_dim, h_dim, (3, 3), h_dim)
        self.resnet6D = ResnetBlockVAE(h_dim, h_dim, (3, 3), h_dim)

        self.decoder = nn.Sequential(
            nn.Conv2d(1, h_dim, (1, 1)),
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

    def forward(self, input_data, labels):
        # Map latent into appropriate size for transposed convolutions
        x = torch.cat((input_data, labels), dim=1)
        x = self.latent_to_features(x)
        # Reshape
        x = x.view(-1, 1, 8, 8)
        # Return generated image
        return self.decoder(x)

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim))


class Discriminator(nn.Module):
    def __init__(self, img_size, dim, label_dim=1):
        """
        img_size : (int, int, int)
            Height and width must be powers of 2.  E.g. (32, 32, 1) or
            (64, 128, 3). Last number indicates number of channels, e.g. 1 for
            grayscale or 3 for RGB
        """
        super(Discriminator, self).__init__()

        self.img_size = img_size

        self.attention1E = AttnBlock(dim)
        self.attention2E = AttnBlock(dim)
        self.resnet1E = ResnetBlockVAE(dim, dim, (3, 3), dim)
        self.resnet2E = ResnetBlockVAE(dim, dim, (3, 3), dim)
        self.resnet3E = ResnetBlockVAE(dim, dim, (3, 3), dim)
        self.resnet4E = ResnetBlockVAE(dim, dim, (3, 3), dim)
        self.resnet5E = ResnetBlockVAE(dim, dim, (3, 3), dim)
        self.resnet6E = ResnetBlockVAE(dim, dim, (3, 3), dim)
        self.maxPool = nn.MaxPool2d((2, 2), 2)

        self.encoder = nn.Sequential(
            nn.Conv2d(1 + label_dim, dim, kernel_size=(3, 3), padding="same"),
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

        self.decrease_channels = nn.Conv2d(dim, 1, kernel_size=1, stride=1)

        # 4 convolutions of stride 2, i.e. halving of size everytime
        # So output size will be 8 * (img_size / 2 ^ 4) * (img_size / 2 ^ 4)
        output_size = 64
        self.features_to_prob = nn.Sequential(nn.Linear(output_size, 1), nn.Sigmoid())

    def forward(self, input_data, labels):
        batch_size = input_data.size()[0]

        labels = labels.unsqueeze(2).unsqueeze(3)
        labels = labels.expand(batch_size, 1, 32, 32)

        x = torch.cat((input_data, labels), dim=1)
        x = self.encoder(x)
        x = self.decrease_channels(x)
        x = x.view(batch_size, -1)

        return self.features_to_prob(x)
