import torch
import torch.nn as nn
from torch.utils.data import Dataset


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
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.feature_sizes = (int(self.img_size[0] // 16), int(self.img_size[1] // 16))

        self.latent_to_features = nn.Sequential(
            nn.Linear(
                latent_dim + labels_dim,
                8 * dim * self.feature_sizes[0] * self.feature_sizes[1],
            ),
            nn.ReLU(),
        )

        self.features_to_image = nn.Sequential(
            nn.ConvTranspose2d(8 * dim, 4 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(4 * dim),
            nn.ConvTranspose2d(4 * dim, 2 * dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(2 * dim),
            nn.ConvTranspose2d(2 * dim, dim, 4, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            nn.ConvTranspose2d(dim, self.img_size[2], 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_data, labels):
        # Map latent into appropriate size for transposed convolutions
        x = torch.cat((input_data, labels), dim=1)
        x = self.latent_to_features(x)
        # Reshape
        x = x.view(-1, 8 * self.dim, self.feature_sizes[0], self.feature_sizes[1])
        # Return generated image
        return self.features_to_image(x)

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

        self.image_to_features = nn.Sequential(
            nn.Conv2d(
                self.img_size[2] + label_dim, dim, 4, 2, 1
            ),  # +1 for label conditioning
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim, 2 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * dim, 4 * dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4 * dim, 8 * dim, 4, 2, 1),
            nn.Sigmoid(),
        )

        # self.label_projector == nn.Sequential(
        #         nn.Linear(1, )
        #         )

        # 4 convolutions of stride 2, i.e. halving of size everytime
        # So output size will be 8 * (img_size / 2 ^ 4) * (img_size / 2 ^ 4)
        output_size = 8 * dim * int(img_size[0] // 16) * int(img_size[1] // 16)
        self.features_to_prob = nn.Sequential(nn.Linear(output_size, 1), nn.Sigmoid())

    def forward(self, input_data, labels):
        batch_size = input_data.size()[0]

        labels = labels.unsqueeze(2).unsqueeze(3)
        labels = labels.expand(batch_size, 1, 32, 32)

        x = torch.cat((input_data, labels), dim=1)
        x = self.image_to_features(x)
        x = x.view(batch_size, -1)
        return self.features_to_prob(x)
