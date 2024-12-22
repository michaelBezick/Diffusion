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


class Model_Type(Enum):
    QUBO = 1
    PUBO = 2
    ISING = 3
    BLUME_CAPEL = 4
    POTTS = 5


def expand_output(tensor: torch.Tensor, num_samples):
    x = torch.zeros([num_samples, 1, 64, 64])
    x[:, :, 0:32, 0:32] = tensor
    x[:, :, 32:64, 0:32] = torch.flip(tensor, dims=[2])
    x[:, :, 0:32, 32:64] = torch.flip(tensor, dims=[3])
    x[:, :, 32:64, 32:64] = torch.flip(tensor, dims=[2, 3])

    return x


def load_FOM_model(model_path, weights_path):
    with open(model_path, "r") as file:
        data = file.read()

    FOM_calculator = tf.keras.models.model_from_json(data)
    FOM_calculator.load_weights(weights_path)

    return FOM_calculator


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

class Ablation_LDM_Pytorch(nn.Module):

    def __init__(
        self,
        DDPM,
        VAE,
        in_channels,
        batch_size,
        num_steps,
        latent_height,
        latent_width,
        lr,
        logger,
        device,
    ):
        super().__init__()
        self.device = device
        self.logger = logger
        self.lr = lr
        self.DDPM = DDPM
        self.VAE = VAE
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.latent_height = latent_height
        self.height = latent_height
        self.latent_width = latent_width
        self.width = latent_width
        self.num_steps = num_steps
        self.random_generator = MultivariateNormal(
            torch.zeros(latent_height * latent_width * in_channels),
            torch.eye(latent_height * latent_width * in_channels),
        )
        self.beta_schedule = self.make_beta_schedule(num_steps, 1e-5, 0.02)
        self.alpha_schedule = torch.from_numpy(
            self.get_alpha_schedule(self.beta_schedule)
        )
        self.alpha_bar_schedule = self.calculate_alpha_bar(self.alpha_schedule).to(
            self.device
        )
        self.alpha_schedule = self.alpha_schedule.to(self.device)
        self.global_step = 0

    def training_step(self, batch):
        images, FOMs = batch

        images = images.to(self.device)
        FOMs = FOMs.to(self.device)

        self.global_step += 1

        if self.global_step % 300 == 0:
            self.sample()

        # random timestep
        t = torch.randint(0, self.num_steps - 1, (self.batch_size,), device=self.device)

        # generating epsilon_0
        epsilon_sample = self.random_generator.sample((self.batch_size,)).to(self.device)
        epsilon_0 = epsilon_sample.view(
            self.batch_size, self.in_channels, self.height, self.width
        )

        # broadcasting alpha_bar_vector
        alpha_bar_vector = self.alpha_bar_schedule[t].float()
        alpha_bar_vector = alpha_bar_vector.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_bar_vector = alpha_bar_vector.expand(
            -1, self.in_channels, self.height, self.width
        )

        # encoding to latent space
        x = images
        with torch.no_grad():
            mu, sigma = self.VAE.encode(x)

        z_reparameterized = mu + torch.multiply(
            sigma, torch.randn_like(sigma, device=self.device)
        )

        # latent_encoding_sample = torch.bernoulli(
        #     latent_encoding_logits
        # )  # I don't think I need a gradient here
        """
        Not doing VAE anymore, just VAE
        """
        # epsilon = torch.randn_like(sigma)
        # z_reparameterized = mu + torch.multiply(sigma, epsilon)

        if self.global_step % 300 == 0:
            latent_grid = torchvision.utils.make_grid(z_reparameterized)
            self.logger.add_image(
                "True latent grid", latent_grid, self.global_step
            )

        ### Creation of x_t, algorithm 1 of Ho et al.###
        mu = torch.mul(torch.sqrt(alpha_bar_vector), z_reparameterized)
        variance = torch.mul(
            torch.sqrt(torch.ones_like(alpha_bar_vector) - alpha_bar_vector), epsilon_0
        )
        x_t = torch.add(mu, variance)

        # experiment with this
        tNotNormalized = t.float()
        FOMs = FOMs.float()

        # predicted true epsilon in latent space
        epsilon_theta_latent = self.DDPM.forward(x_t, FOMs, tNotNormalized)

        # calculating loss
        loss = F.smooth_l1_loss(epsilon_0, epsilon_theta_latent)
        self.logger.add_scalar("train_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def make_beta_schedule(self, num_steps, beta_1, beta_T):
        return np.linspace(beta_1, beta_T, num=num_steps)

    def get_alpha_schedule(self, beta_schedule):
        return np.ones_like(beta_schedule) - beta_schedule

    def calculate_alpha_bar(self, alpha_schedule):
        return np.cumprod(alpha_schedule)

    def on_train_start(self):
        self.alpha_schedule = self.alpha_schedule.to(self.device)
        self.alpha_bar_schedule = self.alpha_bar_schedule.to(self.device)
        self.random_generator = MultivariateNormal(
            torch.zeros(
                self.height * self.width * self.in_channels, device=self.device
            ),
            torch.eye(self.height * self.width * self.in_channels, device=self.device),
        )

    def create_dataset_variable_FOM(self, num_samples, start_mean, end_mean, variance):
        dataset = []
        for _ in tqdm(range(num_samples // self.batch_size)):
            with torch.no_grad():
                x_T = self.random_generator.sample((self.batch_size,))
                x_T = x_T.view(
                    self.batch_size, self.in_channels, self.height, self.width
                )

                previous_image = x_T.to(self.device)

                # runs diffusion process from pure noise to timestep 0
                """
                T ranges from 999 to 0, so must linearly scale to start mean to end mean
                This can be accomplished by first inverting t to (num_steps - t), then dividing by 1000 to get in range [0, 1],
                then multiplying this scalar to the difference between start and end mean, then adding to start mean
                """

                difference_mean = end_mean - start_mean
                for t in range(self.num_steps - 1, -1, -1):

                    scaled_t = (self.num_steps - t) / 1000
                    functional_mean = scaled_t * difference_mean + start_mean

                    FOM_values = (
                        variance * torch.randn(self.batch_size, device=self.device)
                        + functional_mean
                    )

                    if t > 0:
                        z = self.random_generator.sample((self.batch_size,)).view(
                            self.batch_size, self.in_channels, self.height, self.width
                        )
                    elif t == 0:
                        z = torch.zeros_like(x_T)

                    timeStep = torch.tensor(t).to(self.device)
                    timeStep = timeStep.repeat(self.batch_size)
                    epsilon_theta = self.DDPM(previous_image, FOM_values, timeStep)

                    # algorithm 2 from Ho et al., using posterior variance_t = beta_t
                    epsilon_theta = torch.mul(
                        torch.divide(
                            1 - self.alpha_schedule[t],
                            torch.sqrt(1 - self.alpha_bar_schedule[t]),
                        ),
                        epsilon_theta,
                    )
                    within_parentheses = previous_image - epsilon_theta
                    first_term = torch.mul(
                        torch.divide(1, torch.sqrt(self.alpha_schedule[t])),
                        within_parentheses,
                    )
                    previous_image = first_term + torch.mul(
                        torch.sqrt(1 - self.alpha_schedule[t]), z.to(self.device)
                    )

                x_0 = previous_image
                decoded = self.VAE.decode(x_0)
                dataset.extend(decoded.cpu().numpy())

        dataset = np.array(dataset)
        return dataset

    def create_dataset(self, num_samples, FOM_values):
        dataset = []
        for i in tqdm(range(num_samples // self.batch_size)):
            with torch.no_grad():
                x_T = self.random_generator.sample((self.batch_size,))
                x_T = x_T.view(
                    self.batch_size, self.in_channels, self.height, self.width
                )

                previous_image = x_T.to(self.device)

                # runs diffusion process from pure noise to timestep 0
                for t in range(self.num_steps - 1, -1, -1):
                    print(t)

                    if t > 0:
                        z = self.random_generator.sample((self.batch_size,)).view(
                            self.batch_size, self.in_channels, self.height, self.width
                        )
                    elif t == 0:
                        z = torch.zeros_like(x_T)

                    timeStep = torch.tensor(t).to(self.device)
                    timeStep = timeStep.repeat(self.batch_size)
                    epsilon_theta = self.DDPM(previous_image, FOM_values, timeStep)

                    # algorithm 2 from Ho et al., using posterior variance_t = beta_t
                    epsilon_theta = torch.mul(
                        torch.divide(
                            1 - self.alpha_schedule[t],
                            torch.sqrt(1 - self.alpha_bar_schedule[t]),
                        ),
                        epsilon_theta,
                    )
                    within_parentheses = previous_image - epsilon_theta
                    first_term = torch.mul(
                        torch.divide(1, torch.sqrt(self.alpha_schedule[t])),
                        within_parentheses,
                    )
                    previous_image = first_term + torch.mul(
                        torch.sqrt(1 - self.alpha_schedule[t]), z.to(self.device)
                    )

                x_0 = previous_image
                decoded = self.VAE.decode(x_0)
                dataset.extend(decoded.cpu().numpy())

        dataset = np.array(dataset)
        return dataset

    def sample(self):
        with torch.no_grad():
            x_T = self.random_generator.sample((self.batch_size,)).to(self.device)
            x_T = x_T.view(self.batch_size, self.in_channels, self.height, self.width)
            FOMs = torch.rand(self.batch_size, device=self.device) * 1.8

            previous_image = x_T

            # runs diffusion process from pure noise to timestep 0
            for t in range(self.num_steps - 1, -1, -1):
                z = None
                if t > 0:
                    z = self.random_generator.sample((self.batch_size,)).view(
                        self.batch_size, self.in_channels, self.height, self.width
                    ).to(self.device)
                elif t == 0:
                    z = torch.zeros_like(x_T)

                timeStep = torch.tensor(t).to(self.device)
                timeStep = timeStep.repeat(self.batch_size)
                epsilon_theta = self.DDPM(previous_image, FOMs, timeStep)

                # algorithm 2 from Ho et al., using posterior variance_t = beta_t
                epsilon_theta = torch.mul(
                    torch.divide(
                        1 - self.alpha_schedule[t],
                        torch.sqrt(1 - self.alpha_bar_schedule[t]),
                    ),
                    epsilon_theta,
                )
                within_parentheses = previous_image - epsilon_theta
                first_term = torch.mul(
                    torch.divide(1, torch.sqrt(self.alpha_schedule[t])),
                    within_parentheses,
                )
                previous_image = first_term + torch.mul(
                    torch.sqrt(1 - self.alpha_schedule[t]), z
                )

            x_0 = previous_image
            x_0_grid = torchvision.utils.make_grid(x_0)
            self.logger.add_image(
                "Latent_Generated_Images", x_0_grid, self.global_step
            )
            x_0_decoded = self.VAE.decode(x_0)
            grid = torchvision.utils.make_grid(x_0_decoded)
            self.logger.add_image("Generated_Images", grid, self.global_step)

class Ablation_LDM(pl.LightningModule):

    def __init__(
        self,
        DDPM,
        VAE,
        in_channels,
        batch_size,
        num_steps,
        latent_height,
        latent_width,
        lr,
    ):
        super().__init__()
        self.lr = lr
        self.DDPM = DDPM
        self.VAE = VAE
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.latent_height = latent_height
        self.height = latent_height
        self.latent_width = latent_width
        self.width = latent_width
        self.num_steps = num_steps
        self.random_generator = MultivariateNormal(
            torch.zeros(latent_height * latent_width * in_channels),
            torch.eye(latent_height * latent_width * in_channels),
        )
        self.beta_schedule = self.make_beta_schedule(num_steps, 1e-5, 0.02)
        self.alpha_schedule = torch.from_numpy(
            self.get_alpha_schedule(self.beta_schedule)
        ).to(self.device)
        self.alpha_bar_schedule = self.calculate_alpha_bar(self.alpha_schedule).to(
            self.device
        )

    def training_step(self, batch, batch_idx):
        images, FOMs = batch

        if self.global_step % 300 == 0:
            self.sample()

        # random timestep
        t = torch.randint(0, self.num_steps - 1, (self.batch_size,), device=self.device)

        # generating epsilon_0
        epsilon_sample = self.random_generator.sample((self.batch_size,))
        epsilon_0 = epsilon_sample.view(
            self.batch_size, self.in_channels, self.height, self.width
        )

        # broadcasting alpha_bar_vector
        alpha_bar_vector = self.alpha_bar_schedule[t].float()
        alpha_bar_vector = alpha_bar_vector.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_bar_vector = alpha_bar_vector.expand(
            -1, self.in_channels, self.height, self.width
        )

        # encoding to latent space
        x = images
        with torch.no_grad():
            mu, sigma = self.VAE.encode(x)

        z_reparameterized = mu + torch.multiply(
            sigma, torch.randn_like(sigma, device=self.device)
        )

        # latent_encoding_sample = torch.bernoulli(
        #     latent_encoding_logits
        # )  # I don't think I need a gradient here
        """
        Not doing VAE anymore, just VAE
        """
        # epsilon = torch.randn_like(sigma)
        # z_reparameterized = mu + torch.multiply(sigma, epsilon)

        if self.global_step % 300 == 0:
            latent_grid = torchvision.utils.make_grid(z_reparameterized)
            self.logger.experiment.add_image(
                "True latent grid", latent_grid, self.global_step
            )

        ### Creation of x_t, algorithm 1 of Ho et al.###
        mu = torch.mul(torch.sqrt(alpha_bar_vector), z_reparameterized)
        variance = torch.mul(
            torch.sqrt(torch.ones_like(alpha_bar_vector) - alpha_bar_vector), epsilon_0
        )
        x_t = torch.add(mu, variance)

        # experiment with this
        tNotNormalized = t.float()
        FOMs = FOMs.float()

        # predicted true epsilon in latent space
        epsilon_theta_latent = self.DDPM.forward(x_t, FOMs, tNotNormalized)

        # calculating loss
        loss = F.smooth_l1_loss(epsilon_0, epsilon_theta_latent)
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def make_beta_schedule(self, num_steps, beta_1, beta_T):
        return np.linspace(beta_1, beta_T, num=num_steps)

    def get_alpha_schedule(self, beta_schedule):
        return np.ones_like(beta_schedule) - beta_schedule

    def calculate_alpha_bar(self, alpha_schedule):
        return np.cumprod(alpha_schedule)

    def on_train_start(self):
        self.alpha_schedule = self.alpha_schedule.to(self.device)
        self.alpha_bar_schedule = self.alpha_bar_schedule.to(self.device)
        self.random_generator = MultivariateNormal(
            torch.zeros(
                self.height * self.width * self.in_channels, device=self.device
            ),
            torch.eye(self.height * self.width * self.in_channels, device=self.device),
        )

    def create_dataset_variable_FOM(self, num_samples, start_mean, end_mean, variance):
        dataset = []
        for _ in tqdm(range(num_samples // self.batch_size)):
            with torch.no_grad():
                x_T = self.random_generator.sample((self.batch_size,))
                x_T = x_T.view(
                    self.batch_size, self.in_channels, self.height, self.width
                )

                previous_image = x_T.to(self.device)

                # runs diffusion process from pure noise to timestep 0
                """
                T ranges from 999 to 0, so must linearly scale to start mean to end mean
                This can be accomplished by first inverting t to (num_steps - t), then dividing by 1000 to get in range [0, 1],
                then multiplying this scalar to the difference between start and end mean, then adding to start mean
                """

                difference_mean = end_mean - start_mean
                for t in range(self.num_steps - 1, -1, -1):

                    scaled_t = (self.num_steps - t) / 1000
                    functional_mean = scaled_t * difference_mean + start_mean

                    FOM_values = (
                        variance * torch.randn(self.batch_size, device=self.device)
                        + functional_mean
                    )

                    if t > 0:
                        z = self.random_generator.sample((self.batch_size,)).view(
                            self.batch_size, self.in_channels, self.height, self.width
                        )
                    elif t == 0:
                        z = torch.zeros_like(x_T)

                    timeStep = torch.tensor(t).to(self.device)
                    timeStep = timeStep.repeat(self.batch_size)
                    epsilon_theta = self.DDPM(previous_image, FOM_values, timeStep)

                    # algorithm 2 from Ho et al., using posterior variance_t = beta_t
                    epsilon_theta = torch.mul(
                        torch.divide(
                            1 - self.alpha_schedule[t],
                            torch.sqrt(1 - self.alpha_bar_schedule[t]),
                        ),
                        epsilon_theta,
                    )
                    within_parentheses = previous_image - epsilon_theta
                    first_term = torch.mul(
                        torch.divide(1, torch.sqrt(self.alpha_schedule[t])),
                        within_parentheses,
                    )
                    previous_image = first_term + torch.mul(
                        torch.sqrt(1 - self.alpha_schedule[t]), z.to(self.device)
                    )

                x_0 = previous_image
                decoded = self.VAE.decode(x_0)
                dataset.extend(decoded.cpu().numpy())

        dataset = np.array(dataset)
        return dataset

    def create_dataset(self, num_samples, FOM_values):
        dataset = []
        for i in tqdm(range(num_samples // self.batch_size)):
            with torch.no_grad():
                x_T = self.random_generator.sample((self.batch_size,))
                x_T = x_T.view(
                    self.batch_size, self.in_channels, self.height, self.width
                )

                previous_image = x_T.to(self.device)

                # runs diffusion process from pure noise to timestep 0
                for t in range(self.num_steps - 1, -1, -1):
                    print(t)

                    if t > 0:
                        z = self.random_generator.sample((self.batch_size,)).view(
                            self.batch_size, self.in_channels, self.height, self.width
                        )
                    elif t == 0:
                        z = torch.zeros_like(x_T)

                    timeStep = torch.tensor(t).to(self.device)
                    timeStep = timeStep.repeat(self.batch_size)
                    epsilon_theta = self.DDPM(previous_image, FOM_values, timeStep)

                    # algorithm 2 from Ho et al., using posterior variance_t = beta_t
                    epsilon_theta = torch.mul(
                        torch.divide(
                            1 - self.alpha_schedule[t],
                            torch.sqrt(1 - self.alpha_bar_schedule[t]),
                        ),
                        epsilon_theta,
                    )
                    within_parentheses = previous_image - epsilon_theta
                    first_term = torch.mul(
                        torch.divide(1, torch.sqrt(self.alpha_schedule[t])),
                        within_parentheses,
                    )
                    previous_image = first_term + torch.mul(
                        torch.sqrt(1 - self.alpha_schedule[t]), z.to(self.device)
                    )

                x_0 = previous_image
                decoded = self.VAE.decode(x_0)
                dataset.extend(decoded.cpu().numpy())

        dataset = np.array(dataset)
        return dataset

    def sample(self):
        with torch.no_grad():
            x_T = self.random_generator.sample((self.batch_size,))
            x_T = x_T.view(self.batch_size, self.in_channels, self.height, self.width)
            FOMs = torch.rand(self.batch_size, device=self.device) * 1.8

            previous_image = x_T

            # runs diffusion process from pure noise to timestep 0
            for t in range(self.num_steps - 1, -1, -1):
                z = None
                if t > 0:
                    z = self.random_generator.sample((self.batch_size,)).view(
                        self.batch_size, self.in_channels, self.height, self.width
                    )
                elif t == 0:
                    z = torch.zeros_like(x_T)

                timeStep = torch.tensor(t).to(self.device)
                timeStep = timeStep.repeat(self.batch_size)
                epsilon_theta = self.DDPM(previous_image, FOMs, timeStep)

                # algorithm 2 from Ho et al., using posterior variance_t = beta_t
                epsilon_theta = torch.mul(
                    torch.divide(
                        1 - self.alpha_schedule[t],
                        torch.sqrt(1 - self.alpha_bar_schedule[t]),
                    ),
                    epsilon_theta,
                )
                within_parentheses = previous_image - epsilon_theta
                first_term = torch.mul(
                    torch.divide(1, torch.sqrt(self.alpha_schedule[t])),
                    within_parentheses,
                )
                previous_image = first_term + torch.mul(
                    torch.sqrt(1 - self.alpha_schedule[t]), z
                )

            x_0 = previous_image
            x_0_grid = torchvision.utils.make_grid(x_0)
            self.logger.experiment.add_image(
                "Latent_Generated_Images", x_0_grid, self.global_step
            )
            x_0_decoded = self.VAE.decode(x_0)
            grid = torchvision.utils.make_grid(x_0_decoded)
            self.logger.experiment.add_image("Generated_Images", grid, self.global_step)


class LDM(pl.LightningModule):
    """
    Latent Diffusion Model.
    Contains diffusion model and pretrained VAE.
    Diffusion model is trained in the latent space of the
    pretrained VAE.
    """

    """
    New idea: use FOM to condition the LDM
    """

    def __init__(
        self,
        DDPM,
        VAE,
        in_channels,
        batch_size,
        num_steps,
        latent_height,
        latent_width,
        lr,
    ):
        super().__init__()
        self.lr = lr
        self.DDPM = DDPM
        self.VAE = VAE
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.latent_height = latent_height
        self.height = latent_height
        self.latent_width = latent_width
        self.width = latent_width
        self.num_steps = num_steps
        self.random_generator = MultivariateNormal(
            torch.zeros(latent_height * latent_width * in_channels),
            torch.eye(latent_height * latent_width * in_channels),
        )
        self.beta_schedule = self.make_beta_schedule(num_steps, 1e-5, 0.02)
        self.alpha_schedule = torch.from_numpy(
            self.get_alpha_schedule(self.beta_schedule)
        ).to(self.device)
        self.alpha_bar_schedule = self.calculate_alpha_bar(self.alpha_schedule).to(
            self.device
        )

    def training_step(self, batch, batch_idx):
        images, FOMs = batch

        if self.global_step % 300 == 0:
            self.sample()

        # random timestep
        t = torch.randint(0, self.num_steps - 1, (self.batch_size,), device=self.device)

        # generating epsilon_0
        epsilon_sample = self.random_generator.sample((self.batch_size,))
        epsilon_0 = epsilon_sample.view(
            self.batch_size, self.in_channels, self.height, self.width
        )

        # broadcasting alpha_bar_vector
        alpha_bar_vector = self.alpha_bar_schedule[t].float()
        alpha_bar_vector = alpha_bar_vector.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_bar_vector = alpha_bar_vector.expand(
            -1, self.in_channels, self.height, self.width
        )

        # encoding to latent space
        x = images
        with torch.no_grad():
            mu, sigma = self.VAE.encode(x)

        z_reparameterized = mu + torch.multiply(
            sigma, torch.randn_like(sigma, device=self.device)
        )

        # latent_encoding_sample = torch.bernoulli(
        #     latent_encoding_logits
        # )  # I don't think I need a gradient here
        """
        Not doing VAE anymore, just VAE
        """
        # epsilon = torch.randn_like(sigma)
        # z_reparameterized = mu + torch.multiply(sigma, epsilon)

        if self.global_step % 300 == 0:
            latent_grid = torchvision.utils.make_grid(z_reparameterized)
            self.logger.experiment.add_image(
                "True latent grid", latent_grid, self.global_step
            )

        ### Creation of x_t, algorithm 1 of Ho et al.###
        mu = torch.mul(torch.sqrt(alpha_bar_vector), z_reparameterized)
        variance = torch.mul(
            torch.sqrt(torch.ones_like(alpha_bar_vector) - alpha_bar_vector), epsilon_0
        )
        x_t = torch.add(mu, variance)

        # experiment with this
        tNotNormalized = t.float()
        FOMs = FOMs.float()

        # predicted true epsilon in latent space
        epsilon_theta_latent = self.DDPM.forward(x_t, FOMs, tNotNormalized)

        # calculating loss
        loss = F.smooth_l1_loss(epsilon_0, epsilon_theta_latent)
        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def make_beta_schedule(self, num_steps, beta_1, beta_T):
        return np.linspace(beta_1, beta_T, num=num_steps)

    def get_alpha_schedule(self, beta_schedule):
        return np.ones_like(beta_schedule) - beta_schedule

    def calculate_alpha_bar(self, alpha_schedule):
        return np.cumprod(alpha_schedule)

    def on_train_start(self):
        self.alpha_schedule = self.alpha_schedule.to(self.device)
        self.alpha_bar_schedule = self.alpha_bar_schedule.to(self.device)
        self.random_generator = MultivariateNormal(
            torch.zeros(
                self.height * self.width * self.in_channels, device=self.device
            ),
            torch.eye(self.height * self.width * self.in_channels, device=self.device),
        )

    def create_dataset_variable_FOM(self, num_samples, start_mean, end_mean, variance):
        dataset = []
        for _ in tqdm(range(num_samples // self.batch_size)):
            with torch.no_grad():
                x_T = self.random_generator.sample((self.batch_size,))
                x_T = x_T.view(
                    self.batch_size, self.in_channels, self.height, self.width
                )

                previous_image = x_T.to(self.device)

                # runs diffusion process from pure noise to timestep 0
                """
                T ranges from 999 to 0, so must linearly scale to start mean to end mean
                This can be accomplished by first inverting t to (num_steps - t), then dividing by 1000 to get in range [0, 1],
                then multiplying this scalar to the difference between start and end mean, then adding to start mean
                """

                difference_mean = end_mean - start_mean
                for t in range(self.num_steps - 1, -1, -1):

                    scaled_t = (self.num_steps - t) / 1000
                    functional_mean = scaled_t * difference_mean + start_mean

                    FOM_values = (
                        variance * torch.randn(self.batch_size, device=self.device)
                        + functional_mean
                    )

                    if t > 0:
                        z = self.random_generator.sample((self.batch_size,)).view(
                            self.batch_size, self.in_channels, self.height, self.width
                        )
                    elif t == 0:
                        z = torch.zeros_like(x_T)

                    timeStep = torch.tensor(t).to(self.device)
                    timeStep = timeStep.repeat(self.batch_size)
                    epsilon_theta = self.DDPM(previous_image, FOM_values, timeStep)

                    # algorithm 2 from Ho et al., using posterior variance_t = beta_t
                    epsilon_theta = torch.mul(
                        torch.divide(
                            1 - self.alpha_schedule[t],
                            torch.sqrt(1 - self.alpha_bar_schedule[t]),
                        ),
                        epsilon_theta,
                    )
                    within_parentheses = previous_image - epsilon_theta
                    first_term = torch.mul(
                        torch.divide(1, torch.sqrt(self.alpha_schedule[t])),
                        within_parentheses,
                    )
                    previous_image = first_term + torch.mul(
                        torch.sqrt(1 - self.alpha_schedule[t]), z.to(self.device)
                    )

                x_0 = previous_image
                decoded = self.VAE.decode(x_0)
                dataset.extend(decoded.cpu().numpy())

        dataset = np.array(dataset)
        return dataset

    def create_dataset(self, num_samples, FOM_values):
        dataset = []
        for i in tqdm(range(num_samples // self.batch_size)):
            with torch.no_grad():
                x_T = self.random_generator.sample((self.batch_size,))
                x_T = x_T.view(
                    self.batch_size, self.in_channels, self.height, self.width
                )

                previous_image = x_T.to(self.device)

                # runs diffusion process from pure noise to timestep 0
                for t in range(self.num_steps - 1, -1, -1):
                    print(t)

                    if t > 0:
                        z = self.random_generator.sample((self.batch_size,)).view(
                            self.batch_size, self.in_channels, self.height, self.width
                        )
                    elif t == 0:
                        z = torch.zeros_like(x_T)

                    timeStep = torch.tensor(t).to(self.device)
                    timeStep = timeStep.repeat(self.batch_size)
                    epsilon_theta = self.DDPM(previous_image, FOM_values, timeStep)

                    # algorithm 2 from Ho et al., using posterior variance_t = beta_t
                    epsilon_theta = torch.mul(
                        torch.divide(
                            1 - self.alpha_schedule[t],
                            torch.sqrt(1 - self.alpha_bar_schedule[t]),
                        ),
                        epsilon_theta,
                    )
                    within_parentheses = previous_image - epsilon_theta
                    first_term = torch.mul(
                        torch.divide(1, torch.sqrt(self.alpha_schedule[t])),
                        within_parentheses,
                    )
                    previous_image = first_term + torch.mul(
                        torch.sqrt(1 - self.alpha_schedule[t]), z.to(self.device)
                    )

                x_0 = previous_image
                decoded = self.VAE.decode(x_0)
                dataset.extend(decoded.cpu().numpy())

        dataset = np.array(dataset)
        return dataset

    def sample(self):
        with torch.no_grad():
            x_T = self.random_generator.sample((self.batch_size,))
            x_T = x_T.view(self.batch_size, self.in_channels, self.height, self.width)
            FOMs = torch.rand(self.batch_size, device=self.device) * 1.8

            previous_image = x_T

            # runs diffusion process from pure noise to timestep 0
            for t in range(self.num_steps - 1, -1, -1):
                z = None
                if t > 0:
                    z = self.random_generator.sample((self.batch_size,)).view(
                        self.batch_size, self.in_channels, self.height, self.width
                    )
                elif t == 0:
                    z = torch.zeros_like(x_T)

                timeStep = torch.tensor(t).to(self.device)
                timeStep = timeStep.repeat(self.batch_size)
                epsilon_theta = self.DDPM(previous_image, FOMs, timeStep)

                # algorithm 2 from Ho et al., using posterior variance_t = beta_t
                epsilon_theta = torch.mul(
                    torch.divide(
                        1 - self.alpha_schedule[t],
                        torch.sqrt(1 - self.alpha_bar_schedule[t]),
                    ),
                    epsilon_theta,
                )
                within_parentheses = previous_image - epsilon_theta
                first_term = torch.mul(
                    torch.divide(1, torch.sqrt(self.alpha_schedule[t])),
                    within_parentheses,
                )
                previous_image = first_term + torch.mul(
                    torch.sqrt(1 - self.alpha_schedule[t]), z
                )

            x_0 = previous_image
            x_0_grid = torchvision.utils.make_grid(x_0)
            self.logger.experiment.add_image(
                "Latent_Generated_Images", x_0_grid, self.global_step
            )
            x_0_decoded = self.VAE.decode(x_0)
            grid = torchvision.utils.make_grid(x_0_decoded)
            self.logger.experiment.add_image("Generated_Images", grid, self.global_step)

class VAE_Pytorch(nn.Module):
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
            nn.Conv2d(in_channels, h_dim, kernel_size=(3, 3), padding="same"),
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

        self.perceptual_loss = VGGPerceptualLoss()

    def encode(self, x):
        h = self.encoder(x)
        mu = self.to_mu(h)
        sigma = self.to_sigma(h)
        return mu, sigma

    def decode(self, z):
        return self.decoder(z)

    def training_step(self, batch, batch_idx):
        images, FOMs = batch

        x = images

        mu, sigma = self.encode(images)
        epsilon = torch.randn_like(sigma)
        z_reparameterized = mu + torch.multiply(sigma, epsilon)

        x_hat = self.decode(z_reparameterized)

        kl_divergence = -0.5 * torch.mean(
            1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)
        )
        perceptual_loss = self.perceptual_loss(x, x_hat)

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

class VAE(pl.LightningModule):
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
            nn.Conv2d(in_channels, h_dim, kernel_size=(3, 3), padding="same"),
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

        self.perceptual_loss = VGGPerceptualLoss()

    def encode(self, x):
        h = self.encoder(x)
        mu = self.to_mu(h)
        sigma = self.to_sigma(h)
        return mu, sigma

    def decode(self, z):
        return self.decoder(z)

    def training_step(self, batch, batch_idx):
        images, FOMs = batch

        x = images

        mu, sigma = self.encode(images)
        epsilon = torch.randn_like(sigma)
        z_reparameterized = mu + torch.multiply(sigma, epsilon)

        x_hat = self.decode(z_reparameterized)

        kl_divergence = -0.5 * torch.mean(
            1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)
        )
        perceptual_loss = self.perceptual_loss(x, x_hat)

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

class AblationAttentionUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        UNet_channel=8,
        timeEmbeddingLength=100,
        batch_size=100,
        latent_height=8,
        latent_width=8,
        num_steps=1000,
        FOM_condition_vector_size=100,
        kernel_size=(3, 3),
        conditioning_channel_size=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.UNet_channel = UNet_channel
        self.dim = timeEmbeddingLength
        self.timeEmbeddingLength = timeEmbeddingLength
        self.FOM_embedding_length = FOM_condition_vector_size
        self.batch_size = batch_size
        self.latent_height = latent_height
        self.height = latent_height
        self.latent_width = latent_width
        self.width = latent_width
        self.num_steps = num_steps
        self.embedder = SinusoidalPositionalEmbeddings(self.dim)

        # Encoder
        self.layer1a = ResnetBlockAblation(
            in_channels + conditioning_channel_size,
            UNet_channel,
            kernel_size,
            in_channel_image=in_channels,
        )
        self.selfAttention1 = AttnBlock(UNet_channel)
        self.layer1b = ResnetBlockAblation(
            UNet_channel + conditioning_channel_size,
            UNet_channel,
            kernel_size,
            in_channel_image=UNet_channel,
        )
        self.maxPool = nn.MaxPool2d((2, 2), stride=2)
        self.layer2a = ResnetBlockAblation(
            UNet_channel + conditioning_channel_size,
            UNet_channel * 2,
            kernel_size,
            in_channel_image=UNet_channel,
        )
        self.selfAttention2 = AttnBlock(UNet_channel * 2)
        self.layer2b = ResnetBlockAblation(
            UNet_channel * 2 + conditioning_channel_size,
            UNet_channel * 2,
            kernel_size,
            in_channel_image=UNet_channel * 2,
        )
        self.layer3a = ResnetBlockAblation(
            UNet_channel * 2 + conditioning_channel_size,
            UNet_channel * 2,
            kernel_size,
            in_channel_image=UNet_channel * 2,
        )
        self.selfAttention3 = AttnBlock(UNet_channel * 2)
        self.layer3b = ResnetBlockAblation(
            UNet_channel * 2 + conditioning_channel_size,
            UNet_channel * 2,
            kernel_size,
            in_channel_image=UNet_channel * 2,
        )
        self.layer4a = ResnetBlockAblation(
            2 * (UNet_channel * 2) + conditioning_channel_size,
            UNet_channel * 2,
            kernel_size,
            in_channel_image=2 * (UNet_channel * 2),
        )
        self.selfAttention4 = AttnBlock(UNet_channel * 2)
        self.layer4b = ResnetBlockAblation(
            UNet_channel * 2 + conditioning_channel_size,
            UNet_channel,
            kernel_size,
            in_channel_image=UNet_channel * 2,
        )
        self.layer5a = ResnetBlockAblation(
            2 * (UNet_channel) + conditioning_channel_size,
            UNet_channel,
            kernel_size,
            in_channel_image=2 * (UNet_channel),
        )
        self.selfAttention5 = AttnBlock(UNet_channel)
        self.layer5b = ResnetBlockAblation(
            UNet_channel + conditioning_channel_size,
            UNet_channel,
            kernel_size,
            in_channel_image=UNet_channel,
        )

        self.layer6 = nn.Conv2d(UNet_channel, out_channels, kernel_size=(1, 1))

        # Time Embedders
        self.timeEmbedder1 = TimeEmbedder(
            timeEmbeddingLength, batch_size, latent_height, latent_width
        )
        self.timeEmbedder2 = TimeEmbedder(
            timeEmbeddingLength, batch_size, latent_height // 2, latent_width // 2
        )
        self.timeEmbedder3 = TimeEmbedder(
            timeEmbeddingLength, batch_size, latent_height // 4, latent_width // 4
        )
        self.timeEmbedder4 = TimeEmbedder(
            timeEmbeddingLength, batch_size, latent_height // 2, latent_width // 2
        )
        self.timeEmbedder5 = TimeEmbedder(
            timeEmbeddingLength, batch_size, latent_height, latent_width
        )

        self.FOM_embedder = SinusoidalPositionalEmbeddings(self.FOM_embedding_length)
        # FOM Embedders
        self.FOM_embedder1 = FOM_Conditioner(
            batch_size=self.batch_size,
            height=latent_height,
            width=latent_width,
            embedding_length=self.FOM_embedding_length,
        )
        self.FOM_embedder2 = FOM_Conditioner(
            batch_size=self.batch_size,
            height=latent_height // 2,
            width=latent_width // 2,
            embedding_length=self.FOM_embedding_length,
        )
        self.FOM_embedder3 = FOM_Conditioner(
            batch_size=self.batch_size,
            height=latent_height // 4,
            width=latent_width // 4,
            embedding_length=self.FOM_embedding_length,
        )
        self.FOM_embedder4 = FOM_Conditioner(
            batch_size=self.batch_size,
            height=latent_height // 2,
            width=latent_width // 2,
            embedding_length=self.FOM_embedding_length,
        )
        self.FOM_embedder5 = FOM_Conditioner(
            batch_size=self.batch_size,
            height=latent_height,
            width=latent_width,
            embedding_length=self.FOM_embedding_length,
        )

    def forward(self, x, FOM_values, timeStep):
        FOM_values = 0 #redundant

        # creation of timestep embeddings
        embeddings = self.embedder(timeStep)
        embeddings1 = self.timeEmbedder1(embeddings)
        embeddings2 = self.timeEmbedder2(embeddings)
        embeddings3 = self.timeEmbedder3(embeddings)
        embeddings4 = self.timeEmbedder4(embeddings)
        embeddings5 = self.timeEmbedder5(embeddings)

        # creation of FOM embeddings
        FOM_embeddings1 = torch.zeros_like(embeddings1, device=embeddings.device)
        FOM_embeddings2 = torch.zeros_like(embeddings2, device=embeddings.device)
        FOM_embeddings3 = torch.zeros_like(embeddings3, device=embeddings.device)
        FOM_embeddings4 = torch.zeros_like(embeddings4, device=embeddings.device)
        FOM_embeddings5 = torch.zeros_like(embeddings5, device=embeddings.device)

        # FOM_embeddings = self.FOM_embedder(FOM_values)
        # FOM_embeddings1 = self.FOM_embedder1(FOM_embeddings)
        # FOM_embeddings2 = self.FOM_embedder2(FOM_embeddings)
        # FOM_embeddings3 = self.FOM_embedder3(FOM_embeddings)
        # FOM_embeddings4 = self.FOM_embedder4(FOM_embeddings)
        # FOM_embeddings5 = self.FOM_embedder5(FOM_embeddings)

        # 8x8
        x1 = self.layer1a(x, embeddings1)
        x1 = self.selfAttention1(x1)
        x1 = self.layer1b(x1, embeddings1)
        x2 = self.maxPool(x1)
        # 4x4
        x2 = self.layer2a(x2, embeddings2)
        x2 = self.selfAttention2(x2)
        x2 = self.layer2b(x2, embeddings2)
        x3 = self.maxPool(x2)
        # MIDDLE CONNECTION - 2x2
        x3 = self.layer3a(x3, embeddings3)
        x3 = self.selfAttention3(x3)
        x3 = self.layer3b(x3, embeddings3)
        #
        x4 = F.interpolate(
            x3,
            size=(self.latent_height // 2, self.latent_width // 2),
            mode="bilinear",
            align_corners=False,
        )
        x5 = self.layer4a(torch.cat((x2, x4), dim=1), embeddings4)
        x5 = self.selfAttention4(x5)
        x5 = self.layer4b(x5, embeddings4)
        #
        x6 = F.interpolate(
            x5,
            size=(self.latent_height, self.latent_width),
            mode="bilinear",
            align_corners=False,
        )
        x6 = self.layer5a(torch.cat((x1, x6), dim=1), embeddings5)
        x6 = self.selfAttention5(x6)
        x6 = self.layer5b(x6, embeddings5)

        # final convolutional layer, kernel size (1,1)
        out = self.layer6(x6)

        return out

class AttentionUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        UNet_channel=8,
        timeEmbeddingLength=100,
        batch_size=100,
        latent_height=8,
        latent_width=8,
        num_steps=1000,
        FOM_condition_vector_size=100,
        kernel_size=(3, 3),
        conditioning_channel_size=2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.UNet_channel = UNet_channel
        self.dim = timeEmbeddingLength
        self.timeEmbeddingLength = timeEmbeddingLength
        self.FOM_embedding_length = FOM_condition_vector_size
        self.batch_size = batch_size
        self.latent_height = latent_height
        self.height = latent_height
        self.latent_width = latent_width
        self.width = latent_width
        self.num_steps = num_steps
        self.embedder = SinusoidalPositionalEmbeddings(self.dim)

        # Encoder
        self.layer1a = ResnetBlock(
            in_channels + conditioning_channel_size,
            UNet_channel,
            kernel_size,
            in_channel_image=in_channels,
        )
        self.selfAttention1 = AttnBlock(UNet_channel)
        self.layer1b = ResnetBlock(
            UNet_channel + conditioning_channel_size,
            UNet_channel,
            kernel_size,
            in_channel_image=UNet_channel,
        )
        self.maxPool = nn.MaxPool2d((2, 2), stride=2)
        self.layer2a = ResnetBlock(
            UNet_channel + conditioning_channel_size,
            UNet_channel * 2,
            kernel_size,
            in_channel_image=UNet_channel,
        )
        self.selfAttention2 = AttnBlock(UNet_channel * 2)
        self.layer2b = ResnetBlock(
            UNet_channel * 2 + conditioning_channel_size,
            UNet_channel * 2,
            kernel_size,
            in_channel_image=UNet_channel * 2,
        )
        self.layer3a = ResnetBlock(
            UNet_channel * 2 + conditioning_channel_size,
            UNet_channel * 2,
            kernel_size,
            in_channel_image=UNet_channel * 2,
        )
        self.selfAttention3 = AttnBlock(UNet_channel * 2)
        self.layer3b = ResnetBlock(
            UNet_channel * 2 + conditioning_channel_size,
            UNet_channel * 2,
            kernel_size,
            in_channel_image=UNet_channel * 2,
        )
        self.layer4a = ResnetBlock(
            2 * (UNet_channel * 2) + conditioning_channel_size,
            UNet_channel * 2,
            kernel_size,
            in_channel_image=2 * (UNet_channel * 2),
        )
        self.selfAttention4 = AttnBlock(UNet_channel * 2)
        self.layer4b = ResnetBlock(
            UNet_channel * 2 + conditioning_channel_size,
            UNet_channel,
            kernel_size,
            in_channel_image=UNet_channel * 2,
        )
        self.layer5a = ResnetBlock(
            2 * (UNet_channel) + conditioning_channel_size,
            UNet_channel,
            kernel_size,
            in_channel_image=2 * (UNet_channel),
        )
        self.selfAttention5 = AttnBlock(UNet_channel)
        self.layer5b = ResnetBlock(
            UNet_channel + conditioning_channel_size,
            UNet_channel,
            kernel_size,
            in_channel_image=UNet_channel,
        )

        self.layer6 = nn.Conv2d(UNet_channel, out_channels, kernel_size=(1, 1))

        # Time Embedders
        self.timeEmbedder1 = TimeEmbedder(
            timeEmbeddingLength, batch_size, latent_height, latent_width
        )
        self.timeEmbedder2 = TimeEmbedder(
            timeEmbeddingLength, batch_size, latent_height // 2, latent_width // 2
        )
        self.timeEmbedder3 = TimeEmbedder(
            timeEmbeddingLength, batch_size, latent_height // 4, latent_width // 4
        )
        self.timeEmbedder4 = TimeEmbedder(
            timeEmbeddingLength, batch_size, latent_height // 2, latent_width // 2
        )
        self.timeEmbedder5 = TimeEmbedder(
            timeEmbeddingLength, batch_size, latent_height, latent_width
        )

        self.FOM_embedder = SinusoidalPositionalEmbeddings(self.FOM_embedding_length)
        # FOM Embedders
        self.FOM_embedder1 = FOM_Conditioner(
            batch_size=self.batch_size,
            height=latent_height,
            width=latent_width,
            embedding_length=self.FOM_embedding_length,
        )
        self.FOM_embedder2 = FOM_Conditioner(
            batch_size=self.batch_size,
            height=latent_height // 2,
            width=latent_width // 2,
            embedding_length=self.FOM_embedding_length,
        )
        self.FOM_embedder3 = FOM_Conditioner(
            batch_size=self.batch_size,
            height=latent_height // 4,
            width=latent_width // 4,
            embedding_length=self.FOM_embedding_length,
        )
        self.FOM_embedder4 = FOM_Conditioner(
            batch_size=self.batch_size,
            height=latent_height // 2,
            width=latent_width // 2,
            embedding_length=self.FOM_embedding_length,
        )
        self.FOM_embedder5 = FOM_Conditioner(
            batch_size=self.batch_size,
            height=latent_height,
            width=latent_width,
            embedding_length=self.FOM_embedding_length,
        )

    def forward(self, x, FOM_values, timeStep):
        FOM_values = (FOM_values - 0.713) * 2000

        # creation of timestep embeddings
        embeddings = self.embedder(timeStep)
        embeddings1 = self.timeEmbedder1(embeddings)
        embeddings2 = self.timeEmbedder2(embeddings)
        embeddings3 = self.timeEmbedder3(embeddings)
        embeddings4 = self.timeEmbedder4(embeddings)
        embeddings5 = self.timeEmbedder5(embeddings)

        # creation of FOM embeddings
        FOM_embeddings = self.FOM_embedder(FOM_values)
        FOM_embeddings1 = self.FOM_embedder1(FOM_embeddings)
        FOM_embeddings2 = self.FOM_embedder2(FOM_embeddings)
        FOM_embeddings3 = self.FOM_embedder3(FOM_embeddings)
        FOM_embeddings4 = self.FOM_embedder4(FOM_embeddings)
        FOM_embeddings5 = self.FOM_embedder5(FOM_embeddings)

        # 8x8
        x1 = self.layer1a(x, embeddings1, FOM_embeddings1)
        x1 = self.selfAttention1(x1)
        x1 = self.layer1b(x1, embeddings1, FOM_embeddings1)
        x2 = self.maxPool(x1)
        # 4x4
        x2 = self.layer2a(x2, embeddings2, FOM_embeddings2)
        x2 = self.selfAttention2(x2)
        x2 = self.layer2b(x2, embeddings2, FOM_embeddings2)
        x3 = self.maxPool(x2)
        # MIDDLE CONNECTION - 2x2
        x3 = self.layer3a(x3, embeddings3, FOM_embeddings3)
        x3 = self.selfAttention3(x3)
        x3 = self.layer3b(x3, embeddings3, FOM_embeddings3)
        #
        x4 = F.interpolate(
            x3,
            size=(self.latent_height // 2, self.latent_width // 2),
            mode="bilinear",
            align_corners=False,
        )
        x5 = self.layer4a(torch.cat((x2, x4), dim=1), embeddings4, FOM_embeddings4)
        x5 = self.selfAttention4(x5)
        x5 = self.layer4b(x5, embeddings4, FOM_embeddings4)
        #
        x6 = F.interpolate(
            x5,
            size=(self.latent_height, self.latent_width),
            mode="bilinear",
            align_corners=False,
        )
        x6 = self.layer5a(torch.cat((x1, x6), dim=1), embeddings5, FOM_embeddings5)
        x6 = self.selfAttention5(x6)
        x6 = self.layer5b(x6, embeddings5, FOM_embeddings5)

        # final convolutional layer, kernel size (1,1)
        out = self.layer6(x6)

        return out

class ResnetBlockAblation(nn.Module):
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

    def forward(self, x, time_step_embeddings):
        xCopy = x

        x = torch.cat((x, time_step_embeddings), dim=1)

        x = self.layer1(x)
        x = self.SiLU(x)
        x = self.layer2(x)
        xCopy = self.resizeInput(xCopy)
        x = x + xCopy
        x = self.SiLU(x)

        return x

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


class TimeEmbedder(nn.Module):
    def __init__(self, timeEmbeddingLength, batch_size, height, width):
        super().__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.Linear1 = nn.Linear(timeEmbeddingLength, height * width)
        self.SiLU = nn.SiLU()

    def forward(self, x):
        x = self.Linear1(x)
        x = self.SiLU(x)
        x = x.view(self.batch_size, 1, self.height, self.width)

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
