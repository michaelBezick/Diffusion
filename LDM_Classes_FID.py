from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import math as m
import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms.functional import center_crop
from einops import rearrange
from einops.layers.torch import Rearrange
import pytorch_lightning as pl
from torch import Tensor, optim
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
import random
import tensorflow as tf
import time

class LDM(pl.LightningModule):
    '''
    Latent Diffusion Model.
    Contains diffusion model and pretrained VAE.
    Diffusion model is trained in the latent space of the
    pretrained VAE.
    '''
    def __init__(self, DDPM, VAE_GAN, in_channels, batch_size, num_steps, height, width, lr):
        super().__init__()
        self.lr = lr
        self.DDPM = DDPM
        self.VAE_GAN = VAE_GAN
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.num_steps = num_steps
        self.random_generator = MultivariateNormal(torch.zeros(height * width * in_channels),
                                                   torch.eye(height * width * in_channels))
        self.beta_schedule = self.make_beta_schedule(num_steps, 1e-5, 0.02)
        self.alpha_schedule = torch.from_numpy(self.get_alpha_schedule(self.beta_schedule)).to(self.device)
        self.alpha_bar_schedule = self.calculate_alpha_bar(self.alpha_schedule).to(self.device)

    def training_step(self, batch):
        if self.global_step % 300 == 0:
            self.sample()
        #random timestep
        t = torch.randint(0, self.num_steps - 1, (self.batch_size,), device = self.device)

        #generating epsilon_0
        epsilon_sample = self.random_generator.sample(
            (self.batch_size,))
        epsilon_0 = epsilon_sample.view(
            self.batch_size, self.in_channels, self.height, self.width)

        #broadcasting alpha_bar_vector
        alpha_bar_vector = self.alpha_bar_schedule[t].float()
        alpha_bar_vector = alpha_bar_vector.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_bar_vector = alpha_bar_vector.expand(-1, self.in_channels, self.height, self.width)

        #encoding to latent space
        x = batch
        with torch.no_grad():
            mu, sigma = self.VAE_GAN.vae.encode(x)

        epsilon = torch.randn_like(sigma)
        z_reparameterized = mu + torch.multiply(sigma, epsilon)
        
        if self.global_step % 300 == 0:
            latent_grid = torchvision.utils.make_grid(z_reparameterized)
            self.logger.experiment.add_image("True latent grid", latent_grid, self.global_step)

        ### Creation of x_t, algorithm 1 of Ho et al.###
        mu = torch.mul(torch.sqrt(alpha_bar_vector), z_reparameterized)
        variance = torch.mul(torch.sqrt(torch.ones_like(
            alpha_bar_vector) - alpha_bar_vector), epsilon_0)
        x_t = torch.add(mu, variance)

        tNotNormalized = t

        #predicted true epsilon in latent space
        epsilon_theta_latent = self.DDPM.forward(x_t, tNotNormalized)

        #calculating loss
        loss = F.smooth_l1_loss(epsilon_0, epsilon_theta_latent)
        self.log("train_loss", loss)

        return loss
    def configure_optimizers(self):
        #optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay = 0.01)
        #return {"optimizer": optimizer, "clip_gradients": 1.0}
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
        self.random_generator = MultivariateNormal(torch.zeros(self.height * self.width * self.in_channels, device = self.device),
                                                   torch.eye(self.height * self.width * self.in_channels, device = self.device))

    def create_dataset(self):
        dataset = []
        i = 0
        while len(dataset) < 10_000:
            print(i)
            if (i == 0):
                    start_time = time.time()
            if (i == 1):
                    end_time = time.time()
                    print(end_time - start_time)
            i = i + 1
            with torch.no_grad():
                    x_T = self.random_generator.sample((self.batch_size,))
                    x_T = x_T.view(self.batch_size, self.in_channels, self.height, self.width)

                    previous_image = x_T.to(self.device)

                    #runs diffusion process from pure noise to timestep 0
                    for t in range(self.num_steps - 1, -1, -1):

                        if (t > 0):
                            z = self.random_generator.sample((self.batch_size,)).view(self.batch_size, self.in_channels, self.height, self.width).to(self.device)
                        elif (t == 0):
                            z = torch.zeros_like(x_T).to(self.device)

                        timeStep = torch.tensor(t).to(self.device)
                        timeStep = timeStep.repeat(self.batch_size)
                        epsilon_theta = self.DDPM(previous_image, timeStep)

                        #algorithm 2 from Ho et al., using posterior variance_t = beta_t
                        epsilon_theta = torch.mul(torch.divide(1 - self.alpha_schedule[t], torch.sqrt(1 - self.alpha_bar_schedule[t])), epsilon_theta)
                        within_parentheses = previous_image - epsilon_theta
                        first_term = torch.mul(torch.divide(1, torch.sqrt(self.alpha_schedule[t])), within_parentheses)
                        previous_image = first_term + torch.mul(torch.sqrt(1 - self.alpha_schedule[t]), z)

                    x_0 = previous_image
                    x_decoded = self.VAE_GAN.vae.decode(x_0)
                    for image in x_decoded.cpu().numpy():
                        dataset.append(image)

        dataset = np.array(dataset)
        np.save('generated_dataset.npy', dataset)    

    def sample(self):
        with torch.no_grad():
                x_T = self.random_generator.sample((self.batch_size,))
                x_T = x_T.view(self.batch_size, self.in_channels, self.height, self.width)

                previous_image = x_T

                #runs diffusion process from pure noise to timestep 0
                for t in range(self.num_steps - 1, -1, -1):

                    if (t > 0):
                        z = self.random_generator.sample((self.batch_size,)).view(self.batch_size, self.in_channels, self.height, self.width)
                    elif (t == 0):
                        z = torch.zeros_like(x_T)

                    timeStep = torch.tensor(t).to(self.device)
                    timeStep = timeStep.repeat(self.batch_size)
                    epsilon_theta = self.DDPM(previous_image, timeStep)

                    #algorithm 2 from Ho et al., using posterior variance_t = beta_t
                    epsilon_theta = torch.mul(torch.divide(1 - self.alpha_schedule[t], torch.sqrt(1 - self.alpha_bar_schedule[t])), epsilon_theta)
                    within_parentheses = previous_image - epsilon_theta
                    first_term = torch.mul(torch.divide(1, torch.sqrt(self.alpha_schedule[t])), within_parentheses)
                    previous_image = first_term + torch.mul(torch.sqrt(1 - self.alpha_schedule[t]), z)

                x_0 = previous_image
                x_0_grid = torchvision.utils.make_grid(x_0)
                self.logger.experiment.add_image("Latent_Generated_Images", x_0_grid, self.global_step)
                x_0_decoded = self.VAE_GAN.vae.decode(x_0)
                grid = torchvision.utils.make_grid(x_0_decoded)
                self.logger.experiment.add_image("Generated_Images", grid, self.global_step)


class VAE(pl.LightningModule):
    def __init__(self, in_channels = 1, h_dim = 32, lr = 1e-3, batch_size = 100):
        super().__init__()

        self.batch_size = batch_size

        self.lr = lr

        self.attention1E = AttnBlock(h_dim)#LinearSelfAttention(h_dim, heads, dim_head)
        self.attention2E = AttnBlock(h_dim) #LinearSelfAttention(h_dim, heads, dim_head)
        self.resnet1E = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet2E = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet3E = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet4E = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet5E = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet6E = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.maxPool = nn.MaxPool2d((2, 2), 2)

        self.encoder = nn.Sequential(nn.Conv2d(in_channels, h_dim, kernel_size = (3, 3), padding = 'same'),
                                    nn.SiLU(),
                                    self.resnet1E,
                                    self.maxPool, # 32 x 32
                                    self.resnet2E,
                                    self.maxPool, # 16 x 16
                                    self.resnet3E,
                                    self.attention1E,
                                    self.resnet4E,
                                    self.maxPool, # 8 x 8
                                    self.resnet5E,
                                    self.attention2E,
                                    self.resnet6E,
                                    )
        
        self.to_mu = nn.Conv2d(h_dim, 1, (1, 1))

        self.to_sigma = nn.Conv2d(h_dim, 1, (1, 1))

        self.attention1D = AttnBlock(h_dim) #LinearSelfAttention(h_dim, heads, dim_head)
        self.attention2D = AttnBlock(h_dim) #LinearSelfAttention(h_dim, heads, dim_head)
        self.resnet1D = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet2D = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet3D = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet4D = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet5D = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet6D = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        
        self.decoder = nn.Sequential(nn.Conv2d(1, h_dim, (1, 1)),
                                     self.resnet1D,
                                     self.attention1D,
                                     self.resnet2D,
                                     nn.ConvTranspose2d(h_dim, h_dim, (2, 2), 2), # 16 x 16
                                     self.resnet3D,
                                     self.attention2D,
                                     self.resnet4D,
                                     nn.ConvTranspose2d(h_dim, h_dim, (2, 2), 2), # 32 x 32
                                     self.resnet5D,
                                     nn.ConvTranspose2d(h_dim, h_dim, (2,2), 2), # 64 x 64
                                     self.resnet6D,
                                     nn.Conv2d(h_dim, 1, (1, 1))
                                     )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.to_mu(h)
        sigma = self.to_sigma(h)
        return mu, sigma
    
    def decode(self, z):
        return self.decoder(z)
"""
class VAE(pl.LightningModule):
    '''
    Variational autoencoder with UNet structure.
    '''
    def __init__(self, in_channels = 1, h_dim = 32, lr = 1e-3, batch_size = 100):
        super().__init__()

        self.batch_size = batch_size

        self.lr = lr

        self.attention1E = AttnBlock(h_dim)
        self.attention2E = AttnBlock(h_dim)
        self.resnet1E = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet2E = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet3E = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet4E = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet5E = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet6E = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.maxPool = nn.MaxPool2d((2, 2), 2)

        self.encoder = nn.Sequential(nn.Conv2d(in_channels, h_dim, kernel_size = (3, 3), padding = 'same'),
                                    nn.SiLU(),
                                    self.resnet1E,
                                    self.maxPool, # 32 x 32
                                    self.resnet2E,
                                    self.maxPool, # 16 x 16
                                    self.resnet3E,
                                    self.attention1E,
                                    self.resnet4E,
                                    self.maxPool, # 8 x 8
                                    self.resnet5E,
                                    self.attention2E,
                                    self.resnet6E,
                                    nn.Conv2d(h_dim, 1, kernel_size=1),
                                    )
        
        '''
        #self.to_mu = nn.Conv2d(h_dim, 1, (1, 1))
        self.to_logits = nn.Sequential(nn.Linear(64, 256),
                                       nn.GroupNorm(8, 256),
                                       nn.ReLU(),
                                       nn.Linear(256, 25),
                                       )
        '''
        #self.to_sigma = nn.Conv2d(h_dim, 1, (1, 1))

        self.attention1D = AttnBlock(h_dim)
        self.attention2D = AttnBlock(h_dim)
        self.resnet1D = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet2D = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet3D = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet4D = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet5D = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        self.resnet6D = ResnetBlockVAE(h_dim, h_dim, (3,3), h_dim)
        
        '''
        self.to_decoder = nn.Sequential(nn.Linear(25, 256),
                                        nn.GroupNorm(8, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 64),
                                        nn.GroupNorm(8, 64),
                                        nn.ReLU(),
                                        )
        '''
                                        
        self.decoder = nn.Sequential(nn.Conv2d(1, h_dim, (1, 1)),
                                     self.resnet1D,
                                     self.attention1D,
                                     self.resnet2D,
                                     nn.ConvTranspose2d(h_dim, h_dim, (2, 2), 2), # 16 x 16
                                     self.resnet3D,
                                     self.attention2D,
                                     self.resnet4D,
                                     nn.ConvTranspose2d(h_dim, h_dim, (2, 2), 2), # 32 x 32
                                     self.resnet5D,
                                     nn.ConvTranspose2d(h_dim, h_dim, (2,2), 2), # 64 x 64
                                     self.resnet6D,
                                     nn.Conv2d(h_dim, 1, (1, 1))
                                     )

    def encode(self, x):
        logits = self.encoder(x)
        return logits.view(-1, 64)
    
    def decode(self, z):
        return self.decoder(z.view(-1, 1, 8, 8))
"""

class VAE_GAN(pl.LightningModule):
    def __init__(self, a, b, c, d, in_channels = 1, h_dim = 32, heads = 4, dim_head = 4, lr = 1e-3, patch_dim = 16, image_dim = 64, batch_size = 100):
        super().__init__()
        self.automatic_optimization = False

        self.image_dim = image_dim
        self.patch_dim = patch_dim
        self.num_patches = (image_dim - patch_dim + 1) ** 2

        self.batch_size = batch_size

        self.lr = lr
        self.discriminator = Discriminator(h_dim = h_dim)
        self.vae = VAE(in_channels, h_dim, lr, batch_size)

        self.a = a
        self.b = b
        self.c = c
        self.d = d

        self.perceptual_loss = VGGPerceptualLoss()
        self.perceptual_loss.eval()
    
    def training_step(self, batch, batch_idx):
        x = batch

        opt_VAE, opt_Discriminator = self.optimizers()

        #training VAE
        self.toggle_optimizer(opt_VAE)

        mu, sigma = self.vae.encode(x)

        epsilon = torch.randn_like(sigma)

        z_reparameterized = mu + torch.multiply(sigma, epsilon)

        x_hat = self.vae.decode(z_reparameterized)

        #logging generated images
        sample_imgs_generated = x_hat[:30]
        sample_imgs_original = x[:30]
        gridGenerated = torchvision.utils.make_grid(sample_imgs_generated)
        gridOriginal = torchvision.utils.make_grid(sample_imgs_original)

        if self.global_step % 10 == 0:
            self.logger.experiment.add_image("Generated_images", gridGenerated, self.global_step)
            self.logger.experiment.add_image("Original_images", gridOriginal, self.global_step)

        #all fake
        valid = torch.ones(x_hat.size(0), 1, device=self.device)

        adv_loss = self.adversarial_loss(x_hat, valid) * self.d #* min(1.0, self.global_step / 1000)

        kl_divergence = -0.5 * torch.mean(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)) * self.c

        #reconstruction_loss = F.mse_loss(x_hat, x) * self.a

        perceptual_loss_value = self.perceptual_loss(x_hat, x) * self.b

        total_loss = kl_divergence + perceptual_loss_value + adv_loss
        #self.log("reconstruction_loss", reconstruction_loss)
        self.log("perceptual_loss", perceptual_loss_value)
        self.log("kl_divergence", kl_divergence)
        self.log("adversarial_loss", adv_loss)
        self.log("train_loss", total_loss)

        self.manual_backward(total_loss)
        opt_VAE.step()
        opt_VAE.zero_grad()
        self.untoggle_optimizer(opt_VAE)
 
        
        #training discriminator
        self.toggle_optimizer(opt_Discriminator)
        fake = torch.zeros(x_hat.size(0), 1, device=self.device)
        valid = torch.ones(x.size(0), 1, device=self.device)
        fake_loss = self.adversarial_loss(x_hat.detach(), fake)
        real_loss = self.adversarial_loss(x.detach(), valid)

        average_loss = (fake_loss + real_loss) / 2
        self.log("discriminator_loss", average_loss)
        self.manual_backward(average_loss)
        opt_Discriminator.step()
        opt_Discriminator.zero_grad()
        self.untoggle_optimizer(opt_Discriminator)

        #training discriminator
        self.toggle_optimizer(opt_Discriminator)
        fake = torch.zeros(x_hat.size(0), 1, device=self.device)
        fake_loss = self.adversarial_loss(x_hat.detach(), fake)
        real_loss = self.adversarial_loss(x.detach(), valid)

        average_loss = (fake_loss + real_loss) / 2
        self.log("discriminator_loss", average_loss)
        self.manual_backward(average_loss)
        opt_Discriminator.step()
        opt_Discriminator.zero_grad()
        self.untoggle_optimizer(opt_Discriminator)
        
        
            
    def adversarial_loss(self, y_hat, y):
        i = random.randint(0, self.image_dim - self.patch_dim)
        j = random.randint(0, self.image_dim - self.patch_dim)
        
        loss = F.binary_cross_entropy(self.discriminator(y_hat[:, :, i:i+16, j:j+16]), y) #[:, :, i:i+8, j:j+8]

        return loss
   
    def configure_optimizers(self):
        opt_VAE = torch.optim.Adam(self.vae.parameters(), lr = self.lr) #weight_decay=0.01
        opt_Discriminator = torch.optim.Adam(self.discriminator.parameters(), lr = self.lr)
        return [{"optimizer": opt_VAE}, 
                {"optimizer": opt_Discriminator}] #"clip_gradients": 1.0
"""
class VAE_GAN(pl.LightningModule):
    '''
    Houses both VAE and Discriminator.
    VAE trained with both perceptual and adversarial loss.
    '''
    def __init__(self, a, b, c, d, in_channels = 1, h_dim = 32, lr = 1e-3, patch_dim = 32, image_dim = 64, batch_size = 100):
        super().__init__()
        self.automatic_optimization = False

        self.image_dim = image_dim #length or width of original square image
        self.patch_dim = patch_dim #length or width of intended square patch
        self.num_patches = (image_dim - patch_dim + 1) ** 2

        self.batch_size = batch_size

        self.lr = lr
        self.discriminator = Discriminator()
        self.vae = VAE(in_channels, h_dim, lr, batch_size)

        self.a = a
        self.b = b
        self.c = c
        self.d = d

        with open('VGGnet.json', 'r') as file:
            data = file.read()

        self.FOM_model = tf.keras.models.model_from_json(data)
        self.FOM_model.load_weights('VGGnet_weights.h5')

        self.FOM_min = 0.64010394
        self.FOM_max = 1.850003

        self.perceptual_loss = VGGPerceptualLoss()
        self.perceptual_loss.eval()

        self.sigmoid = nn.Sigmoid()
    
    def training_step(self, batch, batch_idx):
        x = batch

        opt_VAE, opt_Discriminator = self.optimizers()
        scheduler = self.lr_schedulers()

        #training VAE
        self.toggle_optimizer(opt_VAE)

        logits = self.vae.encode(x)

        probabilities = self.sigmoid(logits)

        #bernoulli sampling
        binary_vector = torch.bernoulli(probabilities)

        #straight through gradient copying
        binary_vector_with_gradient = (binary_vector - probabilities).detach() + probabilities

        x_hat = self.vae.decode(binary_vector_with_gradient)

        #logging generated images
        sample_imgs_generated = x_hat[:30]
        sample_imgs_original = x[:30]
        gridGenerated = torchvision.utils.make_grid(sample_imgs_generated)
        gridOriginal = torchvision.utils.make_grid(sample_imgs_original)

        if self.global_step % 10 == 0:
            self.logger.experiment.add_image("Generated_images", gridGenerated, self.global_step)
            self.logger.experiment.add_image("Original_images", gridOriginal, self.global_step)

        #all fake
        valid = torch.ones(x_hat.size(0), 1, device=self.device)

        adv_loss = self.adversarial_loss(x_hat, valid) * self.d

        #kl_divergence = -0.5 * torch.mean(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)) * self.c

        reconstruction_loss = F.mse_loss(x_hat, x) * self.a

        perceptual_loss_value = self.perceptual_loss(x_hat, x) * self.b

        total_loss = perceptual_loss_value + adv_loss + reconstruction_loss
        self.log("reconstruction_loss", reconstruction_loss)
        self.log("perceptual_loss", perceptual_loss_value)
        #self.log("kl_divergence", kl_divergence)
        self.log("adversarial_loss", adv_loss)
        self.log("energy_loss", energy_loss)
        self.log("train_loss", total_loss)

        self.manual_backward(total_loss)
        opt_VAE.step()
        scheduler.step()
        opt_VAE.zero_grad()
        self.untoggle_optimizer(opt_VAE)
        
        #training discriminator
        self.toggle_optimizer(opt_Discriminator)
        fake = torch.zeros(x_hat.size(0), 1, device=self.device)
        valid = torch.ones(x.size(0), 1, device=self.device)
        fake_loss = self.adversarial_loss(x_hat.detach(), fake)
        real_loss = self.adversarial_loss(x.detach(), valid)

        average_loss = (fake_loss + real_loss) / 2
        self.log("discriminator_loss", average_loss)
        self.manual_backward(average_loss)
        opt_Discriminator.step()
        opt_Discriminator.zero_grad()
        self.untoggle_optimizer(opt_Discriminator)

        #training discriminator
        self.toggle_optimizer(opt_Discriminator)
        fake = torch.zeros(x_hat.size(0), 1, device=self.device)
        fake_loss = self.adversarial_loss(x_hat.detach(), fake)
        real_loss = self.adversarial_loss(x.detach(), valid)

        average_loss = (fake_loss + real_loss) / 2
        self.log("discriminator_loss", average_loss)
        self.manual_backward(average_loss)
        opt_Discriminator.step()
        opt_Discriminator.zero_grad()
        self.untoggle_optimizer(opt_Discriminator)
        

    def adversarial_loss(self, y_hat, y):
        i = random.randint(0, self.image_dim - self.patch_dim)
        j = random.randint(0, self.image_dim - self.patch_dim)
        
        loss = F.binary_cross_entropy(self.discriminator(y_hat[:, :, i:i+self.patch_dim, j:j+self.patch_dim]), y) #[:, :, i:i+8, j:j+8]

        return loss
   
    def configure_optimizers(self):
        opt_VAE = torch.optim.Adam(self.vae.parameters(), lr = self.lr) #weight_decay=0.01
        opt_Discriminator = torch.optim.Adam(self.discriminator.parameters(), lr = self.lr)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer = opt_VAE, lr_lambda = lambda epoch: self.warmup_lr_schedule(epoch))

        return ({"optimizer": opt_VAE, "lr_scheduler": scheduler}, 
                 {"optimizer": opt_Discriminator}
                 )
    
    def warmup_lr_schedule(self, epoch):
        warmup_epochs = 400
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1
    def quboEnergy(self, x, H):
        '''
        Computes the energy for the specified Quadratic Unconstrained Binary Optimization (QUBO) system.

        Parameters:
            x (torch.Tensor) : Tensor of shape (batch_size, num_dim) representing the configuration of the system.
            H (torch.Tensor) : Tensor of shape (batch_size, num_dim, num_dim) representing the QUBO matrix.

        Returns:
            torch.Tensor : The energy for each configuration in the batch.
        '''
        if len(x.shape) == 1 and len(H.shape) == 2:
            return torch.einsum("i,ij,j->", x, H, x)
        elif len(x.shape) == 2 and len(H.shape) == 3:
            return torch.einsum("bi,bij,bj->b", x, H, x)
        elif len(x.shape) == 2 and len(H.shape) == 2:
            return torch.einsum("bi,ij,bj->b", x, H, x)
        else:
            raise ValueError(
                "Invalid shapes for x and H. x must be of shape (batch_size, num_dim) and H must be of shape (batch_size, num_dim, num_dim)."
            )
""" 
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:8].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[8:14].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[14:20].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
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

class Discriminator(nn.Module):
    def __init__(self, in_channels = 1, h_dim = 64):
        super().__init__()
        self.in_channels = in_channels
        self.h_dim = h_dim
        
        
        self.layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        
        '''
        self.layers = nn.Sequential(
            nn.Conv2d(1, h_dim, kernel_size=4, stride=2, padding=1), #16x16
            nn.BatchNorm2d(h_dim),
            nn.SiLU(),
            nn.Conv2d(h_dim, h_dim * 2, kernel_size=4, stride=2, padding=1), #8x8
            nn.BatchNorm2d(h_dim * 2),
            nn.SiLU(),
            nn.Conv2d(h_dim * 2, h_dim * 4, kernel_size=4, stride=2, padding=1), #4x4
            nn.BatchNorm2d(h_dim * 4),
            nn.SiLU(),
            nn.Conv2d(h_dim * 4, h_dim * 2, kernel_size=4, stride=2, padding=1), #2x2
            nn.BatchNorm2d(h_dim * 2),
            nn.SiLU(),
            nn.Conv2d(h_dim * 2, h_dim, kernel_size=4, stride=2, padding=1), #1x1
            nn.BatchNorm2d(h_dim),
            nn.SiLU(),
        )

        
        self.fc = nn.Sequential(
            nn.Linear(h_dim, 1),
            nn.Sigmoid(),
        )
        '''
    def forward(self, x):
        x = self.layers(x.reshape(-1, 32 * 32))

        return x

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = models.vgg19(pretrained=True).features
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.feature_extractor = nn.Sequential(
            *list(self.vgg.children())[:4]
        )

    def forward(self, x, y):
        x = torch.cat((x, x, x), dim=1)
        y = torch.cat((y, y, y), dim=1)
        fx = self.feature_extractor(x)
        fy = self.feature_extractor(y)

        loss = F.mse_loss(fx, fy)

        return loss


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=8, num_channels=self.in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


class ResnetBlockVAE(nn.Module):
    #Employs intra block skip connection, which needs a (1, 1) convolution to scale to out_channels

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
            nn.Conv2d(in_channels, out_channels, kernel_size, padding = 'same', bias = False),
            nn.GroupNorm(8, out_channels),
        )

    def forward(self, x):
        x = self.layer(x)

        return x

class AttentionUNet(pl.LightningModule):
    def __init__(self, in_channels=1, out_channels=1, UNet_channel = 8, timeEmbeddingLength=16, batch_size=100,
                 height=8, width=8, num_steps=1000, condition_vector_size=16, kernel_size=(3, 3)):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.UNet_channel = UNet_channel
        self.dim = timeEmbeddingLength
        self.timeEmbeddingLength = timeEmbeddingLength
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.num_steps = num_steps
        self.embedder = SinusoidalPositionalEmbeddings(self.dim)

        # Encoder
        self.layer1a = ResnetBlock(in_channels + 1, UNet_channel, kernel_size, in_channel_image = in_channels)
        self.selfAttention1 = AttnBlock(UNet_channel)
        self.layer1b = ResnetBlock(UNet_channel + 1, UNet_channel, kernel_size, in_channel_image = UNet_channel)
        self.maxPool = nn.MaxPool2d((2,2), stride = 2)
        self.layer2a = ResnetBlock(UNet_channel + 1, UNet_channel * 2, kernel_size, in_channel_image = UNet_channel)
        self.selfAttention2 = AttnBlock(UNet_channel * 2)
        self.layer2b = ResnetBlock(UNet_channel * 2 + 1, UNet_channel * 2, kernel_size, in_channel_image = UNet_channel * 2)
        self.layer3a = ResnetBlock(UNet_channel * 2 + 1, UNet_channel * 2, kernel_size, in_channel_image = UNet_channel * 2)
        self.selfAttention3 = AttnBlock(UNet_channel * 2)
        self.layer3b = ResnetBlock(UNet_channel * 2 + 1, UNet_channel * 2, kernel_size, in_channel_image = UNet_channel * 2)
        self.layer4a = ResnetBlock(2 * (UNet_channel * 2) + 1, UNet_channel * 2, kernel_size, in_channel_image = 2 * (UNet_channel * 2))
        self.selfAttention4 = AttnBlock(UNet_channel * 2)
        self.layer4b = ResnetBlock(UNet_channel * 2 + 1, UNet_channel, kernel_size, in_channel_image = UNet_channel * 2)
        self.layer5a = ResnetBlock(2 * (UNet_channel) + 1, UNet_channel, kernel_size, in_channel_image = 2 * (UNet_channel))
        self.selfAttention5 = AttnBlock(UNet_channel)
        self.layer5b = ResnetBlock(UNet_channel + 1, UNet_channel, kernel_size, in_channel_image = UNet_channel)

        self.layer6 = nn.Conv2d(UNet_channel, out_channels, kernel_size = (1, 1))

         #Time Embedders
        self.timeEmbedder1 = TimeEmbedder(timeEmbeddingLength, batch_size, height, width)
        self.timeEmbedder2 = TimeEmbedder(timeEmbeddingLength, batch_size, height // 2, width // 2)
        self.timeEmbedder3 = TimeEmbedder(timeEmbeddingLength, batch_size, height // 4, width // 4)
        self.timeEmbedder4 = TimeEmbedder(timeEmbeddingLength, batch_size, height // 2, width // 2)
        self.timeEmbedder5 = TimeEmbedder(timeEmbeddingLength, batch_size, height, width)

    def forward(self, x, timeStep):

        #creation of timestep embeddings
        embeddings = self.embedder(timeStep)
        embeddings1 = self.timeEmbedder1(embeddings)
        embeddings2 = self.timeEmbedder2(embeddings)
        embeddings3 = self.timeEmbedder3(embeddings)
        embeddings4 = self.timeEmbedder4(embeddings)
        embeddings5 = self.timeEmbedder5(embeddings)
        
        #8x8
        x1 = self.layer1a(x, embeddings1)
        x1 = self.selfAttention1(x1)
        x1 = self.layer1b(x1, embeddings1)
        x2 = self.maxPool(x1)
        #4x4
        x2 = self.layer2a(x2, embeddings2)
        x2 = self.selfAttention2(x2)
        x2 = self.layer2b(x2, embeddings2)
        x3 = self.maxPool(x2)
        #MIDDLE CONNECTION - 2x2
        x3 = self.layer3a(x3, embeddings3)
        x3 = self.selfAttention3(x3)
        x3 = self.layer3b(x3, embeddings3)
        #
        x4 = F.interpolate(x3, size = (self.height // 2, self.width // 2), mode = 'bilinear', align_corners = False)
        x5 = self.layer4a(torch.cat((x2, x4), dim = 1), embeddings4)
        x5 = self.selfAttention4(x5)
        x5 = self.layer4b(x5, embeddings4)
        #
        x6 = F.interpolate(x5, size = (self.height, self.width), mode = 'bilinear', align_corners = False)        
        x6 = self.layer5a(torch.cat((x1, x6), dim = 1), embeddings5)
        x6 = self.selfAttention5(x6)
        x6 = self.layer5b(x6, embeddings5)
        
        #final convolutional layer, kernel size (1,1)
        out = self.layer6(x6)

        return out

class ResnetBlock(nn.Module):
    #Employs intra block skip connection, which needs a (1, 1) convolution to scale to out_channels

    def __init__(self, in_channels, out_channels, kernel_size, in_channel_image):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer1 = Block(in_channels, out_channels, kernel_size)
        self.SiLU = nn.SiLU()
        self.layer2 = Block(out_channels, out_channels, kernel_size)
        self.resizeInput = nn.Conv2d(in_channel_image, out_channels, (1, 1))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, embeddings):
        xCopy = x

        x = torch.cat((x, embeddings), dim = 1)

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

class GumbelSoftmax(pl.LightningModule):
    def __init__(self, annealing_rate = 1e-5, epsilon = 1e-20):
        super().__init__()
        self.annealing_rate = annealing_rate
        self.epsilon = epsilon

    def forward(self, logits):
        uniform_variables = torch.rand(logits.size(), device = self.device)
        gumbel_noise = -torch.log(-torch.log(uniform_variables + self.epsilon) + self.epsilon)
        gumbel_logits = (logits + gumbel_noise) / self.temperature()
        
        softmax_probabilities = F.softmax(gumbel_logits, dim=-1)

        y = softmax_probabilities
        y_hard = torch.zeros_like(y, device = self.device)
        indices = torch.argmax(y, dim = 1)
        y_hard.scatter_(1, indices.unsqueeze(1), 1)

        return (y_hard - y).detach() + y

    def temperature(self):
        temperature = max(0.5, torch.exp(torch.tensor([-self.annealing_rate * self.global_step], device=self.device)))
        return temperature

