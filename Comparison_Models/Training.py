import torch
import torchvision
from PIL import Image
from torch.autograd import grad as torch_grad
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(
        self,
        generator,
        discriminator,
        gen_optimizer,
        dis_optimizer,
        logger,
        gp_weight=1,
        critic_iterations=5,
        print_every=50,
        use_cuda=True,
    ):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {"G": [], "D": [], "GP": [], "gradient_norm": []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.i = 0
        self.logger = logger

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def _critic_train_iteration(self, data, labels):

        self.D_opt.zero_grad()

        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size, labels).detach()

        # Calculate probabilities on real and generated data
        data = data.requires_grad_()
        if self.use_cuda:
            data = data.cuda()
        d_real = self.D(data, labels)
        d_generated = self.D(generated_data, labels)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(data, generated_data, labels)
        self.losses["GP"].append(gradient_penalty.item())
        self.logger.add_scalar("gradient_penalty", gradient_penalty.item(), self.i)

        # Create total loss and optimize
        d_loss = torch.mean(d_generated) - torch.mean(d_real) + gradient_penalty
        d_loss.backward()
        self.D_opt.step()

        d_loss_without_gradient_penalty = d_generated.mean() - d_real.mean()

        # Record loss
        self.losses["D"].append(d_loss_without_gradient_penalty.item())
        self.logger.add_scalar("d_loss_without_gradient_penalty", d_loss_without_gradient_penalty.item(), self.i)

    def _generator_train_iteration(self, data, labels):
        """ """
        self.G_opt.zero_grad()

        # Get generated data
        batch_size = data.size()[0]
        generated_data = self.sample_generator(batch_size, labels)

        # Calculate loss and optimize
        d_generated = self.D(generated_data, labels)
        g_loss = -torch.mean(d_generated)
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.losses["G"].append(g_loss.item())
        self.logger.add_scalar("generator loss", g_loss.item(), self.i)

    def _gradient_penalty(self, real_data, generated_data, labels):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1, requires_grad=True)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated, labels)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=(
                torch.ones(prob_interpolated.size()).cuda()
                if self.use_cuda
                else torch.ones(prob_interpolated.size())
            ),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        self.losses["gradient_norm"].append(gradients.norm(dim=1).mean().item())
        self.logger.add_scalar("gradient_norm", gradients.norm(dim=1).mean().item(), self.i)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader):
        for i, data in enumerate(data_loader):
            self.i += i
            images, labels = data
            images = images.cuda().float()
            labels = labels.cuda().float()
            labels = labels.unsqueeze(1)
            self.num_steps += 1

            self._critic_train_iteration(images, labels)

            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(images, labels)

            if i % self.print_every == 0:
                print("Iteration {}".format(i + 1))
                print("D: {}".format(self.losses["D"][-1]))
                print("GP: {}".format(self.losses["GP"][-1]))
                print("Gradient norm: {}".format(self.losses["gradient_norm"][-1]))
                if self.num_steps > self.critic_iterations:
                    print("G: {}".format(self.losses["G"][-1]))

    def train(self, data_loader, epochs):
        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch + 1))
            self._train_epoch(data_loader)

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

    def sample_generator(self, num_samples, labels):
        latent_samples = self.G.sample_latent(num_samples)
        if self.use_cuda:
            latent_samples = latent_samples.cuda()
        generated_data = self.G(latent_samples, labels)
        if self.i % 300 == 0:

            grid_image = torchvision.utils.make_grid(
                generated_data, normalize=True
            )
            self.logger.add_image("generated images", grid_image, self.i)

        return generated_data

    def sample(self, num_samples, labels):
        generated_data = self.sample_generator(num_samples, labels)
        # Remove color channel
        return generated_data.data.cpu().numpy()[:, 0, :, :]
