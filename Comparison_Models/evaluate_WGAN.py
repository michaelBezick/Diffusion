import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import torchvision
from Models import Generator
from PIL import Image
from tqdm import tqdm

img_size = (32, 32, 1)
batch_size = 100
generator = Generator(img_size=img_size, latent_dim=64, dim=32, batch_size=batch_size)
generator.load_state_dict(torch.load("./gen_mnist_model.pt"))
generator.eval()


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
variance = 0.2

FOM_calculator = load_FOM_model("../Files/VGGnet.json", "../Files/VGGnet_weights.h5")

labels_list = []
FOMs_list = []

generator = generator.cuda()

with torch.no_grad():
    for i in tqdm(range(num_samples // batch_size)):
        noise = generator.sample_latent(batch_size).cuda()
        labels = variance * torch.randn((batch_size, 1), device="cuda") + mean
        labels_list.extend(labels.cpu().detach().numpy())
        images = generator(noise, labels)
        images = expand_output(images, batch_size)
        if i == 0:
            save_image_grid(images, "WGAN_Sample.png")
        FOMs = FOM_calculator(
            torch.permute(images.repeat(1, 3, 1, 1), (0, 2, 3, 1)).cpu().numpy()
        )

        FOMs_list.extend(FOMs.numpy().flatten().tolist())

plt.figure()
plt.scatter(labels_list, FOMs_list)
plt.title("WGAN conditioning FOM values versus generated FOM values")
plt.xlabel("Conditioning FOM values")
plt.ylabel("Generated FOM values")
plt.plot()
plt.savefig("WGAN_Scatter.png", dpi=300)

with open("WGAN_Evaluation.txt", "w") as file:
    file.write(f"FOM max: {max(FOMs_list)}\n")
    file.write(f"FOM mean: {sum(FOMs_list) / len(FOMs_list)}")
