import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

num_images_to_save = 5


def clamp_output(tensor: torch.Tensor, threshold):
    return torch.where(tensor > threshold, torch.tensor(1.0), torch.tensor(0.0))


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


WGAN_dataset = torch.from_numpy(np.load("./WGAN_dataset.npy"))
cVAE_dataset = torch.from_numpy(np.load("./cVAE_dataset.npy"))
AAE_dataset = torch.from_numpy(np.load("./AAE_dataset.npy"))


num_samples = 20_000
WGAN_dataset = expand_output(WGAN_dataset, num_samples)
cVAE_dataset = expand_output(cVAE_dataset, num_samples)
AAE_dataset = expand_output(AAE_dataset, num_samples)

batch_size = 100

WGAN_train_loader = DataLoader(
    WGAN_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
)
cVAE_train_loader = DataLoader(
    cVAE_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
)
AAE_train_loader = DataLoader(
    AAE_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
)

train_loader_list = [
    (WGAN_train_loader, "WGAN"),
    (cVAE_train_loader, "cVAE"),
    (AAE_train_loader, "AAE"),
]

for train_loader, model_name in train_loader_list:

    FOM_calculator = load_FOM_model(
        "../Files/VGGnet.json", "../Files/VGGnet_weights.h5"
    )
    FOM_measurements = []
    best_images_tuple_list = []

    for i in range(num_images_to_save):
        best_images_tuple_list.append((-100, 1))

    for batch in tqdm(train_loader):
        batch = clamp_output(batch, 0.5)
        FOM = FOM_calculator(
            torch.permute(batch.repeat(1, 3, 1, 1), (0, 2, 3, 1)).numpy()
        )
        FOM_measurements.extend(FOM.numpy().flatten().tolist())
        for index, FOM_item in enumerate(FOM.numpy().flatten().tolist()):
            for i in range(len(best_images_tuple_list)):
                compare = best_images_tuple_list[i][0]
                if FOM_item > compare:
                    # this works because, given a sorted list, if x is greater than at least 1 element in list, the least element needs
                    # to be swapped with x
                    best_images_tuple_list[0] = (FOM_item, batch[index, :, :, :])
                    # only need to sort after insertion; first iteration works fine
                    best_images_tuple_list = sorted(
                        best_images_tuple_list, key=lambda x: x[0]
                    )
                    break

    best_images = torch.zeros(num_images_to_save, 1, 64, 64)
    for i in range(num_images_to_save):
        best_images[i, :, :, :] = best_images_tuple_list.pop()[1]

    best_images = clamp_output(best_images, 0.5)
    torch.save(best_images, f"best_topologies_{model_name}_discrete_measurement.pt")
