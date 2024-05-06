import torch
from tqdm import tqdm
import tensorflow as tf
from torch.utils.data import DataLoader

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

dataset = torch.load("./Experiment_8/generated_dataset.pt")

num_samples = 20_000

dataset = expand_output(dataset, num_samples)
batch_size = 100

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
)

FOM_calculator = load_FOM_model("../Files/VGGnet.json", "../Files/VGGnet_weights.h5")
FOM_measurements = []
best_images_tuple_list = []
best_images_tuple_list.append((-100, 1))
best_images_tuple_list.append((-100, 1))
best_images_tuple_list.append((-100, 1))

for batch in tqdm(train_loader):
    FOM = FOM_calculator(torch.permute(batch.repeat(1, 3, 1, 1), (0, 2, 3, 1)).numpy())
    FOM_measurements.extend(FOM.numpy().flatten().tolist())
    for index, FOM_item in enumerate(FOM.numpy().flatten().tolist()):
        for i in range(len(best_images_tuple_list)):
            compare = best_images_tuple_list[i][0]
            if FOM_item > compare:
                best_images_tuple_list[i] = (FOM_item, batch[index, :, :, :])
                break

best_images = torch.zeros(3, 1, 64, 64)
print(best_images_tuple_list)
for i in range(3):
    best_images[i, :, :, :] = best_images_tuple_list.pop()[1]

best_images = clamp_output(best_images, 0.5)
print(best_images)
torch.save(best_images, "best_topologies.pt")
