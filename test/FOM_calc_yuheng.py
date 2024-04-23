import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import tensorflow as tf

def load_FOM_model(model_path, weights_path):
    with open(model_path, "r") as file:
        data = file.read()

    FOM_calculator = tf.keras.models.model_from_json(data)
    FOM_calculator.load_weights(weights_path)

    return FOM_calculator

dataset_path = ""
dataset = torch.load(dataset_path)
FOM_calculator = load_FOM_model("Files/VGGnet.json", "Files/VGGnet_weights.h5")
batch_size = 100

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
)

FOM_measurements = []
i = 0
for batch in tqdm(train_loader):
    FOM = FOM_calculator(torch.permute(batch.repeat(1, 3, 1, 1), (0, 2, 3, 1)).numpy())
    FOM_measurements.extend(FOM.numpy().flatten().tolist())
