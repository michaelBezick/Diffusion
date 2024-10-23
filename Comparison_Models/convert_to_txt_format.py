import torch

num_saved_images = 100


WGAN = torch.load("./best_topologies_WGAN_discrete_measurement.pt").flatten(start_dim=2, end_dim=3).squeeze()
cVAE = torch.load("./best_topologies_cVAE_discrete_measurement.pt").flatten(start_dim=2, end_dim=3).squeeze()
AAE = torch.load("./best_topologies_AAE_discrete_measurement.pt").flatten(start_dim=2, end_dim=3).squeeze()

model_list = [(WGAN, "WGAN"), (cVAE, "cVAE"), (AAE, "AAE")]

for dataset, model_name in model_list:
    for i in range(num_saved_images):
        image = dataset[i, :]
        with open(f"./{model_name}/{i}.txt", "w") as file:
            file.write("64 -0.14 0.14\n")
            file.write("64 -0.14 0.14\n")
            file.write("2 0 0.12\n")
            for pixel in image:
                if pixel.item() == 0:
                    file.write("0\n")
                elif pixel.item() == 1:
                    file.write("1\n")
                else:
                    print("SLKDFJA;DLKAJSD")
                    exit()
            for pixel in image:
                if pixel.item() == 0:
                    file.write("0\n")
                elif pixel.item() == 1:
                    file.write("1\n")
