import numpy as np
import torch

discrete = torch.load("./best_topologies_discrete.pt").to(torch.uint8)
discrete = discrete.flatten(start_dim=2, end_dim=3).squeeze()


continuous = torch.load("./best_topologies_continuous_measurement.pt").to(torch.uint8)
continuous = continuous.flatten(start_dim=2, end_dim=3).squeeze()


for i in range(100):
    image = discrete[i, :]
    with open(f"./Discrete/{i}.txt", "w") as file:
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


for i in range(100):
    image = continuous[i, :]
    with open(f"./Continuous/{i}.txt", "w") as file:
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
