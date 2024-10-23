import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import polytensor
from torchvision.utils import save_image
from tqdm import tqdm

from Energy_Encoder_Classes import BVAE, CorrelationalLoss
from Functions import (
    clamp_output,
    compute_pearson_correlation,
    expand_output,
    get_annealing_vectors,
    get_sampling_vars,
    load_FOM_model,
    threshold,
)

device = "cuda"
save_images = True

lowest_num = 5000

clamp = True
threshold = 0.5

num_vars = 64

num_per_degree = [num_vars]
sample_fn = lambda: torch.randn(1, device="cuda")
terms = polytensor.generators.coeffPUBORandomSampler(
    n=num_vars, num_terms=num_per_degree, sample_fn=sample_fn
)

energy_loss_fn = CorrelationalLoss(1, 1, 1)

terms = polytensor.generators.denseFromSparse(terms, num_vars)
terms.append(torch.randn(num_vars, num_vars))

energy_fn = polytensor.DensePolynomial(terms)

second_degree_model = BVAE.load_from_checkpoint("../Annealing_Learnable/Models/QUBO_order_2/epoch=9999-step=200000.ckpt", energy_fn=energy_fn, energy_loss_fn=energy_loss_fn, h_dim=128)

terms.append(torch.randn(num_vars, num_vars, num_vars))

energy_fn = polytensor.DensePolynomial(terms)

third_degree_model = BVAE.load_from_checkpoint("../Annealing_Learnable/Models/QUBO_order_3/epoch=9999-step=200000.ckpt", energy_fn=energy_fn, energy_loss_fn=energy_loss_fn, h_dim=128)

terms.append(torch.randn(num_vars, num_vars, num_vars, num_vars))
energy_fn = polytensor.DensePolynomial(terms)

fourth_degree_model = BVAE.load_from_checkpoint("../Annealing_Learnable/Models/QUBO_order_4/epoch=9999-step=200000.ckpt", energy_fn=energy_fn, energy_loss_fn=energy_loss_fn, h_dim=128)

composite_model = BVAE.load_from_checkpoint("../Annealing_Learnable/Models/Composite/epoch=9999-step=200000.ckpt", energy_fn=energy_fn, energy_loss_fn=energy_loss_fn, h_dim=128)

model_list = [second_degree_model, third_degree_model, fourth_degree_model, composite_model]

energy_loss_fn = CorrelationalLoss()

FOM_calculator = load_FOM_model("../Files/VGGnet.json", "../Files/VGGnet_weights.h5")

model_dir_list = ["./Models/2nd_degree/", "./Models/3rd_degree/", "./Models/4th_degree/", "./Models/Composite/"]

log_dir = ""

second, third, fourth, composite_vectors = get_annealing_vectors()

optimal_vectors = [second, third, fourth, composite_vectors]


def mean_normalize(images: torch.Tensor):
    return (images - torch.min(images)) / (torch.max(images) - torch.min(images))

new_optimal_vectors = []
degree = 2

for model_dir in model_dir_list:
    if degree == 2:
        optimal_vector_list = optimal_vectors[0].cuda()
        model = model_list[0]
        degree = 3
    elif degree == 3:
        optimal_vector_list = optimal_vectors[1].cuda()
        model = model_list[1]
        degree = 4
    elif degree == 4:
        optimal_vector_list = optimal_vectors[2].cuda()
        model = model_list[2]
        degree = 5
    else:
        optimal_vector_list = optimal_vectors[3].cuda()
        model = model_list[3]
        degree = 6

    model = model.cuda()
    new_list = []
    for vector in tqdm(optimal_vector_list):
        energy = model.energy_fn(vector)
        if degree == 4:
            energy = abs(energy - (-260))
        new_list.append((energy, vector))

    sorted_new_list = sorted(new_list, key=lambda x: x[0])
    # sorted in increasing order, want first vectors
    cut_off_new_tuple_list = sorted_new_list[0:lowest_num]

    cut_off_new_list = [x[1] for x in cut_off_new_tuple_list]
    new_optimal_vectors.append(torch.stack(cut_off_new_list))
    

optimal_vectors = new_optimal_vectors


largest_FOM_global = 0

degree = 2

for model_dir in tqdm(model_dir_list):
    energies = []
    FOM_global = []

    energy_fn = None
    folder_path = None
    model = None

    if degree == 2:
        optimal_vector_list = optimal_vectors[0]
        folder_path = "./Models/2nd_degree/"
        degree = 3
        model = model_list[0]
        """ADDED"""
        continue
    elif degree == 3:
        optimal_vector_list = optimal_vectors[1]
        folder_path = "./Models/3rd_degree/"
        degree = 4
        model = model_list[1]
        """ADDED"""
        continue
    elif degree == 4:
        optimal_vector_list = optimal_vectors[2]
        folder_path = "./Models/4th_degree/"
        model = model_list[2]
        degree = 5
        continue
    else:
        optimal_vector_list = optimal_vectors[3]
        folder_path = "./Models/Composite/"
        model = model_list[3]
        degree = 6

    print(optimal_vector_list.size())
    num_iters = optimal_vector_list.size()[0] // 100
    print(num_iters)

    model = model.to(device)
    model = model.eval()

    num_logits, scale = get_sampling_vars(model)

    zero_tensor = torch.zeros([100, 64])
    FOM_measurements = []
    largest_FOM = 0
    been_saved = False

    with torch.no_grad():
        for iters in range(num_iters):
            zero_tensor = optimal_vector_list[iters * 100 : (iters + 1) * 100]

            vectors = zero_tensor.cuda()

            vectors_energies = model.energy_fn(vectors)
            numpy_energies = vectors_energies.detach().cpu().numpy()
            energies.extend(numpy_energies)

            output = model.vae.decode(vectors)
            output_expanded = expand_output(output)

            if clamp:
                output_expanded = clamp_output(output_expanded, threshold)

            FOM = FOM_calculator(
                torch.permute(output_expanded.repeat(1, 3, 1, 1), (0, 2, 3, 1)).numpy()
            )

            FOM_measurements.extend(FOM.numpy().flatten().tolist())
            FOM_global.extend(FOM.numpy().flatten().tolist())

            if np.max(np.array(FOM_measurements)) > largest_FOM and np.max(np.array(FOM_measurements)) < 3.0:
                largest_FOM = np.max(np.array(FOM_measurements))

            grid = torchvision.utils.make_grid(output_expanded.cpu())

            log_dir = folder_path

            if save_images and been_saved == False:
                print(log_dir)
                save_image(grid, log_dir + "image.jpg")
                been_saved = True

        FOM_measurements = np.array(FOM_measurements)
        average = np.mean(FOM_measurements)

        if largest_FOM > largest_FOM_global:
            largest_FOM_global = largest_FOM

        # calculating pearson correlation
        pearson_correlation = compute_pearson_correlation(FOM_global, energies)

        with open(log_dir + "/FOM_data_lowest.txt", "w") as file:
            file.write(f"Average FOM: {average}\n")
            file.write(f"Max FOM: {largest_FOM}\n")
            file.write(f"pearson_correlation: {pearson_correlation}")

        plt.figure()
        plt.scatter(energies, FOM_global)
        plt.xlabel("Energy of vectors")
        plt.ylabel("FOM of samples")
        plt.title("Sampled correlation")
        plt.savefig(log_dir + "/LowestSampledCorrelation.png")


with open("Experiment_Summary_lowest.txt", "w") as file:
    file.write(f"Experiment max FOM: {largest_FOM_global}")
