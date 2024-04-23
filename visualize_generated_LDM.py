import torch
import matplotlib.pyplot as plt

def expand_output(tensor: torch.Tensor, num_samples):
    x = torch.zeros([num_samples, 1, 64, 64])
    x[:, :, 0:32, 0:32] = tensor
    x[:, :, 32:64, 0:32] = torch.flip(tensor, dims=[2])
    x[:, :, 0:32, 32:64] = torch.flip(tensor, dims=[3])
    x[:, :, 32:64, 32:64] = torch.flip(tensor, dims=[2, 3])

    return x
def save_image_grid(tensor: torch.Tensor, filename):
    grid_image = (
        tensor.permute(1, 2, 0).expand(64, 64, 3).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    )
    plt.imshow(grid_image)
    plt.axis("off")
    plt.savefig(filename, dpi=300, bbox_inches='tight')

dataset = torch.load("./Generated_Datasets/Experiment_8/generated_dataset.pt")
image = dataset[0, :, :, :].unsqueeze(0)

save_image_grid(expand_output(image, 1).squeeze(0), "generated_sample_ldm.png")
