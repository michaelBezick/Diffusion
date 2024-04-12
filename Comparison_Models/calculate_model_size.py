import numpy as np
import torch

from LDM_Classes import LDM, VAE, AttentionUNet

experiment_name = "Experiment_5"
experiment_notes = "Having sampling mean start and end at 1.8 with variance 0.1. Will give some variation of conditioning during training"
num_samples = 20_000
batch_size = 1000
mean = 1.7
variance = 0.1
variable_conditioning = True

checkpoint_path_LDM = "../logs/LDM/version_1/checkpoints/epoch=3552-step=106590.ckpt"
############################################################

FOM_values = None
if variable_conditioning == False:
    FOM_values = variance * torch.randn(batch_size) + mean
    FOM_values = FOM_values.float()
    FOM_values = FOM_values.to("cuda")
ddpm = AttentionUNet(UNet_channel=64, batch_size=batch_size).to("cuda")
vae = VAE(h_dim=128, batch_size=batch_size).to("cuda")
ldm = (
    LDM.load_from_checkpoint(
        checkpoint_path_LDM,
        DDPM=ddpm,
        VAE=vae,
        in_channels=1,
        batch_size=batch_size,
        num_steps=1000,
        latent_height=8,
        latent_width=8,
        lr=1e-3,
    )
    .to("cuda")
    .eval()
)
torch.set_float32_matmul_precision("high")

model_parameters = filter(lambda p: p.requires_grad, ldm.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)
