import torch
import LDM_Classes_FID

checkpoint_path_LDM = "/home/mbezick/Desktop/logs/LDM_128_PatchGAN/version_15/checkpoints/good-v1.ckpt"

batch_size = 100
reconstruction_term = 0
perceptual_term = 0
kl_term = 0
adv_term = 0
lr_VAE = 0
lr_DDPM = 0

torch.set_float32_matmul_precision('high')

DDPM = LDM_Classes_FID.AttentionUNet(in_channels=1, out_channels=1, UNet_channel=128, timeEmbeddingLength=16,
                     batch_size=batch_size, height=8, width=8, num_steps=1000, condition_vector_size=16)

vae = LDM_Classes_FID.VAE_GAN(reconstruction_term, perceptual_term, kl_term, adv_term, 1, 64, lr = lr_VAE, batch_size = batch_size)

checkpoint_path_VAE = "/home/mbezick/Desktop/logs/VAE_PatchGAN/version_33/checkpoints/good-v1.ckpt"

vae = vae.load_from_checkpoint(checkpoint_path_VAE, a=reconstruction_term, b=perceptual_term, c=kl_term, d=adv_term, in_channels=1, h_dim=64, lr=lr_VAE, batch_size=batch_size)
vae.eval()

ldm = LDM_Classes_FID.LDM(DDPM, vae, in_channels=1, batch_size=batch_size, num_steps=1000, height=8, width=8, lr=lr_DDPM)

checkpointLDM = torch.load(checkpoint_path_LDM)

ldm.load_state_dict(checkpointLDM['state_dict'])

ldm = ldm.to('cuda')

ldm.eval()


ldm.create_dataset()
