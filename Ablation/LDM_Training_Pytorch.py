import os

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from LDM_Ablation_Classes import (
    VAE,
    Ablation_LDM_Pytorch,
    AblationAttentionUNet,
    LabeledDataset,
)

num_devices = 1
num_nodes = 1
num_workers = 1
accelerator = "gpu"
batch_size = 100
epochs = 10_000
in_channels = 1
out_channels = 1

# terms VAE
lr_VAE = 1e-3
lr_DDPM = 1e-3
perceptual_loss_scale = 1
kl_divergence_scale = 0.3

resume_from_checkpoint = False

checkpoint_path_VAE = "../logs/VAE/version_0/checkpoints/epoch=1115-step=33480.ckpt"

checkpoint_path_LDM = ""
device = "cuda"

###########################################################################################


def ddp_setup(rank, world_size):

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


# checkpoint_callback = ModelCheckpoint(filename="good", every_n_train_steps=300)

torch.set_float32_matmul_precision("high")

vae = VAE.load_from_checkpoint(
    checkpoint_path=checkpoint_path_VAE,
    in_channels=in_channels,
    h_dim=128,
    lr=lr_VAE,
    batch_size=batch_size,
    perceptual_loss_scale=perceptual_loss_scale,
    kl_divergence_scale=kl_divergence_scale,
)
vae = vae.to(device)
vae.eval()

DDPM = AblationAttentionUNet(
    in_channels=in_channels,
    out_channels=out_channels,
    UNet_channel=64,
    timeEmbeddingLength=100,
    batch_size=batch_size,
    latent_height=8,
    latent_width=8,
    num_steps=1000,
    FOM_condition_vector_size=100,
)

# logger = TensorBoardLogger(save_dir="logs/", name="LDM")
logger = SummaryWriter(log_dir="runs/")

ldm = Ablation_LDM_Pytorch(
    DDPM,
    vae,
    in_channels=in_channels,
    batch_size=batch_size,
    num_steps=1000,
    latent_height=8,
    latent_width=8,
    lr=lr_DDPM,
    logger=logger,
    device=device,
)

gpu_ids = ["gpu:0"]


ldm = ldm.to(device)


def clamp_output(tensor: torch.Tensor, threshold):
    return torch.where(tensor > threshold, torch.tensor(1.0), torch.tensor(0.0))


dataset = np.expand_dims(np.load("../Files/TPV_dataset.npy"), 1)
normalizedDataset = (dataset - np.min(dataset)) / (np.max(dataset) - np.min(dataset))

normalizedDataset = normalizedDataset.astype(np.float32)

dataset = torch.from_numpy(normalizedDataset)

dataset = clamp_output(dataset, 0.5)

labels = torch.load("../Files/FOM_labels_new.pt")

labeled_dataset = LabeledDataset(dataset, labels)


# defining training classes


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        max_epochs: int,
    ):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.save_every = save_every
        self.ldm = DDP(model, device_ids=[self.local_rank])
        self.max_epochs = max_epochs

    def train(self):

        for epoch in range(self.max_epochs):
            if (epoch + 1) % self.save_every == 0 and self.local_rank == 0:
                checkpoint_path = (
                    f"epoch={epoch+1}-step={epoch*len(self.train_loader)}.ckpt"
                )
                checkpoint = {
                    "state_dict": self.ldm.module.state_dict(),
                    "optimizer_states": self.optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "global_step": epoch * len(self.train_loader),
                }
                torch.save(checkpoint, checkpoint_path)

            for step, batch in enumerate(self.train_loader):

                loss = self.ldm.module.training_step(batch)

                if step % 100 == 0:
                    print(f"GPU: {self.global_rank}\tEpoch: {epoch}\tStep: {step}\tLoss: {loss}\t")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


def main(rank: int, world_size: int, total_epochs: int, save_every: int, resume: bool):
    ddp_setup(rank, world_size)

    if resume and os.path.isfile(checkpoint_path_LDM):
        print(f"Resuming from checkpoint: {checkpoint_path_LDM}")
        checkpoint = torch.load(checkpoint_path_LDM, map_location=f"cuda:{rank}")
        ldm.load_state_dict(checkpoint["state_dict"])
        optimizer = torch.optim.Adam(ldm.parameters(), lr=lr_DDPM)
        optimizer.load_state_dict(checkpoint["optimizer_states"])
        start_epoch = checkpoint["epoch"]
        print(f"Resumed from epoch {start_epoch}")
    else:
        print("Starting training from scratch")
        optimizer = torch.optim.Adam(ldm.parameters(), lr=lr_DDPM)
        start_epoch = 0


    train_loader = DataLoader(
        labeled_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        sampler=DistributedSampler(labeled_dataset),
    )

    trainer = Trainer(
        model=ldm,
        train_loader=train_loader,
        optimizer=optimizer,
        gpu_id=rank,
        save_every=save_every,
        max_epochs=total_epochs,
    )
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    import sys

    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])
    resume_training = bool(int(sys.argv[3]))

    device = 0
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, total_epochs, save_every, resume_training), nprocs=world_size)
    logger.close()
