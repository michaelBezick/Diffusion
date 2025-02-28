#!/bin/bash
module load anaconda/2020.11-py38
cd $SLURM_SUBMIT_DIR
export WORLD_SIZE=NUM_GPUS
export NODE_RANK=RANK
export MASTER_ADDR=MASTER_NODE_IP
export MASTER_PORT=PORT
python -m torch.distributed.launch --nproc_per_node=2 cVAE_Training.py
