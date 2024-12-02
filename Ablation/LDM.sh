#!/bin/bash
cd $SLURM_SUBMIT_DIR
module load anaconda/2020.11-py38
export WORLD_SIZE=NUM_GPUS
export NODE_RANK=RANK
export MASTER_ADDR=MASTER_NODE_IP
export MASTER_PORT=PORT
python -m torch.distributed.launch --nproc_per_node=1 LDM_Training.py

