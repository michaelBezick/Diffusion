#!/bin/bash
#SBATCH --job-name=ldm_ablation
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
cd $SLURM_SUBMIT_DIR
module load anaconda/2020.11-py38

nodes= ( $( scontrol show hostnames $SLURM_JOB_NODELIST ))
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_pi=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

export WORLD_SIZE=NUM_GPUS
export NODE_RANK=RANK
export MASTER_ADDR=MASTER_NODE_IP
export MASTER_PORT=PORT

srun torchrun \
--nodes 4 \
--nproc_per_node 1 \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:29500 \

LDM_Training_Pytorch.py 10000 50 0
