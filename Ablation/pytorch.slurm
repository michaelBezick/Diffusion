#!/bin/bash
#SBATCH --job-name=ldm_ablation
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --time=04:00:00
cd $SLURM_SUBMIT_DIR
module load anaconda/2020.11-py38
conda activate myenv
conda list
export WORLD_SIZE=NUM_GPUS
export NODE_RANK=RANK
export MASTER_ADDR=MASTER_NODE_IP
export MASTER_PORT=PORT

srun python3 LDM_Training_Pytorch.py 10000 50 0
