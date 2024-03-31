#!/bin/bash

sbatch --nodes=2 --gpus-per-node=2 --cpus-per-gpu=8 --constraint="B|D" --time=4:00:00 ./VAE.sh
