#!/bin/bash

sbatch --nodes=1 --gpus-per-node=1 --cpus-per-gpu=8 --constraint="B|D|K" --time=4:00:00 ./AAE.sh