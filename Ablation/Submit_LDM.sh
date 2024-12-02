#!/bin/bash
sbatch -A kildisha-k --nodes=1 --gpus-per-node=1 --cpus-per-gpu=8 --time=14-00:00:00 ./LDM.sh
