#!/bin/bash
module load anaconda/2020.11-py38
cd $SLURM_SUBMIT_DIR
python -m generate_dataset_Gilbreth.py
