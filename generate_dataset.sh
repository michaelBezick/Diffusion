#!/bin/bash
module load anaconda
cd $SLURM_SUBMIT_DIR
python -m generate_dataset_Gilbreth.py
