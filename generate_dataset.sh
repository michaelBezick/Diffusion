#!bin/bash
cd $SLURM_SUBMIT_DIR
module load anaconda
python3 generate_dataset_Gilbreth.py
