#!/bin/bash
module load anaconda
cd $SLURM_SUBMIT_DIR
python3 -m main.py
