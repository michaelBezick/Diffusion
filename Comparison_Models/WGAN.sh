#!/bin/bash
module load anaconda/2020.11-py38
cd $SLURM_SUBMIT_DIR
python3 -m main.py
