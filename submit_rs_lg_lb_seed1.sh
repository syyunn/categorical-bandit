#!/bin/bash
seed=$1

# Slurm sbatch options
#SBATCH -o myScript.sh.log-%j

# # Loading the required module
# source /etc/profile
# module load anaconda/2020a

# Run the script
python main.py --seed 1
