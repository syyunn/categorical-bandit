#!/bin/bash
ptemp=$1
expid=$2
 
# Slurm sbatch options
#SBATCH -o myScript.sh.log-%j

# # Loading the required module
# source /etc/profile
# module load anaconda/2020a

# Run the script
python main.py --ptemp 0.16 --expid 47
