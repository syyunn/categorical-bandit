import numpy as np

seeds = range(0, 10)

for seed in seeds:
    with open(f'./submit_rs_lg_lb_seed{seed}.sh', 'w') as f:
        f.write(
f"""#!/bin/bash
seed=$1

# Slurm sbatch options
#SBATCH -o myScript.sh.log-%j

# # Loading the required module
# source /etc/profile
# module load anaconda/2020a

# Run the script
python main.py --seed {seed}
""")
