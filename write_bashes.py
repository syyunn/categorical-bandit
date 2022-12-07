import numpy as np

ptemps = np.linspace(0,1,11)
expids = range(0, 100)

for ptemp in ptemps:
    for expid in expids:
        with open(f'./submit_ptemp{ptemp}.sh', 'w') as f:
            f.write(
f"""#!/bin/bash
ptemp=$1
expid=$2
 
# Slurm sbatch options
#SBATCH -o myScript.sh.log-%j

# # Loading the required module
# source /etc/profile
# module load anaconda/2020a

# Run the script
python main.py --ptemp {ptemp}
""")
