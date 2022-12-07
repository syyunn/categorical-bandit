import numpy as np

ptemps = np.linspace(0.21,0.29,9)
expids = range(0, 10)

for ptemp in ptemps:
    for expid in expids:
        with open(f'./bashes/submit_ptemp{ptemp}_expid{expid}.sh', 'w') as f:
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
python main.py --ptemp {ptemp} --expid {expid}
""")
