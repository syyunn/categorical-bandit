import os
import pickle
from env import CategoricalBanditEnv
from collections import Counter
import numpy as np

dirs = os.listdir("./experiment4/")
dirs.sort()

hire_ratio = []
ptemps = list(np.linspace(0,1,11)) + list(np.linspace(0.21,0.29,9))
ptemps.sort()

expids = range(0, 10)

X = []
Y = []

for ptemp in ptemps:
    mean_hr = []
    for expid in expids:
# for dir in dirs:
#     if "expid" in dir:
#         with open(f"./experiment2/{dir}/env.pickle", "rb") as f:
        with open(f"/Users/suyeol/Dropbox (MIT)/categorical-bandit/experiment2/results_K112_C26_N2000_B_5_L1_priorTrue_priorTemp_{ptemp}_expid{expid}/env.pickle", "rb") as f:
            envp = pickle.load(f)
            for b in envp.bandits:
                counts = Counter(b.hires)
                mean_hr.append(counts[0] / (counts[-1] + counts[0]))
    X.append(ptemp)
    Y.append(np.mean(mean_hr))
print("hi")

import numpy as np
import matplotlib.pyplot as plt

plt.scatter(X, Y, alpha=0.5, s=1.4)
plt.plot(X,Y)
plt.xticks(np.linspace(0,1,21))
plt.show()