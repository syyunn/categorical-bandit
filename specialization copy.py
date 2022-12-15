import os
import pickle
from env import CategoricalBanditEnv
from collections import Counter
import numpy as np

dirs = os.listdir("./experiment7/")
dirs.sort()

Xs = list(set([float(dirs[i].split('_')[9]) for i in range(len(dirs))]))
Xs.sort()

print(Xs)

dir_pair_true = []
dir_pair_false = []

X = []
Y = []
Z = []
for x in Xs:
    if (f'results_K112_C26_N2000_B_10_L2_priorTrue_priorTemp_{x}_expid0_seed0_legiswiseFalse_seed_lobbyist100' in dirs) and (f'results_K112_C26_N2000_B_10_L2_priorTrue_priorTemp_{x}_expid0_seed0_legiswiseTrue_seed_lobbyist100' in dirs):
        X.append(x)
        dir_pair_true.append(f'results_K112_C26_N2000_B_10_L2_priorTrue_priorTemp_{x}_expid0_seed0_legiswiseTrue_seed_lobbyist100')
        dir_pair_false.append(f'results_K112_C26_N2000_B_10_L2_priorTrue_priorTemp_{x}_expid0_seed0_legiswiseFalse_seed_lobbyist100')

print(len(dir_pair_true))
print(len(dir_pair_false))

for dir in dir_pair_false:
    match_total = 0
    with open(f"/Users/suyeol/Dropbox (MIT)/categorical-bandit/experiment6/{dir}/env.pickle", "rb") as f:
        env = pickle.load(f)
    bandits = env.bandits
    for b in bandits:
        counts = Counter(b.hires)
        # print(counts)
        match = counts[b.coi]
        unmatch = counts[abs(1-b.coi)]
        # print(match, unmatch)
        match_total += match
    Y.append(match_total/(len(bandits)*env.n))

for dir in dir_pair_true:
    match_total = 0
    with open(f"/Users/suyeol/Dropbox (MIT)/categorical-bandit/experiment6/{dir}/env.pickle", "rb") as f:
        env = pickle.load(f)
    bandits = env.bandits
    for b in bandits:
        counts = Counter(b.hires)
        # print(counts)
        match = counts[b.coi]
        unmatch = counts[abs(1-b.coi)]
        # print(match, unmatch)
        match_total += match
    Z.append(match_total/(len(bandits)*env.n))


import numpy as np
import matplotlib.pyplot as plt

plt.plot(X,Y, label="Expertise in Issue Area")
plt.plot(X,Z, label="Expertise in Legislator")
plt.xticks(np.linspace(0,1,101))
plt.legend()
plt.xlabel("Prior Temperature")
plt.ylabel("Ratio of Hiring Matching Expertise Lobbyist")

plt.show()