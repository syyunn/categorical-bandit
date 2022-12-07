import os
import pickle
from env import CategoricalBanditEnv
from collections import Counter
import numpy as np

dirs = os.listdir("./experiment4/legiswiseFalse")
dirs.sort()

X = np.linspace(0,1,11)
Y = []
Z = []
for dir in dirs:
    match_total = 0
    with open(f"/Users/suyeol/Dropbox (MIT)/categorical-bandit/experiment4/legiswiseFalse/{dir}/env.pickle", "rb") as f:
        env = pickle.load(f)
    bandits = env.bandits
    for b in bandits:
        counts = Counter(b.hires)
        # print(counts)
        match = counts[b.coi]
        unmatch = counts[abs(1-b.coi)]
        print(match, unmatch)
        match_total += match
    Y.append(match_total/(len(bandits)*env.n))

import numpy as np
import matplotlib.pyplot as plt

plt.plot(X,Y, label="Expertise in Issue Area")

dirs = os.listdir("./experiment4/legiswiseTrue")
dirs.sort()

for dir in dirs:
    match_total = 0
    with open(f"/Users/suyeol/Dropbox (MIT)/categorical-bandit/experiment4/legiswiseTrue/{dir}/env.pickle", "rb") as f:
        env = pickle.load(f)
    bandits = env.bandits
    for b in bandits:
        counts = Counter(b.hires)
        match = counts[b.coi]
        match_total += match
    Z.append(match_total/(len(bandits)*env.n))

plt.plot(X,Z, label="Expertise in Legislator")
plt.xticks(np.linspace(0,1,11))
plt.legend()
plt.xlabel("Prior Temperature")
plt.ylabel("Ratio of Hiring Matching Expertise Lobbyist")


plt.show()