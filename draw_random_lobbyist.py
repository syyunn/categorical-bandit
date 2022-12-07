import os
import pickle
from env import CategoricalBanditEnv
from collections import Counter
import numpy as np



import numpy as np
import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(14, 10))

    # Sub.fig. 4: Regrets in time in ratio.
plt.title("Normalized regrets in time")


with open(f"/Users/suyeol/Dropbox (MIT)/categorical-bandit/experiment/results_K112_C26_N2000_B_5_L1_priorFalse_priorTemp_0.5_expid0_seed2139_legiswiseFalse_seed_lobbyist{100}/env.pickle", "rb") as f:
    envp = pickle.load(f)

bandit = envp.bandits[0]

plt.plot(
    range(len(envp.bandits[0].regrets)),
    [
        (bandit.regrets[i] / ((bandit.best_proba - bandit.worst_proba) * i))
        for i in range(len(bandit.regrets))
    ],
    label=f"flat prior",
)

for i in range(1,7):
    with open(f"/Users/suyeol/Dropbox (MIT)/categorical-bandit/experiment/results_K112_C26_N2000_B_5_L1_priorFalse_priorTemp_0.5_expid0_seed2139_legiswiseFalse_seed_lobbyist{i}/env.pickle", "rb") as f:
        envp = pickle.load(f)

    bandit = envp.bandits[0]


    plt.plot(
        range(len(envp.bandits[0].regrets)),
        [
            (bandit.regrets[i] / ((bandit.best_proba - bandit.worst_proba) * i))
            for i in range(len(bandit.regrets))
        ],
        label=f"random seed {i}",
    )
    pass


plt.xlabel("Time step")
plt.ylabel("cum_regret / ub of regret")
# ax1.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
plt.grid(ls="--", alpha=0.3)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.legend()
plt.savefig("random_lobbyist.png")
plt.show()
