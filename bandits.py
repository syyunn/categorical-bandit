import numpy as np


class Bandit(object):
    def generate_reward(self, i):
        raise NotImplementedError


class CategoricalBandit(Bandit):
    """
    k: # of arms
    c: # of categories for each arm
    probas: custom prob. size of k*c
    coi: category of interest
    seed: random seed to generate underlying probabilities for all arms
    """

    def __init__(self, k, c, probas=None, coi=0, seed=2139):  # k is number of arms
        assert probas is None or probas.shape == (k, c)

        self.k = k
        self.c = c
        self.seed = seed
        self.coi = coi
        self.ncoi = [i for i in range(c)].remove(self.coi)

        if probas is None:
            np.random.seed(self.seed)
            self.probas = np.random.dirichlet(
                np.ones(self.c), size=self.k
            )  # generate true proba randomly
        else:
            self.probas = probas  # give true probas maunally

        self.best_arm = np.argmax(self.probas[:, self.coi])

        self.best_proba = max(
            self.probas[:, self.coi]
        )  # unlike Bern, we need reward function of bandit to compute best

        self.worst_proba = min(
            self.probas[:, self.coi]
        )  # unlike Bern, we need reward function of bandit to compute best

    def generate_reward(self, i):  # i is the best arm choice at run-timt t
        # The player selected the i-th machine. We use actual probas in this case.
        sampled = np.random.choice(self.c, size=1, p=self.probas[i])[
            0
        ]  # pull the lever of slot-machine i. If the machine gives the same category of interest, it gets reward 1. You can accomodate any types of reward as you want.

        res = {"reward": None, "sampled": sampled}

        if res == self.coi:
            reward = 1
            res["reward"] = reward
        else:
            reward = 0
            res["reward"] = reward

        return res
