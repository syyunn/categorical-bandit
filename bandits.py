import time
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

    def __init__(
        self, k, c, probas=None, coi=0, seed=int(time.time())
    ):  # k is number of arms
        assert probas is None or probas.shape == (k, c)
        self.k = k
        self.c = c
        self.seed = seed
        self.coi = 0
        if probas is None:
            np.random.seed(self.seed)
            self.probas = np.random.dirichlet(np.ones(self.c), size=self.k)
        else:
            self.probas = probas  # give initial probas maunally

        self.best_proba = max(
            self.probas[:, self.coi]
        )  # unlike Bern, we need reward function of bandit to compute best

    def generate_reward(self, i):
        # The player selected the i-th machine.
        res = np.random.choice(self.c, size=1, p=self.probas[i])[0]
        if res == self.coi:
            return 1
        else:
            return 0
