# pylint: disable=no-member
import numpy as np
from scipy.special import rel_entr


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

    def __init__(self, env, probas=None, coi=None, seed=2139):  # k is number of arms
        assert probas is None or probas.shape == (k, c)

        self.env = env
        self.k = self.env.k
        self.c = self.env.c
        self.seed = seed

        if coi is None:
            np.random.seed(self.seed)
            coi = np.random.dirichlet(np.ones(self.c), size=self.k)

        self.coi = coi
        # self.ncoi = [i for i in range(c)].remove(self.coi)

        # if probas is None:
        #     self.probas = np.random.dirichlet(np.ones(self.c), size=self.k)
        # else:
        #     self.probas = probas  # give initial probas maunally

        self.best_arm = np.argmax(
            [rel_entr(self.env.probas[i], self.coi) for i in range(self.k)]
        )
        # self.best_proba = max(
        #     self.probas[:, self.coi]
        # )  # unlike Bern, we need reward function of bandit to compute best

    def generate_reward(self, i):
        # The player selected the i-th machine.
        # res = np.random.choice(self.c, size=1, p=self.probas[i])[0]
        # if res == self.coi:
        #     return 1
        # else:
        #     return 0
        return rel_entr(
            self.env.probas[i], self.coi
        )  # rel_entr returns KL divgergence. It measure how the arm
