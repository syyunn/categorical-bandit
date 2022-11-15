from __future__ import division

import time

import numpy as np
from scipy.stats import beta


class Solver(object):
    def __init__(self, bandit):
        """
        bandit (Bandit): the target bandit to solve.
        """
        np.random.seed(int(time.time()))

        self.bandit = bandit

        self.counts = [0] * self.bandit.k
        self.actions = []  # A list of machine ids, 0 to bandit.n-1.
        self.regret = 0.0  # Cumulative regret.
        self.regrets = [0.0]  # History of cumulative regret.

    def update_regret(self, i):
        # i (int): index of the selected machine.
        self.regret += self.bandit.best_proba - self.bandit.probas[i][self.bandit.coi]
        self.regrets.append(self.regret)

    @property
    def estimated_probas(self):
        raise NotImplementedError

    def run_one_step(self):
        """Return the machine index to take action on."""
        raise NotImplementedError

    def run(self, num_steps):
        assert self.bandit is not None
        for _ in range(num_steps):
            i = self.run_one_step()

            self.counts[i] += 1
            self.actions.append(i)
            self.update_regret(i)
        return None


class ThompsonSamplingCategorical(Solver):
    def __init__(self, bandit):
        super(ThompsonSamplingCategorical, self).__init__(bandit)

        self._alpha = np.ones(
            [self.bandit.k, self.bandit.c]
        )  # initialize Dirichlet distribution's param alpha to 1s

    @property
    def estimated_probas(self):
        return [
            self._alpha[i][self.bandit.coi] / np.sum(self._alpha[i])
            for i in range(self.bandit.k)
        ]

    def run_one_step(self):
        samples = [
            np.random.dirichlet(self._alpha[i]) for i in range(self.bandit.k)
        ]  # exploit what agent knows
        i = max(
            range(10), key=lambda x: samples[x][self.bandit.coi]
        )  # best arm as far as bandit knows
        res = self.bandit.generate_reward(i)
        self._alpha[i][res["sampled"]] += 1

        # print(self._alpha)
        return i
