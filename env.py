from typing import List
from multiprocessing import Pool

import numpy as np

from bandits import CategoricalBandit


class CategoricalBanditEnv(object):
    def __init__(
        self, n, k, c, bandits: List[CategoricalBandit], probas=None, seed=2139
    ):
        self.n = n  # Number of trials
        self.k = k  # Number of arms
        self.c = c  # Number of categories for each arm
        self.seed = seed
        if probas is None:
            np.random.seed(
                self.seed
            )  # To fix the true probabilities of slot machines for all different settings of experiments.
            self.probas = np.random.dirichlet(
                np.ones(self.c), size=self.k
            )  # Generate true probababilities of slot machines randomly.
        else:  # We assume the case that we use the actual probabilities from LDA data
            self.probas = probas  # Assign true proba maunally
        self.bandits = bandits  # All bandits participating in this casino is stored in this variable.
        self.actions = np.empty([len(bandits), n])
        self.rewards = np.empty([len(bandits), n])

    def get_action(self, bandit: CategoricalBandit):
        """
        Run the `make_choice` method in given bandit instance from Categorical Bandit
        """
        return (
            bandit.get_action()
        )  # It returns a integer that refers to a arm among K number of possible choices of the given bandit.

    def get_actions(self, t):
        """
        Get all bandits' decision in parallel at time t
        """

        pool = Pool(processes=len(self.bandits))
        self.actions[:, t] = pool.map(self.get_action, self.bandits)

    def generate_reward(self, bandit: CategoricalBandit, i: int):
        "Generate output for a bandit for his/her choice"
        sampled = np.random.choice(self.c, size=1, p=self.probas[i])[
            0
        ]  # 0 is to read value out of np.array
        return bandit.generate_reward(sampled)

        # if sampled == bandit.coi:
        #     reward = 1  # We use vanilla reward yet
        #     # but you can design your own reward like:
        #     # reward = 1 * CUSTOM_REWARD(ARGS)
        # else:
        #     reward = 0
        # return reward

    def generate_rewards(self, t: int):
        """
        Generate rewards for all bandits based on their action choices at time t.
        """
        pool = Pool(processes=len(self.bandits))
        self.rewards[:, t] = pool.map(
            self.generate_reward, zip(self.bandits, self.actions[:, t])
        )

    def update_bandit(self, bandit: CategoricalBandit, action: int, reward: int):
        """
        Update bandit's parameters.
        """
        bandit.update(action, reward)

    def update_bandits(self, t):
        """
        Update all bandits' parameters.
        """
        pool = Pool(processes=len(self.bandits))
        self.rewards[:, t] = pool.map(
            self.update_bandit,
            zip(self.bandits, self.actions[:, t], self.rewards[:, t]),
        )  # Don't worry! Pool gives results in order.

