from typing import List
from threading import Thread
import queue

import numpy as np

from bandits import CategoricalBandit


class CategoricalBanditEnv(object):
    def __init__(self, b, n, k, c, probas=None, seed=2139):
        self.b = b  # number of bandits
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
        self.bandits = (
            None  # All bandits participating in this casino is stored in this variable.
        )
        self.actions = np.empty([b, n])
        self.rewards = np.empty([b, n])

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
        que = queue.Queue()
        for i in range(len(self.bandits)):
            thr = Thread(
                target=lambda q, arg: q.put(self.get_action(arg)),
                args=(que, self.bandits[i]),
            )
            thr.start()
        self.actions[:, t] = np.array([que.get() for i in range(len(self.bandits))])  # type: ignore

    def generate_reward(self, bandit: CategoricalBandit, i: int):
        """
        Generate sampling output for a bandit for his/her choice.
        bandit: bandit instance
        i: arm index that the bandit chose
        """
        sampled = np.random.choice(self.c, size=1, p=self.probas[i])[
            0
        ]  # 0 is to read the value out of np.array
        return bandit.generate_reward(
            i, sampled
        )  # This process includes the update of internal belief at the bandit's side.

    def generate_rewards(self, t: int):
        """
        Generate rewards for all bandits based on their action choices at time t.
        """

        que = queue.Queue()
        for i in range(len(self.bandits)):  # type: ignore
            thr = Thread(
                target=lambda q, arg1, arg2: q.put(self.generate_reward(arg1, arg2)),
                args=(que, self.bandits[i], int(self.actions[i, t])),
            )
            thr.start()

        self.rewards[:, t] = np.array([que.get() for i in range(len(self.bandits))])

    def run(self):
        for t in range(self.n):
            self.get_actions(t)
            self.generate_rewards(t)
