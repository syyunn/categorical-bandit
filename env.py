from typing import List, Tuple
from threading import Thread
import queue

import numpy as np

from bandits import CategoricalBandit
from lobbyists import CategoricalLobbyist


class CategoricalBanditEnv(object):
    def __init__(self, b, n, k, c, l, probas=None, seed=2139):
        self.b = b  # number of bandits
        self.n = n  # number of trials
        self.k = k  # number of arms
        self.c = c  # number of categories for each arm
        self.l = l  # number of lobbyists
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

        # Define bandits living in this env.
        self.bandits = None  # All bandits participating in this political casino is stored in this variable.
        self.actions = np.empty([b, 2, n])  # 2 is to store (i, l) tuples
        self.rewards = np.empty([b, n])

        # Define lobbyists living in this env.
        self.lobbyists = None  # All lobbyists participating in this political casino is stored in this variable.
        self.counts_lobbyists = [0] * self.env.l

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
        self.actions[:, :, t] = [que.get() for i in range(len(self.bandits))]  # type: ignore

    def generate_reward(self, bandit: CategoricalBandit, action: Tuple[int, int]):
        """
        Generate sampling output for a bandit for his/her choice.
        bandit: bandit instance
        i: arm index that the bandit chose
        """
        i, l = action
        sampled = np.random.choice(self.c, size=1, p=self.probas[i])[
            0
        ]  # this is actual pulling in the real world # 0 is to read the value out of np.array

        self.update_lobbyist(i, l, sampled)

        return bandit.generate_reward(
            i, l, sampled
        )  # This process includes the update of internal belief at the bandit's side.

    def update_lobbyist(self, i, l, sampled):
        """
        Update lobbyist's belief about the true probability of the arm i.
        i: arm index that the bandit has chosen
        l: lobbyist index that the bandit has chosen
        sampled: index of category that is sampled from the arm i
        """
        lobbyist = self.lobbyists[l]

        return lobbyist.update_belief(
            i, sampled
        )  # This process includes the update of internal belief of the lobbyists.

    def generate_rewards(self, t: int):
        """
        Generate rewards for all bandits based on their action choices at time t.
        """

        que = queue.Queue()
        for i in range(len(self.bandits)):  # type: ignore
            thr = Thread(
                target=lambda q, arg1, arg2: q.put(self.generate_reward(arg1, arg2)),
                args=(que, self.bandits[i], int(self.actions[i, :, t])),
            )
            thr.start()

        self.rewards[:, t] = np.array([que.get() for i in range(len(self.bandits))])

    def update_lobbyists(self, t: int):
        """
        update belief parameters of all lobbyists based on their action choices at time t.
        """

        que = queue.Queue()
        for i in range(len(self.lobbyists)):  # type: ignore
            thr = Thread(
                target=lambda q, arg1, arg2: q.put(self.generate_reward(arg1, arg2)),
                args=(que, self.bandits[i], int(self.actions[i, :, t])),
            )
            thr.start()

        self.rewards[:, t] = np.array([que.get() for i in range(len(self.bandits))])

    def run(self):
        if self.l == 0:
            for t in range(self.n):
                self.get_actions(t)
                self.generate_rewards(t)
        else:
            for t in range(self.n):
                self.get_actions(t)
                self.generate_rewards(t)
