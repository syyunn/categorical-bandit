from typing import List, Tuple
from threading import Thread
import queue
from tqdm import tqdm

import numpy as np

from bandits import CategoricalBandit
from lobbyists import CategoricalLobbyist


class CategoricalBanditEnv(object):
    def __init__(self, b, n, k, c, l, cois, probas=None, seed=2139):
        self.b = b  # number of bandits
        self.n = n  # number of trials
        self.k = k  # number of arms
        self.c = c  # number of categories for each arm
        self.l = l  # number of lobbyists
        self.cois = cois  # category of interest for each bandit in order
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
        self.counts_lobbyists = [0] * self.l

        # Define environment level reward
        self.sum_rewards_of_bandits = []
        self.mean_rewards_of_bandits = []

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
        que = queue.Queue(maxsize=len(self.bandits))
        for i in range(len(self.bandits)):
            thr = Thread(
                target=lambda q, arg: q.put(
                    self.get_action(arg)
                ),  # arg is bandit instance
                args=(que, self.bandits[i]),
            )
            thr.start()  # parallelize the process of getting actions from multiple bandits using multi-threading
        self.actions[:, :, t] = [que.get(block=True) for i in range(len(self.bandits))]  # type: ignore

    def generate_reward(self, bandit: CategoricalBandit, action: Tuple[int, int]):
        """
        Generate sampling output for a bandit for his/her choice.
        bandit: bandit instance
        i: arm index that the bandit chose
        """
        i, l = action
        i, l = int(i), int(l)
        sampled = np.random.choice(self.c, size=1, p=self.probas[i])[
            0
        ]  # this is actual pulling in the real world # [0] is just to read out the value from np.array

        if l != -1:
            self.update_lobbyist(i, l, sampled)

        reward = bandit.generate_reward(
            i, l, sampled
        )  # This process includes the update of internal belief at the bandit's side.

        return reward

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

    def compute_environment_level_reward(self, t):
        """
        Compute the environment level reward at time t.
        """
        self.mean_rewards_of_bandits.append(np.sum(self.rewards[:, t]) / self.b)

    def generate_rewards(self, t: int):
        """
        1. Generate rewards for all bandits based on their action choices at time t.
        2. Update bandits' internal belief based on the rewards.
        2-1. We don't update bandits' internal belief in case they used the lobbyist's belief.
        3. Update lobbyists' internal belief based on the rewards.
        """

        threads = []
        que = queue.Queue(
            maxsize=len(self.bandits)
        )  # limit concurrent threads to number of bandits

        def _worker(bandit, action, queue):
            result = self.generate_reward(bandit, action)
            queue.put(result)

        for b in range(len(self.bandits)):  # type: ignore
            bandit = self.bandits[b]
            thread = Thread(
                target=_worker,
                args=(
                    bandit,
                    self.actions[b, :, t],
                    que,
                ),  # arg2 is (i, l) tuple
            )
            threads.append(thread)
            thread.start()

        [thread.join() for thread in threads]  # wait for all threads to finish

        self.rewards[:, t] = np.array(
            [que.get(block=True) for i in range(len(self.bandits))]
        )

    def run(self):
        for t in tqdm(range(self.n)):
            self.get_actions(t)
            self.generate_rewards(t)
            self.compute_environment_level_reward(t)
