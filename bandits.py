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

    def __init__(self, env, probas=None, coi=0):
        assert probas is None or probas.shape == (k, c)

        self.k = env.k
        self.c = env.c
        self.seed = env.seed
        self.coi = coi
        self.ncoi = [i for i in range(self.c)].remove(
            self.coi
        )  # bandit doesn't interest in any categories in self.ncoi

        self.best_arm = np.argmax(self.env.probas[:, self.coi])
        self.best_proba = max(self.env.probas[:, self.coi])
        self.worst_proba = min(self.env.probas[:, self.coi])

        self.belief = np.ones(
            [self.bandit.k, self.bandit.c]
        )  # Initialize Dirichlet distribution's param \alpha to 1s. This is internal belief of the agent over slot-machines.

    def get_action(self):
        samples = [
            np.random.dirichlet(self.belief[i]) for i in range(self.k)
        ]  # exploit what agent knows
        i = max(
            range(self.k), key=lambda x: samples[x][self.bandit.coi]
        )  # best rewarding arm for category of interest as far as bandit knows
        return i

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
