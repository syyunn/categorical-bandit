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

        self.env = env
        self.seed = env.seed
        self.coi = coi
        self.ncoi = [i for i in range(self.env.c)].remove(
            self.coi
        )  # bandit doesn't interest in any categories in self.ncoi

        self.best_arm = np.argmax(self.env.probas[:, self.coi])
        self.best_proba = max(self.env.probas[:, self.coi])
        self.worst_proba = min(self.env.probas[:, self.coi])

        self.belief = np.ones(
            [self.env.k, self.env.c]
        )  # Initialize Dirichlet distribution's param \alpha to 1s. This is internal belief of the agent over slot-machines.

        self.counts = [0] * self.env.k
        self.actions = []  # A list of machine ids, 0 to k-1.
        self.regret = 0.0  # Cumulative regret.
        self.regrets = [0.0]  # History of cumulative regret.

    def get_action(self):
        samples = [
            np.random.dirichlet(self.belief[i]) for i in range(self.env.k)
        ]  # exploit what agent believes about each arms' probas
        i = max(
            range(self.env.k), key=lambda k: samples[k][self.coi]
        )  # best rewarding arm for category of interest as far as bandit knows
        
        # update counts and actions
        self.counts[i] += 1
        self.actions.append(i)

        return i

    def generate_reward(self, i, sampled):  
        # update belief
        self.belief[i][sampled] += 1

        # recognize the reward
        if sampled == self.coi:
            reward = 1
        else:
            reward = 0

        # update regret
        self.regret += self.best_proba - self.env.probas[i][self.coi]
        self.regrets.append(self.regret)

        return reward

    @property
    def estimated_probas(self):
        return [
            self.belief[i][self.coi] / np.sum(self.belief[i])
            for i in range(self.env.k)
        ]
