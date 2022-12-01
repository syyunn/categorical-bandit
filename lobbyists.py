import numpy as np


class Lobbyist(object):
    def generate_reward(self, i):
        raise NotImplementedError


class CategoricalLobbyist(Lobbyist):
    """
    k: # of arms
    c: # of categories for each arm
    probas: custom prob. size of k*c
    coi: category of interest
    seed: random seed to generate underlying probabilities for all arms
    """

    def __init__(self, env):

        self.env = env
        self.belief = np.ones(  # create concentration parameters for Dirichlet distribution for each arm.
            [
                self.env.k,
                self.env.c,
            ]  # k is number of arms, c is number of categories. So k is legislators and c is categories of topcis.
        )  # Initialize Dirichlet distribution's param \alpha to 1s. This is internal belief of the agent over slot-machines.

    def update_belief(self, i, sampled):
        self.belief[i][sampled] += 1

    def estimated_probas(self, coi):
        return [self.belief[i][coi] / np.sum(self.belief[i]) for i in range(self.env.k)]
