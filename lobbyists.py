import numpy as np
import sys


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

    def __init__(self, env, coe, prior_temp):
        print("coe", coe)
        """
        coe: category of expertise
        env: CategoricalEnv instance
        """
        self.env = env
        self.prior_temp = prior_temp
        self.belief = np.ones(  # create concentration parameters for Dirichlet distribution for each arm.
            [
                self.env.k,
                self.env.c,
            ]  # k is number of arms, c is number of categories. So k is legislators and c is categories of topcis.
        )  # Initialize Dirichlet distribution's param \alpha to 1s. This is internal belief of the agent over slot-machines.

        if coe != -1:
            topicwise_true_probas = env.probas[:, coe]
            # topicwise_expertise_bias = [
            #     round(e, 1) * prior_temp + sys.float_info.epsilon
            #     for e in topicwise_true_probas  # +1 to prevent 0
            # ]  # it represents incomplete prior knowledge of the agent.
            topicwise_expertise_bias = [
                e * env.c * prior_temp + sys.float_info.epsilon
                for e in topicwise_true_probas  # +1 to prevent 0
            ]  # it represents incomplete prior knowledge of the agent.

            self.belief[:, coe] = topicwise_expertise_bias

    def update_belief(self, i, sampled):
        self.belief[i][sampled] += 1

    def estimated_probas(self, coi):
        return [self.belief[i][coi] / np.sum(self.belief[i]) for i in range(self.env.k)]
