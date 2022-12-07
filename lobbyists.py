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

    def __init__(self, env, coe, prior_temp, legiswise, seed_lobbyist):
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

        #random prior
        # np.random.seed(
        #         1
        #     )
        # self.belief = np.random.dirichlet(
        #         np.ones(self.env.c), size=self.env.k
        #     )

        if coe != -1:
            print("prior_temp", prior_temp)
            for row in range(env.k):
                self.belief[row, :] = (env.c - env.probas[row, coe] * env.c * prior_temp)/ (env.c-1) 
                self.belief[row, coe] = env.probas[row, coe] * env.c * prior_temp

            # for sanity check
            print(self.belief[50, coe])
            print(env.probas[50, coe] * env.c * prior_temp)

            print(sum(self.belief[50,:]))

        # know everything
        # for row in range(env.k):
        #     self.belief = env.probas * env.c # this makes specialization

        # know one very well
        if legiswise:
            print(f"legiswise is {legiswise}")
            self.belief = np.ones(  # create concentration parameters for Dirichlet distribution for each arm.
                [
                    self.env.k,
                    self.env.c,
                ]  # k is number of arms, c is number of categories. So k is legislators and c is categories of topcis.
            ) # reset  
            best_arm = np.argmax(self.env.probas[:, coe])
            self.belief[best_arm, coe] = env.probas[best_arm, coe] * env.c * prior_temp
            for col in range(env.c):
                if col == coe:
                    continue
                else:
                    self.belief[best_arm, col] = env.probas[best_arm, col] * env.c + env.probas[best_arm, coe] * env.c * (1-prior_temp) * (1/(env.c-1))
            
            print("legiswise sanity check", sum(self.belief[best_arm, :]), "should be close to", env.c)

        # # random prior
        # if seed_lobbyist is not None:
        #     np.random.seed(
        #             seed_lobbyist
        #         )
        #     self.belief = np.random.dirichlet(
        #             np.ones(self.env.c), size=self.env.k
        #         )


    def update_belief(self, i, sampled):
        self.belief[i][sampled] += 1

    def estimated_probas(self, coi):
        return [self.belief[i][coi] / np.sum(self.belief[i]) for i in range(self.env.k)]
