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

    def __init__(self, env, id, coi=1):
        self.id = id  # bandit's id

        self.env = env
        self.seed = self.env.seed
        self.coi = coi
        self.ncoi = [i for i in range(self.env.c)].remove(
            self.coi
        )  # bandit doesn't interest in any categories in self.ncoi

        self.best_arm = np.argmax(self.env.probas[:, self.coi])
        self.best_proba = max(self.env.probas[:, self.coi])
        self.worst_proba = min(self.env.probas[:, self.coi])

        self.belief = np.ones(  # create concentration parameters for Dirichlet distribution for each arm.
            [
                self.env.k,
                self.env.c,
            ]  # k is number of arms, c is number of categories. So k is legislators and c is categories of topcis.
        )  # Initialize Dirichlet distribution's param \alpha to 1s. This is internal belief of the agent over slot-machines.

        self.cum_rewards = [0]
        self.counts = [0] * self.env.k  # how many times each arm is pulled
        self.actions = []  # A list of machine ids, 0 to k-1.
        self.hires = []  # A history of hiring lobbyists or not.
        self.regret = 0.0  # Cumulative regret.
        self.regrets = [0.0]  # History of cumulative regret.
        self.counts_lobbyists = [
            0
        ] * self.env.l  # how many times each lobbyist is pulled

    def get_action(self):
        def _exploit(belief, coi=self.coi):
            """
            Return best arm based on the sampled proba from the given parameter set of belief.
            """
            samples = [
                np.random.dirichlet(belief[k]) for k in range(self.env.k)
            ]  # exploit what agent believes about each arms' probas - sample from the belief posterior which is reprsented by Dirichlet distribution.
            # sample is a list of k arrays, each array is a sample from the Dirichlet distribution of the k-th arm. So the shape is k*c.
            i = max(
                range(self.env.k), key=lambda k: samples[k][coi]
            )  # i is the best rewarding arm for category of interest based on samples
            return (
                i,  # which is the best arm
                samples[i][coi],  # with which probablity of category of interest.
            )  # tuple of selection of arm among k arms and its proba

        candidates = []
        candidates.append(_exploit(self.belief))  # choice from bandit's own belief
        for l in range(
            self.env.l
        ):  # choice from lobbyists' belief # assume that each bandit can access to every lobbyist's belief.
            candidates.append(_exploit(self.env.lobbyists[l].belief, coi=self.coi))

        c = max(
            range(len(candidates)), key=lambda c: candidates[c][1]
        )  # select the best action among all candidates where the candidates = [bandit's own belief, lobbyists' belief]
        # c refer to candidate

        i = candidates[c][0]  # i should be one of among k arms
        l = (
            c - 1
        )  # l should be among l lobbyists; if l=-1, then it means bandit's using own belief and not that of a lobbyist.

        self.counts[i] += 1
        self.actions.append(i)

        self.hires.append(
            l
        )  # history of hiring lobbyists. if l=-1, then it means not hiring lobbyists.

        if l != -1:
            self.counts_lobbyists[l] += 1
            self.env.counts_lobbyists[l] += 1  # update global counter as well
        return i, l  # return tuple as (action, lobbyist)

    def generate_reward(self, i, l, sampled):
        if l == -1:
            # update belief
            self.belief[i][sampled] += 1

        # recognize the reward
        if (
            sampled == self.coi
        ):  # if my category of interest coincides with sampled choice, then the agent gets reward 1.
            reward = 1
        else:  # else reward is 0.
            reward = 0

        # update regret
        self.regret += self.best_proba - self.env.probas[i][self.coi]
        self.regrets.append(self.regret)
        self.cum_rewards.append(self.cum_rewards[-1] + reward)

        return reward

    @property
    def estimated_probas(self):
        return [
            self.belief[i][self.coi] / np.sum(self.belief[i]) for i in range(self.env.k)
        ]

    @property
    def most_freq_hired_lobbyist(self):
        # print(set(self.hires))
        return max(set(self.hires), key=self.hires.count)

    @property
    def least_freq_hired_lobbyist(self):
        return min(set(self.hires), key=self.hires.count)
