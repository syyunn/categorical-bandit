import numpy as np


class CatEnv:
    def __init__(self, k, c, probas=None, seed=2139):
        """
        k (int): number of arms.
        c (int): number of categories.
        """

        self.k = k
        self.c = c
        self.seed = seed
        if probas is None:
            np.random.seed(self.seed)
            self.probas = np.random.dirichlet(np.ones(self.c), size=self.k)
        else:
            self.probas = probas
