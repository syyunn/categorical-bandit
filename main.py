import matplotlib  # noqa

# matplotlib.use("Agg")  # noqa

import matplotlib.pyplot as plt
import numpy as np

from bandits import CategoricalBandit
from solvers import Solver, ThompsonSamplingCategorical


def plot_results(solvers, solver_names, figname, show):
    """
    Plot the results by multi-armed bandit solvers.

    Args:
        solvers (list<Solver>): All of them should have been fitted.
        solver_names (list<str)
        figname (str)
    """
    assert len(solvers) == len(solver_names)
    assert all(map(lambda s: isinstance(s, Solver), solvers))
    assert all(map(lambda s: len(s.regrets) > 0, solvers))

    b = solvers[0].bandit

    fig = plt.figure(figsize=(14, 8))
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)

    ax4 = fig.add_subplot(234)
    # ax5 = fig.add_subplot(235)
    # ax6 = fig.add_subplot(236)

    # Sub.fig. 1: Regrets in time.
    for i, s in enumerate(solvers):
        ax1.plot(range(len(s.regrets)), s.regrets, label="regret")
        ax1.plot(
            range(len(s.regrets)),
            [(s.bandit.best_proba-s.bandit.worst_proba) * i for i in range(len(s.regrets))],
            label="upper bound of regret"
        )
        pass

    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Cumulative regret")
    # ax1.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
    ax1.grid("k", ls="--", alpha=0.3)
    ax1.legend()

    # Sub.fig. 4: Regrets in time in ratio.
    for i, s in enumerate(solvers):
        # ax1.plot(range(len(s.regrets)), s.regrets, label="Regret")
        ax4.plot(
            range(len(s.regrets)),
            [(s.regrets[i] / ((s.bandit.best_proba-s.bandit.worst_proba) * i)) for i in range(len(s.regrets))],
            # label="Upper bound of regret"
        )
        pass

    ax4.set_xlabel("Time step")
    ax4.set_ylabel("Normalized Cumulative regret")
    # ax1.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
    ax4.grid("k", ls="--", alpha=0.3)
    ax4.set_yticks(np.arange(0, 1, 0.1))
    # ax1.legend()

    # Sub.fig. 2: Probabilities estimated by solvers.
    ax2.plot(range(b.k), [b.probas[i][b.coi] for i in range(b.k)], "k--", markeredgewidth=2, label="real prob")
    for s in solvers:
        ax2.plot(
            range(b.k),
            [s.estimated_probas[i] for i in range(b.k)],
            "x",
            markeredgewidth=2,
            label="estimated prob",
        )
    # ax2.set_xlabel("Actions sorted by " + r"$\theta$")
    ax2.set_xlabel("Actions by " + r"$\theta$")
    ax2.set_ylabel("Estimated")
    ax2.grid("k", ls="--", alpha=0.3)
    ax2.legend()

    # Sub.fig. 3: Action counts
    for s in solvers:
        print("Most frequently used arm: {}".format(np.argmax(s.counts)))
        if np.argmax(s.counts) == b.best_arm:
            print("Best arm found.")
        else:
            print("Best arm not found.")
        
        # ax3.plot(range(b.n), np.array(s.counts) / float(len(solvers[0].regrets)), ls='steps', lw=2)
        ax3.plot(
            range(b.k),
            np.array(s.counts) / float(len(solvers[0].regrets)),
            ls="--",
            lw=2,
        )
    ax3.set_xlabel("Actions")
    ax3.set_ylabel("# of trials (in ratio)")
    ax3.grid("k", ls="--", alpha=0.3)
    plt.savefig(figname)
    if show:
        plt.show()


def experiment(K, C, N, show=False):
    """
    Run a small experiment on solving a Categorical bandit with K slot machines,
    each with a randomly initialized reward probability.

    Args:
        K (int): number of slot machiens.
        C (int): number of categories.
        N (int): number of time steps to try.
    """

    b = CategoricalBandit(K, C)
    print("Config: K={}, C={}, N={}".format(K, C, N))
    # print("Randomly generated Categorical bandit has reward probabilities:\n", b.probas)
    print(
        "The best machine has index {} for category of interest {} and best proba {}".format(
            b.best_arm, b.coi, b.best_proba
        )
    )

    test_solvers = [ThompsonSamplingCategorical(b)]
    names = ["Thompson Sampling (Categorical)"]

    for s in test_solvers:
        s.run(N)

    plot_results(test_solvers, names, "results_K{}_C{}_N{}.png".format(K, C, N), show)


if __name__ == "__main__":
    for i in range(12):
        experiment(10, 2**(i+1), 5000, show=False)
