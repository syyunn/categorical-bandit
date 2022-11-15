import matplotlib  # noqa

# matplotlib.use("Agg")  # noqa

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from bandits import CategoricalBandit
from lobbyists import CategoricalLobbyist
from env import CategoricalBanditEnv


def plot_results(bandit: CategoricalBandit, env: CategoricalBanditEnv, show=False):
    """
    Plot the results after playing in the env.
    """

    fig = plt.figure(figsize=(14, 8))
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)

    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    # ax6 = fig.add_subplot(236)

    # Sub.fig 1: Regrets in time.
    ax1.plot(range(len(bandit.regrets)), bandit.regrets, label="regret")
    ax1.plot(
        range(len(bandit.regrets)),
        [
            (bandit.best_proba - bandit.worst_proba) * i
            for i in range(len(bandit.regrets))
        ],
        label="upper bound of regret",
    )
    pass

    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Cumulative regret/reward")
    # ax1.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
    ax1.legend()
    ax1.grid("k", ls="--", alpha=0.3)
    ax1.legend()

    # Sub.fig. 4: Regrets in time in ratio.
    ax4.plot(
        range(len(bandit.regrets)),
        [
            (bandit.regrets[i] / ((bandit.best_proba - bandit.worst_proba) * i))
            for i in range(len(bandit.regrets))
        ],
        label="Regrets in time normalized into [0, 1]",
    )
    pass

    ax4.set_xlabel("Time step")
    ax4.set_ylabel("Normalized regret")
    # ax1.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
    ax4.grid("k", ls="--", alpha=0.3)
    ax4.set_yticks(np.arange(0, 1.1, 0.1))
    # ax1.legend()

    # Sub.fig. 2: Probabilities estimated by solvers.
    ax2.plot(
        range(env.k),
        [env.probas[i][bandit.coi] for i in range(env.k)],
        "k--",
        markeredgewidth=2,
        label="real prob",
    )
    ax2.plot(
        range(env.k),
        [bandit.estimated_probas[i] for i in range(env.k)],
        "x",
        markeredgewidth=2,
        label="estimated prob",
    )
    ax2.set_xlabel("Actions")
    ax2.set_ylabel("(Estimated) Probabilities")
    ax2.grid("k", ls="--", alpha=0.3)
    ax2.legend()

    # Sub.fig. 3: Action counts
    print("Best arm is: ", bandit.best_arm)
    print("Most frequently used arm: {}".format(np.argmax(bandit.counts)))
    if np.argmax(bandit.counts) == bandit.best_arm:
        print("Best arm found.")
    else:
        print("Best arm not found.")

    ax3.plot(
        range(env.k),
        np.array(bandit.counts) / float(len(bandit.regrets)),
        ls="--",
        lw=2,
    )
    ax3.set_xlabel("Actions")
    ax3.set_ylabel("# of trials (in ratio)")
    ax3.grid("k", ls="--", alpha=0.3)
    figname = "results_K{}_C{}_N{}.png".format(env.k, env.c, env.n)

    # Sub.fig. 5: Ratio between using the self belief and lobbyists' belief.
    df = pd.DataFrame({"freq": bandit.hires})
    df["freq"].value_counts().plot(
        ax=ax5, kind="bar", xlabel="lobbyist", ylabel="frequency"
    )  # Save & Show plot

    # Save & Show plot
    plt.savefig(figname)
    if show:
        plt.show()


def experiment(B, K, C, N, L=1, show=False):
    """
    Run a small experiment on solving a Categorical bandit with K slot machines,
    each with a randomly initialized reward probability.

    Args:
        B (int): number of bandits.
        K (int): number of slot machiens.
        C (int): number of categories.
        N (int): number of time steps to try.
        show (bool): whether to show the plot or not.
    """

    env = CategoricalBanditEnv(B, N, K, C, L)
    env.bandits = [
        CategoricalBandit(env) for _ in range(B)
    ]  # since we need env to initialize bandits, we need to do this after env is initialized
    env.lobbyists = [CategoricalLobbyist(env) for _ in range(L)]  # same as above

    env.run()

    plot_results(env.bandits[1], env, show=show)


if __name__ == "__main__":
    experiment(B=2, K=10, C=4, L=0, N=5000, show=True)
