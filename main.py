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

    ax1 = fig.add_subplot(331)
    ax2 = fig.add_subplot(332)
    ax3 = fig.add_subplot(333)

    ax4 = fig.add_subplot(334)
    ax5 = fig.add_subplot(335)
    ax6 = fig.add_subplot(336)

    ax7 = fig.add_subplot(337)

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
    ax1.set_ylabel("Cumulative regret")
    # ax1.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
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
        best_arm_found_text = "Best arm found."
        print(best_arm_found_text)
    else:
        best_arm_found_text = "Best arm not found."
        print(best_arm_found_text)

    ax3.plot(
        range(env.k),
        np.array(bandit.counts) / float(len(bandit.regrets)),
        ls="--",
        lw=2,
    )

    ax3.set_xlabel("Actions")
    ax3.set_ylabel("# of trials (in ratio)")
    ax3.annotate(best_arm_found_text, xy=(0.5, 0.5), xycoords="axes fraction")
    ax3.grid("k", ls="--", alpha=0.3)
    figname = "results_K{}_C{}_N{}_B_{}_L{}_cois{}.png".format(
        env.k,
        env.c,
        env.n,
        len(env.bandits),
        env.l,
        "".join([str(i) for i in env.cois]),
    )

    # Sub.fig. 5: Ratio between using the self belief and lobbyists' belief.
    df = pd.DataFrame({"freq": bandit.hires})
    df["freq"].value_counts().plot(
        ax=ax5, kind="bar", xlabel="lobbyist", ylabel="frequency"
    )

    # Sub.fig 6: Cumulative reward.
    ax6.plot(range(len(bandit.regrets)), bandit.cum_rewards, label="cumulative reward")
    ax6.set_xlabel("Time step")
    ax6.set_ylabel("Cumulative reward")
    ax6.grid("k", ls="--", alpha=0.3)
    ax6.legend()

    # Sub.fig 7: Estimated Proba & Real Proba of Most Frequently Selected Lobbyist
    # Sub.fig. 2: Probabilities estimated by solvers.
    ax7.plot(
        range(env.k),
        [env.probas[i][bandit.coi] for i in range(env.k)],
        "k--",
        markeredgewidth=2,
        label="real prob",
    )
    ax7.plot(
        range(env.k),
        env.lobbyists[bandit.most_freq_hired_lobbyist].estimated_probas(bandit.coi),
        "x",
        markeredgewidth=2,
        label="estimated prob",
    )
    ax7.set_xlabel("Actions")
    ax7.set_ylabel("(Estimated) Probabilities")
    ax7.grid("k", ls="--", alpha=0.3)
    ax7.legend()

    # Save & Show plot
    plt.tight_layout()  # too add margins btw subplots
    plt.savefig(figname)
    if show:
        plt.show()


def experiment(B, K, C, N, L, cois, show=False, bandit_index_to_plot=0):
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

    env = CategoricalBanditEnv(B, N, K, C, L, cois)
    env.bandits = [
        CategoricalBandit(env, coi=coi) for _, coi in zip(range(B), cois)
    ]  # since we need env to initialize bandits, we need to do this after env is initialized
    env.lobbyists = [CategoricalLobbyist(env) for _ in range(L)]  # same as above

    env.run()

    plot_results(env.bandits[bandit_index_to_plot], env, show=show)


if __name__ == "__main__":
    # experiment(B=2, K=256, C=8, L=1, N=5000, show=True)
    # experiment(B=1, K=256, C=8, L=0, N=2500, show=False, bandit_index_to_plot=0)
    # K = 256
    # C = 32
    # B = [1, 2, 4, 8, 16, 32, 64]
    # L = [0, 1, 2, 4, 8]
    # N = [2500, 5000, 7500]
    # for b in B:
    #     for l in L:
    #         for n in N:
    #             experiment(B=b, K=K, C=C, L=l, N=n, show=False, bandit_index_to_plot=0)
    experiment(
        B=4,
        K=256,
        C=32,
        L=1,
        N=5000,
        cois=[0, 0, 1, 1],
        show=True,
        bandit_index_to_plot=0,
    )
