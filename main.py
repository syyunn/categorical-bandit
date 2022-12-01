import matplotlib  # noqa
from tqdm import tqdm

# matplotlib.use("Agg")  # noqa

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from bandits import CategoricalBandit
from lobbyists import CategoricalLobbyist
from env import CategoricalBanditEnv


def plot_results(bandit_index: int, env: CategoricalBanditEnv, show=False):
    """
    Plot the results after playing in the env.
    """

    bandit = env.bandits[bandit_index]

    fig = plt.figure(figsize=(14, 10))
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    ax1 = fig.add_subplot(431)
    ax2 = fig.add_subplot(432)
    ax3 = fig.add_subplot(433)

    ax4 = fig.add_subplot(434)
    ax5 = fig.add_subplot(435)
    ax6 = fig.add_subplot(436)

    ax7 = fig.add_subplot(437)
    ax8 = fig.add_subplot(438)

    ax9 = fig.add_subplot(439)
    ax10 = fig.add_subplot(4, 3, 10)

    ax11 = fig.add_subplot(4, 3, 11)

    fig.suptitle(
        f"Bandit{bandit_index}, B={env.b} w/ coi{bandit.coi}, L={env.l}, N={env.n}, C={env.c}",
        fontsize=16,
    )

    # Sub.fig 1: Regrets in time.
    ax1.set_title(f"Bandit{bandit_index}'s regret in time, coi={bandit.coi}")
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
    ax4.set_title("Normalized regrets in time")
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
    ax4.set_ylabel("cum_regret / lb of regret")
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
    ax2.set_title(f"Underlying reward distribution of each arm for coi {bandit.coi}")
    ax2.plot(
        range(env.k),
        [bandit.estimated_probas[i] for i in range(env.k)],
        "x",
        markeredgewidth=2,
        label="estimated prob",
    )
    ax2.set_xlabel("Arms (Legislators)")
    ax2.set_ylabel("(Estimated) Probabilities")
    ax2.grid("k", ls="--", alpha=0.3)
    ax2.legend()

    # Sub.fig. 3: Action counts
    ax3.set_title("Proprotion of actions taken by bandit")
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

    ax3.set_xlabel("Arms (Legislators)")
    ax3.set_ylabel("Proportion of actions taken")
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
    ax5.set_title("Frequency of using lobbyists")
    df = pd.DataFrame({"freq": bandit.hires})
    df["freq"].value_counts().sort_index().plot(
        ax=ax5, kind="bar", xlabel="lobbyist", ylabel="frequency"
    )

    # Sub.fig 6: Cumulative reward.
    ax6.set_title(f"Cumulative reward of bandit{bandit_index} in time")
    ax6.plot(range(len(bandit.regrets)), bandit.cum_rewards, label="cumulative reward")
    ax6.set_xlabel("Time step")
    ax6.set_ylabel("Cumulative reward")
    ax6.grid("k", ls="--", alpha=0.3)
    ax6.legend()

    # Sub.fig. 11: Mean Rewards of Entire Bandits.
    ax11.set_title(f"Mean rewards of entire bandits for each time step")
    ax11.plot(
        range(len(env.mean_rewards_of_bandits)),
        env.mean_rewards_of_bandits,
        "x",
        markeredgewidth=2,
        label="Mean of rewards of entire bandits",
    )
    ax11.set_xlabel("Time step")
    ax11.set_ylabel("Sum rewards")
    ax11.grid("k", ls="--", alpha=0.3)
    ax11.legend()

    # Sub.fig 7: Estimated Proba & Real Proba of Most Frequently Selected Lobbyist w/ bandit's coi.
    if env.l > 0:
        true_probs = [env.probas[i][bandit.coi] for i in range(env.k)]
        estimated_probs_lobbyist = env.lobbyists[
            bandit.most_freq_hired_lobbyist
        ].estimated_probas(bandit.coi)
        ax7.set_title(
            f"Lobbyist{bandit.most_freq_hired_lobbyist}coi{bandit.coi}mse{np.square(np.subtract(true_probs, estimated_probs_lobbyist)).mean()}"
        )
        ax7.plot(
            range(env.k),
            true_probs,
            "k--",
            markeredgewidth=2,
            label="real prob",
        )
        ax7.plot(
            range(env.k),
            estimated_probs_lobbyist,
            "x",
            markeredgewidth=2,
            label="estimated prob",
        )
        ax7.set_xlabel("Actions")
        ax7.set_ylabel("(Estimated) Probabilities")
        ax7.grid("k", ls="--", alpha=0.3)
        ax7.legend()

        # Sub.fig 8: Estimated Proba & Real Proba of Most Frequently Selected Lobbyist w/ opposite of bandit's coi.
        coi = int(abs(bandit.coi - 1))
        true_probs = [env.probas[i][coi] for i in range(env.k)]
        estimated_probs_lobbyist = env.lobbyists[
            bandit.most_freq_hired_lobbyist
        ].estimated_probas(coi)
        ax8.set_title(
            f"Lobbyist{bandit.most_freq_hired_lobbyist}coi{coi}mse{np.square(np.subtract(true_probs, estimated_probs_lobbyist)).mean()}"
        )
        ax8.plot(
            range(env.k),
            true_probs,
            "k--",
            markeredgewidth=2,
            label="real prob",
        )
        ax8.plot(
            range(env.k),
            estimated_probs_lobbyist,
            "x",
            markeredgewidth=2,
            label="estimated prob",
        )
        ax8.set_xlabel("Actions")
        ax8.set_ylabel("(Estimated) Probabilities")
        ax8.grid("k", ls="--", alpha=0.3)
        ax8.legend()

        # Sub.fig 9: Estimated Proba & Real Proba of Least Frequently Selected Lobbyist
        true_probs = [env.probas[i][bandit.coi] for i in range(env.k)]
        estimated_probs_lobbyist = env.lobbyists[
            bandit.least_freq_hired_lobbyist
        ].estimated_probas(bandit.coi)
        ax9.set_title(
            f"Lobbyist{bandit.least_freq_hired_lobbyist}coi{bandit.coi}mse{np.square(np.subtract(true_probs, estimated_probs_lobbyist)).mean()}"
        )
        ax9.plot(
            range(env.k),
            true_probs,
            "k--",
            markeredgewidth=2,
            label="real prob",
        )
        ax9.plot(
            range(env.k),
            estimated_probs_lobbyist,
            "x",
            markeredgewidth=2,
            label="estimated prob",
        )
        ax9.set_xlabel("Actions")
        ax9.set_ylabel("(Estimated) Probabilities")
        ax9.grid("k", ls="--", alpha=0.3)
        ax9.legend()

        # Sub.fig 10: Estimated Proba & Real Proba of Least Frequently Selected Lobbyist
        coi = int(abs(bandit.coi - 1))
        true_probs = [env.probas[i][coi] for i in range(env.k)]
        estimated_probs_lobbyist = env.lobbyists[
            bandit.least_freq_hired_lobbyist
        ].estimated_probas(coi)

        ax10.set_title(
            f"Lobbyist{bandit.least_freq_hired_lobbyist}coi{coi}mse{np.square(np.subtract(true_probs, estimated_probs_lobbyist)).mean()}"
        )
        ax10.plot(
            range(env.k),
            true_probs,
            "k--",
            markeredgewidth=2,
            label="real prob",
        )
        ax10.plot(
            range(env.k),
            estimated_probs_lobbyist,
            "x",
            markeredgewidth=2,
            label="estimated prob",
        )
        ax10.set_xlabel("Actions")
        ax10.set_ylabel("(Estimated) Probabilities")
        ax10.grid("k", ls="--", alpha=0.3)
        ax10.legend()

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
    assert B == len(cois)

    uniq_cois = np.unique(cois)
    print("uniq_cois", uniq_cois)

    env = CategoricalBanditEnv(B, N, K, C, L, cois)
    env.bandits = [
        CategoricalBandit(env, id=id, coi=coi) for id, coi in zip(range(B), cois)
    ]  # since we need env to initialize bandits, we need to do this after env is initialized
    env.lobbyists = [
        CategoricalLobbyist(env=env, coe=coe) for _, coe in zip(range(L), uniq_cois)
    ]  # same as above

    env.run()

    plot_results(bandit_index_to_plot, env, show=show)


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
    # B = 64
    # experiment(
    #     B=B,
    #     K=128,
    #     C=16,
    #     L=0,
    #     N=200,
    #     cois=[0] * int(B / 2) + [1] * int(B / 2),
    #     show=True,
    #     bandit_index_to_plot=B - 1,
    # )

    experiment(
        B=6,
        K=64,
        C=8,
        L=2,
        N=1000,
        cois=[0, 0, 0, 1, 1, 1],
        show=True,
        bandit_index_to_plot=0,
    )
