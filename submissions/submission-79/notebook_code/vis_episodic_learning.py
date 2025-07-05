"""
This file contains plotting functions for visualizing the results of episodic learning.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_qlearning_v_oracle_ai_rewards_by_lr(
    dfs_by_lr: dict[str, "pd.DataFrame"],
    oracle_df: "pd.DataFrame",
    p1: float, p2: float, ps: float,
    oracle_n_trials: int = 60,
    ci: float = 1.0,
    figsize: tuple[int, int] = (8, 5),
    savepath: str = None
) -> None:
    """
    Plots the total discounted reward gained by the AI under Q-learning
    at the end of each episode for each inputted df (which should each
    correspond to a different learning rate for learning rate tuning).  
    Plots its mean and error band (with minor smoothing).  Also plots
    against the oracle AI agent (which solves for its optimal policy
    using value iteration).

    Parameters
    ----------
    dfs : dict
        {label -> dataframe with cols ['trial','episode','AI_rwds_q', ...]}  
    oracle_df : pd.DataFrame
        Must include: 'p_fwd_chain_1', 'p_fwd_chain_2', 'p_fwd_stuck',
        and 'AI_rwds_avg', 'AI_rwds_std'.
    p1, p2, ps : float
        Values used to match the correct row in oracle_df.
    ci : float
        Confidence interval multiplier (e.g. 1.96 ≈ 95% CI if SE used).
    figsize : tuple
        Size of the figure in inches.
    """
    fig, ax = plt.subplots(figsize=figsize)
    colour_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # --- plot AI Q-learning rewards for each df ---------------------------------------------
    for i, (label, df) in enumerate(sorted(dfs_by_lr.items())):
        g = df.groupby("episode")["AI_rwds_q"]
        mean = g.mean().rolling(window=10, center=True).mean()
        std  = g.std()
        count = g.count()

        # Calculate standard error
        stderr = std / np.sqrt(count)
        episodes = mean.index.to_numpy()

        # NaN handling: mask where stderr is NaN
        valid = ~np.isnan(stderr)
        m, s = mean[valid].to_numpy(), stderr[valid].to_numpy()
        e = episodes[valid]

        colour = colour_cycle[i % len(colour_cycle)]

        ax.plot(e, m, label=label, lw=2, color=colour)
        ax.fill_between(e, m - ci*s, m + ci*s, color=colour, alpha=0.2, linewidth=0)

    # --- Oracle -----------------------------------------------------------
    oracle_row = oracle_df[
        (oracle_df["p_fwd_chain_1"] == p1) &
        (oracle_df["p_fwd_chain_2"] == p2) &
        (oracle_df["p_fwd_stuck"] == ps)
    ]

    if oracle_row.empty:
        print(f"[WARNING] No oracle row matches (p1={p1}, p2={p2}, ps={ps}). Skipping oracle line.")
    else:
        o_mean = oracle_row["AI_rwds_avg"].iloc[0]
        o_std  = oracle_row["AI_rwds_std"].iloc[0]

        if "n_trials" in oracle_row.columns:
            n_trials = oracle_row["n_trials"].iloc[0]
        else:
            n_trials = oracle_n_trials

        xmin, xmax = ax.get_xlim()
        ep = np.array([xmin, xmax])
        m = np.repeat(o_mean, 2)
        s = np.repeat(o_std / np.sqrt(n_trials), 2)

        ax.plot(ep, m, label="oracle", color="black", lw=2.5, ls="--", zorder=5)
        ax.fill_between(ep, m - ci*s, m + ci*s, color="black", alpha=0.1, zorder=4)

    # --- cosmetics ---------------------------------------------------------
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average AI reward per trial")
    ax.set_title("Q-learning vs Oracle: Learning-rate comparison")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    ax.margins(x=0)
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()


def plot_qlearning_v_oracle_stuck_freq_mk_goal(
    q_df: "pd.DataFrame",
    oracle_df: "pd.DataFrame",
    disengage_est_df: "pd.DataFrame",
    p1: float, p2: float, ps: float,
    colour,
    oracle_n_trials: int = 200,
    stuck_est_n_trials: int = 500,
    ci: float = 1.0,
    figsize: tuple[int, int] = (8, 5),
    savepath: str = None,
    suptitle: str = None
) -> None:
    """
    Plots the frequency of the human agent reaching the goal state under
    the AI using Q-learning. Plots with minor smoothing. Also plots
    against the oracle AI agent (which solves for its optimal policy
    using value iteration) and the AI agent under disengagement estimation.
    
    Parameters
    ----------
    q_df: pd.DataFrame
        dataframe with cols ['trial','episode','mk_goal_q', ...]
    oracle_df : pd.DataFrame
        Must include: 'p_fwd_chain_1', 'p_fwd_chain_2', 'p_fwd_stuck',
        and 'freq_mk_goal_inter'.
    disengage_est_df: pd.DataFrame
        Must include: 'p_fwd_chain_1', 'p_fwd_chain_2', 'p_fwd_stuck',
        and 'freq_mk_goal_seq'.
    p1, p2, ps : float
        Values used to match the correct row in oracle_df.
    ci : float
        Confidence interval multiplier (e.g. 1.96 ≈ 95% CI if SE used).
    figsize : tuple
        Size of the figure in inches.
    suptitle: str or None
        if not None: uses this as the suptitle
    """
    plt.rcParams.update({
        "axes.labelsize": 18,      # Axis labels (x and y)
        "axes.titlesize": 20,      # Title font size
        "xtick.labelsize": 20,     # X-axis tick labels
        "ytick.labelsize": 20,     # Y-axis tick labels
        "legend.fontsize": 20,     # Legend
    })
    fig, ax = plt.subplots(figsize=figsize)

    # --- learning-rate curves ---------------------------------------------
    g = q_df.groupby("episode")["mk_goal_q"]
    mean = g.mean().rolling(window=10, center=True).mean()
    std  = g.std() 
    count = g.count()

    # Calculate standard error
    stderr = std / np.sqrt(count)
    episodes = mean.index.to_numpy()

    # NaN handling: mask where stderr is NaN
    valid = ~np.isnan(stderr)
    m, s = mean[valid].to_numpy(), stderr[valid].to_numpy()
    e = episodes[valid]

    ax.plot(e, m, label="Q-learning", lw=2, color=colour)
    ax.fill_between(e, m - ci*s, m + ci*s, color=colour, alpha=0.2, linewidth=0)

    # --- Oracle -----------------------------------------------------------
    oracle_row = oracle_df[
        (oracle_df["p_fwd_chain_1"] == p1) &
        (oracle_df["p_fwd_chain_2"] == p2) &
        (oracle_df["p_fwd_stuck"] == ps)
    ]

    if oracle_row.empty:
        print(f"[WARNING] No oracle row matches (p1={p1}, p2={p2}, ps={ps}). Skipping oracle line.")
    else:
        o_mean = oracle_row["freq_mk_goal_inter"].iloc[0]

        if "n_trials" in oracle_row.columns:
            n_trials = oracle_row["n_trials"].iloc[0]
        else:
            n_trials = oracle_n_trials

        xmin, xmax = ax.get_xlim()
        ep = np.array([0, 1000])
        m = np.repeat(o_mean, 2)

        ax.plot(ep, m, label="oracle", color="black", lw=2.5, ls="--", zorder=5)

    # optimal AI but with disengagement estimation
    disengage_est_row = disengage_est_df[
        (disengage_est_df["p_fwd_chain_1"] == p1) &
        (disengage_est_df["p_fwd_chain_2"] == p2) &
        (disengage_est_df["p_fwd_stuck"] == ps)
    ]

    if disengage_est_row.empty:
        print(f"[WARNING] No disengagement row matches (p1={p1}, p2={p2}, ps={ps}). Skipping disengagement line.")
    else:
        o_mean = disengage_est_row["freq_mk_goal_seq"].iloc[0]

        if "n_trials" in disengage_est_row.columns:
            n_trials = disengage_est_row["n_trials"].iloc[0]
        else:
            n_trials = stuck_est_n_trials

        xmin, xmax = ax.get_xlim()
        ep = np.array([0, 1000])
        m = np.repeat(o_mean, 2)

        ax.plot(ep, m, label="diseng. est.", color="gray", lw=2.5, ls="--", zorder=5)

    # --- cosmetics ---------------------------------------------------------
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Frequency of Making Goal")
    ax.set_title("Q-learning vs Oracle: Frequency Make Goal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.margins(x=0)
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
    else:
        plt.tight_layout()
    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()




def plot_qlearning_v_oracle_stuck_n_interventions(
    q_df: pd.DataFrame,
    oracle_df: "pd.DataFrame",
    disengage_est_df: "pd.DataFrame",
    p1: float, p2: float, ps: float,
    colour,
    oracle_n_trials: int = 200,# 60,
    stuck_est_n_trials: int = 500,
    ci: float = 1.0,
    figsize: tuple[int, int] = (8, 5),
    savepath: str = None,
    suptitle: str = None
) -> None:
    """
    Plots the total number of interventions done by the AI under Q-learning
    by the end of each episode. 
    Plots its mean and error band (with minor smoothing).  Also plots
    against the oracle AI agent (which solves for its optimal policy
    using value iteration) and the AI agent under disengagement estimation.
    
    Parameters
    ----------
    q_df: pd.DataFrame
        dataframe with cols ['trial','episode','n_interventions_q', ...]}  
    oracle_df : pd.DataFrame
        Must include: 'p_fwd_chain_1', 'p_fwd_chain_2', 'p_fwd_stuck',
        and 'n_interventions_avg', 'n_interventions_std'
    disengage_est_df: pd.DataFrame
        Must include: 'p_fwd_chain_1', 'p_fwd_chain_2', 'p_fwd_stuck',
        and 'n_interventions_seq_avg', 'n_interventions_seq_std'
    p1, p2, ps : float
        Values used to match the correct row in oracle_df.
    ci : float
        Confidence interval multiplier (e.g. 1.96 ≈ 95% CI if SE used).
    figsize : tuple
        Size of the figure in inches.
    suptitle: str or None
        if not None: uses this as the suptitle
    """
    plt.rcParams.update({
        "axes.labelsize": 18,      # Axis labels (x and y)
        "axes.titlesize": 20,      # Title font size
        "xtick.labelsize": 20,     # X-axis tick labels
        "ytick.labelsize": 20,     # Y-axis tick labels
        "legend.fontsize": 20,     # Legend
    })

    fig, ax = plt.subplots(figsize=figsize)

    # --- q-learning curve ---------------------------------------------
    g = q_df.groupby("episode")["n_interventions_q"]
    mean = g.mean().rolling(window=10, center=True).mean()
    std  = g.std()
    count = g.count()

    # Calculate standard error
    stderr = std / np.sqrt(count)
    episodes = mean.index.to_numpy()

    # NaN handling: mask where stderr is NaN
    valid = ~np.isnan(stderr)
    m, s = mean[valid].to_numpy(), stderr[valid].to_numpy()
    e = episodes[valid]


    ax.plot(e, m, label="Q-learning", lw=2, color=colour)
    ax.fill_between(e, m - ci*s, m + ci*s, color=colour, alpha=0.2, linewidth=0)

    # --- Oracle -----------------------------------------------------------
    oracle_row = oracle_df[
        (oracle_df["p_fwd_chain_1"] == p1) &
        (oracle_df["p_fwd_chain_2"] == p2) &
        (oracle_df["p_fwd_stuck"] == ps)
    ]


    if oracle_row.empty:
        print(f"[WARNING] No oracle row matches (p1={p1}, p2={p2}, ps={ps}). Skipping oracle line.")
    else:
        o_mean = oracle_row["n_interventions_avg"].iloc[0]
        o_std  = oracle_row["n_interventions_std"].iloc[0]

        if "n_trials" in oracle_row.columns:
            n_trials = oracle_row["n_trials"].iloc[0]
        else:
            n_trials = oracle_n_trials

        xmin, xmax = ax.get_xlim()
        ep = np.array([0, 1000])
        m = np.repeat(o_mean, 2)
        s = np.repeat(o_std / np.sqrt(n_trials), 2)

        ax.plot(ep, m, label="oracle", color="black", lw=2.5, ls="--", zorder=5)
        ax.fill_between(ep, m - ci*s, m + ci*s, color="black", alpha=0.1, zorder=4)

    # Optimal AI but with disengagement estimation
    disengage_est_row = disengage_est_df[
        (disengage_est_df["p_fwd_chain_1"] == p1) &
        (disengage_est_df["p_fwd_chain_2"] == p2) &
        (disengage_est_df["p_fwd_stuck"] == ps)
    ]
    if disengage_est_row.empty:
        print(f"[WARNING] No oracle row matches (p1={p1}, p2={p2}, ps={ps}). Skipping oracle line.")
    else:
        o_mean = disengage_est_row["n_interventions_seq_avg"].iloc[0]
        o_std  = disengage_est_row["n_interventions_seq_std"].iloc[0]

        if "n_trials" in disengage_est_row.columns:
            n_trials = disengage_est_row["n_trials"].iloc[0]
        else:
            n_trials = stuck_est_n_trials

        xmin, xmax = ax.get_xlim()
        ep = np.array([0, 1000])
        m = np.repeat(o_mean, 2)
        s = np.repeat(o_std / np.sqrt(n_trials), 2)

        ax.plot(ep, m, label="diseng. est.", color="gray", lw=2.5, ls="--", zorder=5)
        ax.fill_between(ep, m - ci*s, m + ci*s, color="gray", alpha=0.1, zorder=4)

    # --- cosmetics ---------------------------------------------------------
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Num Interventions")
    ax.set_title("Q-learning vs Oracle: Num Interventions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.margins(x=0)
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
    else:
        plt.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show()