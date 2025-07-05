"""
File containing code to plot the metrics of the human performance
with and without AI intervention.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from matplotlib.patches import Patch
import matplotlib.patches as mpatches

############################################################################
def human_easier_vs_harder_n_ts_end_boxplots(
    df: pd.DataFrame,
    configs: list[dict],
    *,
    version: str = "baseline",          # use "baseline", "inter", etc. as they appear in the df, for specifying human baseline vs under intervention
    easier_color: str = "#1f77b4",      
    harder_color: str = "#ff7f0e",      
    gap: float = 0.8,
    width: float = 0.35,
    figsize: tuple[int, int] = (12, 4),
    save: bool = False,
    show: bool = True,
    save_dir: str = "plots",
    filename: str = "n_ts_end_easier_vs_harder.png",
):
    """
    Make 1 side-by-side boxplot of human n ts to reach end of each chain
    for each config.

    For each config, draws two box-plots:

        blue   - n_ts_to_end_easier_<version>
        orange - n_ts_to_end_harder_<version>

    `configs` must be a list of dicts with keys
    ['label','p_fwd_chain_1','p_fwd_chain_2','p_fwd_stuck'].
    """

    # -------------------------------------------------------
    # prep
    # -------------------------------------------------------
    if save and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    plt.rcParams.update({
        'font.size': 12,           # Base font size
        'axes.titlesize': 14.5,      # Title font
        'axes.labelsize': 14,      # Axis label font
        'xtick.labelsize': 12,     # X tick label
        'ytick.labelsize': 12,     # Y tick label
        'legend.fontsize': 12      # Legend
    })

    box_stats, colours, positions = [], [], []
    x = 0

    for cfg in configs:
        p1, p2, ps = (cfg["p_fwd_chain_1"],
                      cfg["p_fwd_chain_2"],
                      cfg["p_fwd_stuck"])

        row = df.loc[
            (df["p_fwd_chain_1"] == p1) &
            (df["p_fwd_chain_2"] == p2) &
            (df["p_fwd_stuck"]  == ps)
        ]
        if row.empty:
            raise ValueError(f"No row matches config {cfg}")
        row = row.iloc[0]

        # easier (blue)
        root = f"n_ts_to_end_easier_{version}"
        box_stats.append({
            "med":    row[f"{root}_median"],
            "q1":     row[f"{root}_q1"],
            "q3":     row[f"{root}_q3"],
            "whislo": row[f"{root}_min"],
            "whishi": row[f"{root}_max"],
            "fliers": []
        })
        colours.append(easier_color)
        positions.append(x)
        x += width

        # harder (orange)
        root = f"n_ts_to_end_harder_{version}"
        box_stats.append({
            "med":    row[f"{root}_median"],
            "q1":     row[f"{root}_q1"],
            "q3":     row[f"{root}_q3"],
            "whislo": row[f"{root}_min"],
            "whishi": row[f"{root}_max"],
            "fliers": []
        })
        colours.append(harder_color)
        positions.append(x)

        # move to next config slot
        x += gap

    # -------------------------------------------------------
    # draw
    # -------------------------------------------------------
    result = ax.bxp(box_stats, positions=positions,
                    widths=width, patch_artist=True)

    for patch, col in zip(result["boxes"], colours):
        patch.set_facecolor(col)
        patch.set_alpha(0.9)

    # x-ticks centred on each (easier, harder) pair
    ticks = [i*(2*width+gap)+width/2 for i in range(len(configs))]
    labels = [cfg["label"] for cfg in configs]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Timesteps")
    ax.set_title("Timesteps to reach end of chain", loc="left")

    # legend
    legend_patches = [
        Patch(facecolor=easier_color, label="Easier chain"),
        Patch(facecolor=harder_color, label="Harder chain")
    ]
    ax.legend(handles=legend_patches, loc="upper right")

    # tidy up & output
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(save_dir, filename), dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)



def AI_only_metric_boxplots_solid_colors(
    df: pd.DataFrame,
    configs: list[dict],
    metrics: list[str] = ["n_interventions", "AI_rwds"],
    shared_yaxis: bool = True,
    save_combined: bool = False,
    save_individual: bool = False,
    save_dir: str = ".",
    dpi: int = 300,
    base_colors = None
):
    """
    Plots number of AI interventions and total discounted AI rewards using
    boxplots.  Makes 1 set of plots per config (for inputting policy class
    parameters).

    Plot grouped box-plots (one subplot per metric) from pre-computed five-number
    summaries stored in *df*.  Each config dict must contain the three p_fwd
    values PLUS a 'label'.

    Parameters
    ----------
    df : DataFrame
        One row per config with *_min/_q1/_median/_q3/_max columns.
    configs : list of dict
        Each dict = {label, p_fwd_chain_1, p_fwd_chain_2, p_fwd_stuck}.
    metrics : sequence of str
        Which metrics to plot (must exist in df as the *_min etc. columns).
    shared_yaxis : bool
        True = all subplots share the same y-axis.
    save_combined : bool
        Write one PNG with all subplots.
    save_individual : bool
        Write one PNG per subplot.
    save_dir : str | Path
        Folder where files are written (created if missing).
    dpi : int
        Resolution of saved PNGs.
    base_colors : dict {metric: hex}
        Override the default blue/orange bases if you like.

    NOTE: Example configs: 
    configs = [
        {"label": "Easy Only", "p_fwd_chain_1": 0.9,  "p_fwd_chain_2": 0.05, "p_fwd_stuck": 0.05},
        {"label": "Easy Prior", "p_fwd_chain_1": 0.05, "p_fwd_chain_2": 0.9,  "p_fwd_stuck": 0.05},
        {"label": "Hard Prior",   "p_fwd_chain_1": 0.05, "p_fwd_chain_2": 0.05, "p_fwd_stuck": 0.9},
    ]
    these are the sets of (p_fwd_chain_1, p_fwd_chain_2, p_fwd_stuck) to plot 
    side-by-side as grouped box-plot for a metric variable
    """
    plt.rcParams.update({
        "axes.labelsize": 16,      # Axis labels (x and y)
        # "axes.titlesize": 16,      # Title font size
        "xtick.labelsize": 11,     # X-axis tick labels
        "ytick.labelsize": 15,     # Y-axis tick labels
        # "legend.fontsize": 16,     # Legend
    })

    # Color prep
    if base_colors is None:
        base_colors = {"n_interventions": "#970097", "AI_rwds": "#708090"}

    n_cfg = len(configs)

    shades = {
        m: [base_colors[m]] * n_cfg
        for m in metrics
    }

    # ------------------------------------------------------------------
    # figure & axes
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(5 * len(metrics), 4), sharey=shared_yaxis
    )
    axes = np.atleast_1d(axes)

    # store everything we need to redraw later
    cached_data = {}

    for ax, metric in zip(axes, metrics):
        boxes, positions, colours = [], [], []

        for idx, cfg in enumerate(configs, start=1):
            row = df[
                (df["p_fwd_chain_1"] == cfg["p_fwd_chain_1"])
                & (df["p_fwd_chain_2"] == cfg["p_fwd_chain_2"])
                & (df["p_fwd_stuck"]   == cfg["p_fwd_stuck"])
            ].squeeze()

            boxes.append({
                "label":  cfg["label"],
                "whislo": row[f"{metric}_min"],
                "q1":     row[f"{metric}_q1"],
                "med":    row[f"{metric}_median"],
                "q3":     row[f"{metric}_q3"],
                "whishi": row[f"{metric}_max"],
                "fliers": [],
            })
            positions.append(idx)
            colours.append(shades[metric][idx - 1])

        # draw on combined axis
        bp = ax.bxp(boxes, positions=positions, widths=0.6,
                    showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], colours):
            patch.set_facecolor(c); patch.set_edgecolor("black")
        for part in ("whiskers", "caps", "medians"):
            for ln in bp[part]:
                ln.set_color("black")

        ax.set_xticks(positions)
        ax.set_xticklabels([c["label"] for c in configs], rotation=25, ha="right")
        ax.set_title(metric.replace("_", " ").title())
        if ax is axes[0]:
            ax.set_ylabel("Value")

        # cache for later single-subplot redraw
        cached_data[metric] = dict(boxes=boxes, positions=positions, colours=colours)

    fig.suptitle("AI Metrics by Human Policy Class", y=1.02, fontsize=14)
    fig.tight_layout()

    # -------------------------------------------------- saving
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if save_combined:
        combined_path = save_dir / "boxplots_combined.png"
        fig.savefig(combined_path, dpi=dpi, bbox_inches="tight")
        print(f"[saved] {combined_path.resolve()}")

    if save_individual:
        for metric in metrics:
            data = cached_data[metric]
            f_ind, ax_ind = plt.subplots(figsize=(4, 4), sharey=shared_yaxis)

            bp = ax_ind.bxp(data["boxes"],
                            positions=data["positions"],
                            widths=0.6,
                            showfliers=False,
                            patch_artist=True)
            for patch, c in zip(bp["boxes"], data["colours"]):
                patch.set_facecolor(c); patch.set_edgecolor("black")
            for part in ("whiskers", "caps", "medians"):
                for ln in bp[part]:
                    ln.set_color("black")

            ax_ind.set_xticks(data["positions"])
            ax_ind.set_xticklabels([c["label"] for c in configs],
                                   rotation=25, ha="right")
            ax_ind.set_title(metric.replace("_", " ").title())
            if shared_yaxis:
                ax_ind.set_ylim(axes[0].get_ylim())
            if metric == metrics[0]:
                ax_ind.set_ylabel(metric.replace("_", " ").title())

            f_ind.tight_layout()
            out = save_dir / f"boxplot_{metric}.png"
            f_ind.savefig(out, dpi=dpi, bbox_inches="tight")
            plt.close(f_ind)
            print(f"[saved] {out.resolve()}")

    plt.show()



def plot_freq_n_timesteps_compar_from_config(
        df: pd.DataFrame,
        configs: list,
        figsize: tuple = (13, 5),
        hatch: str = "///", # for baseline
        save_path: str = None,
        use_suptitle: bool = False):
    """
    Plots the baseline human vs human under AI intervention frequency of
    reaching the goal, dropping out, number of timesteps to reach end of
    easier chain, and again for harder chain.  Makes groups of 2 to have
    the baseline and under intervention side-by-side.

    For every config in ``configs``:
    • Left subplot: freq-make-goal (green) and freq-drop-out (red)
    • Right subplot: boxplots for time to easier and harder chain ends

    Baseline = hatched & left; Intervention = solid & right

    If ``save_path`` is provided, figure is saved to file; otherwise shown.

    NOTE: Example configs: 
    configs = [
        {"label": "Easy Only", "p_fwd_chain_1": 0.9,  "p_fwd_chain_2": 0.05, "p_fwd_stuck": 0.05},
        {"label": "Easy Prior", "p_fwd_chain_1": 0.05, "p_fwd_chain_2": 0.9,  "p_fwd_stuck": 0.05},
        {"label": "Hard Prior",   "p_fwd_chain_1": 0.05, "p_fwd_chain_2": 0.05, "p_fwd_stuck": 0.9},
    ]
    these are the sets of (p_fwd_chain_1, p_fwd_chain_2, p_fwd_stuck) to plot 
    side-by-side as grouped box-plot for a metric variable
    """
    # Set font sizes
    plt.rcParams.update({
        "axes.labelsize": 16,      # Axis labels (x and y)
        "axes.titlesize": 15,      # Title font size
        "xtick.labelsize": 15,     # X-axis tick labels
        "ytick.labelsize": 15,     # Y-axis tick labels
        "legend.fontsize": 16,     # Legend
    })

    def get_row(c):
        """Return the row matching the config."""
        mask = (
            (df["p_fwd_chain_1"] == c["p_fwd_chain_1"]) &
            (df["p_fwd_chain_2"] == c["p_fwd_chain_2"]) &
            (df["p_fwd_stuck"]   == c["p_fwd_stuck"])
        )
        sel = df[mask]
        if sel.empty:
            raise ValueError(f"No dataframe row matches config {c}")
        return sel.iloc[0]

    for cfg in configs:
        r = get_row(cfg)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=False)

        offset = 0.2

        # ── Subplot 1: Bar plot (reach goal, dropout) ────────────────────────
        bar_metrics = [
            ("freq_mk_goal",   "Reach goal",   "green"),
            ("freq_dropout",   "Dropout",     "red")
        ]

        xpos_bar = np.arange(len(bar_metrics))

        for i, (k, label, color) in enumerate(bar_metrics):
            base_val = r[f"{k}_baseline"]
            int_val  = r[f"{k}_inter"]

            ax1.bar(xpos_bar[i] - offset, base_val,
                    width=0.35, color=color, hatch=hatch,
                    label="Baseline" if i == 0 else None)

            ax1.bar(xpos_bar[i] + offset, int_val,
                    width=0.35, color=color,
                    label="Intervention" if i == 0 else None)

        ax1.set_xticks(xpos_bar)
        ax1.set_xticklabels([lbl for _, lbl, _ in bar_metrics], rotation=15)
        ax1.set_ylabel("Frequency")
        ax1.set_title("Outcome Frequencies")
        ax1.grid(axis="y", linestyle="--", alpha=0.5)

        # ── Subplot 2: Boxplot (timesteps to end) ───────────────────────────
        box_metrics = [
            ("n_ts_to_end_easier", "Easier Chain", "blue"),
            ("n_ts_to_end_harder", "Harder Chain", "orange")
        ]

        xpos_box = np.arange(len(box_metrics))

        for j, (k, label, color) in enumerate(box_metrics):
            stats_bl = dict(
                med   = r[f"{k}_baseline_median"],
                q1    = r[f"{k}_baseline_q1"],
                q3    = r[f"{k}_baseline_q3"],
                whislo= r[f"{k}_baseline_min"],
                whishi= r[f"{k}_baseline_max"],
                label ="")

            stats_in = dict(
                med   = r[f"{k}_inter_median"],
                q1    = r[f"{k}_inter_q1"],
                q3    = r[f"{k}_inter_q3"],
                whislo= r[f"{k}_inter_min"],
                whishi= r[f"{k}_inter_max"],
                label ="")

            ax2.bxp([stats_bl], positions=[xpos_box[j] - offset],
                    widths=0.35, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=color, hatch=hatch, linewidth=1.2),
                    medianprops=dict(linewidth=1.2))

            ax2.bxp([stats_in], positions=[xpos_box[j] + offset],
                    widths=0.35, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=color, linewidth=1.2),
                    medianprops=dict(linewidth=1.2))

        ax2.set_xticks(xpos_box)
        ax2.set_xticklabels([lbl for _, lbl, _ in box_metrics], rotation=15)
        ax2.set_ylabel("Timesteps")
        ax2.set_title("Timesteps to End")
        ax2.grid(axis="y", linestyle="--", alpha=0.5)

        # ── Legend (shared) ────────────────────────────────────────────────
        baseline_patch     = mpatches.Patch(facecolor="lightgray", hatch=hatch, label="Baseline")
        intervention_patch = mpatches.Patch(facecolor="lightgray", label="Intervention")
        # ax2.legend(handles=[baseline_patch, intervention_patch], loc="upper right")
        # ── Legend (shared, upper center) ─────────────────────────────────────
        baseline_patch     = mpatches.Patch(facecolor="lightgray", hatch=hatch, label="Baseline")
        intervention_patch = mpatches.Patch(facecolor="lightgray", label="Intervention")
        fig.legend(handles=[baseline_patch, intervention_patch],
                loc="lower center", bbox_to_anchor=(0.5, -0.02),
                ncol=2, frameon=False)

        if use_suptitle:
            fig.suptitle(f"Baseline vs Intervention – {cfg['label']}", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
        else:
            plt.tight_layout(rect=[0, 0.05, 1, 1])

        # ── Save or show ────────────────────────────────────────────────────
        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.show()
            plt.close()
            print(f"Saved plot to {save_path}")
        else:
            plt.show()