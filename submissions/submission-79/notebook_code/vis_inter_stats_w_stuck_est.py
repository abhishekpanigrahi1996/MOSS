"""
File containing code to plot the metrics of the human performance
with and without AI intervention, as well as AI intervention using
the iterative and stuck estimators.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import to_rgb
from pathlib import Path
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec


def plot_n_inters_ts_droput_freq_goal_dropout_compar_seq(df, config: dict, save_path=None, save_subplots=False):
    """
    Creates 3 subplots comparing the human baseline to under AI intervention
    to under AI intervention with stuck estimation via the sequential estimator.
    Makes boxplots for n interventions (first subplot), ts to dropout (second
    subplot).  Makes bar plots for goal/dropout frequency (third subplot).

    NOTE: this version is without ts to goal and is for sequential estimator only.
    
    Three-panel comparison: interventions, ts to dropout, goal/dropout frequency.

    The three version markings are Baseline (// hatch), Oracle (solid) and
    Seq Est (.. hatch). 

    NOTE: config should have keys 'p_fwd_chain_1', 'p_fwd_chain_2', 'p_fwd_stuck'
    and is for selecting the right parameter configuration from the df to select
    that row and plot its statistics.
    """
    plt.rcParams.update({
        "axes.labelsize": 16,      # Axis labels (x and y)
        "axes.titlesize": 16,      # Title font size
        "xtick.labelsize": 14,     # X-axis tick labels
        "ytick.labelsize": 15,     # Y-axis tick labels
        "legend.fontsize": 16,     # Legend
    })
    # ── pull the matching row ────────────────────────────────────────────────
    row = df[
        (df["p_fwd_chain_1"] == config["p_fwd_chain_1"]) &
        (df["p_fwd_chain_2"] == config["p_fwd_chain_2"]) &
        (df["p_fwd_stuck"]   == config["p_fwd_stuck"])
    ].iloc[0]

    # ── figure & axes (narrow first two panels) ─────────────────────────────
    fig = plt.figure(figsize=(11, 4))
    gs  = fig.add_gridspec(1, 3, width_ratios=[0.8, 0.8, 1.0], wspace=0.28)
    ax_int   = fig.add_subplot(gs[0])
    ax_box   = fig.add_subplot(gs[1])
    ax_freq  = fig.add_subplot(gs[2])

    hatches = {"Baseline": "//", "Oracle": "", "Seq Est": ".."}

    # ─────────────────────────────────────────────────────────────────────────
    # 1) Average number of interventions (only Oracle & Seq Est)
    # ─────────────────────────────────────────────────────────────────────────
    bar_width = 0.55
    ax_int.bar(
        ["Oracle", "Seq Est"],
        [row["n_interventions_avg"], row["n_interventions_seq_avg"]],
        color="#970097",
        hatch=[hatches["Oracle"], hatches["Seq Est"]],
        width=bar_width
    )
    ax_int.set_title("Avg # Interventions")
    ax_int.set_ylabel("Avg # Interventions")

    # ─────────────────────────────────────────────────────────────────────────
    # 2) Timesteps to dropout (box-plots with caps)
    # ─────────────────────────────────────────────────────────────────────────
    def five_num(prefix):
        return dict(
            med   = row[f"{prefix}_median"],
            q1    = row[f"{prefix}_q1"],
            q3    = row[f"{prefix}_q3"],
            whislo= row[f"{prefix}_min"],
            whishi= row[f"{prefix}_max"],
            label = "",
        )

    stats = [
        five_num("n_ts_to_dropout_baseline"),
        five_num("n_ts_to_dropout_inter"),
        five_num("n_ts_to_dropout_seq"),
    ]
    positions = [1, 2, 3]

    # draw each box separately so we can colour/hatch them
    for pos, stat, lbl in zip(positions, stats, ["Baseline", "Oracle", "Seq Est"]):
        ax_box.bxp(
            [stat], positions=[pos], widths=0.55, patch_artist=True,
            showfliers=False,
            boxprops   = dict(facecolor="#D94D02",
                              hatch=hatches[lbl],
                              linewidth=1.1),
            medianprops= dict(linewidth=1.3),
            whiskerprops=dict(linewidth=1.1),
            capprops    =dict(linewidth=1.1)
        )

    ax_box.set_xticks(positions)
    ax_box.set_xticklabels(["Baseline", "Oracle", "Seq Est"])
    ax_box.set_title("Timesteps to Dropout")
    ax_box.set_ylabel("Timesteps")

    # ─────────────────────────────────────────────────────────────────────────
    # 3) Frequencies of goal / dropout
    # ─────────────────────────────────────────────────────────────────────────
    freq_labels = ["Goal", "Dropout"]
    x = np.arange(len(freq_labels))          # [0,1]
    width = 0.22                             # bar thickness
    offset = [-width, 0, width]              # baseline left, oracle mid, seq right
    method_order = ["Baseline", "Oracle", "Seq Est"]

    freq_data = {
        "Baseline": [row["freq_mk_goal_baseline"], row["freq_dropout_baseline"]],
        "Oracle"  : [row["freq_mk_goal_inter"],    row["freq_dropout_inter"]],
        "Seq Est" : [row["freq_mk_goal_seq"],      row["freq_dropout_seq"]],
    }
    var_colour = {0: "green", 1: "red"}

    for m_idx, method in enumerate(method_order):
        for j, var in enumerate(freq_labels):
            ax_freq.bar(
                x[j] + offset[m_idx],
                freq_data[method][j],
                width=width,
                color=var_colour[j],
                hatch=hatches[method],
                edgecolor="black" if hatches[method] else None,
                label=method if (j == 0) else None,
            )

    ax_freq.set_xticks(x)
    ax_freq.set_xticklabels(freq_labels)
    ax_freq.set_ylim(0, 1)
    ax_freq.set_title("Outcome Frequencies")
    ax_freq.set_ylabel("Frequency")

    # ─────────────────────────────────────────────────────────────────────────
    # Shared legend below all subplots
    # ─────────────────────────────────────────────────────────────────────────
    leg_patches = [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="Baseline"),
        mpatches.Patch(facecolor="white", edgecolor="black", label="Oracle"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="..", label="Seq Est"),
    ]

    fig.tight_layout()  # Do initial layout
    fig.subplots_adjust(top=0.82, bottom=0.25)  # Reserve space for title and legend
    fig.suptitle(f"{config['label']}", fontsize=16)

    # Legend
    fig.legend(handles=leg_patches, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02))

    # ── save full figure ────────────────────────────────────────────────────
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    # ── optionally save each subplot seperately ─────────────────────────────
    if save_subplots:
        names = ["interventions", "timesteps_dropout", "frequencies"]
        for ax, name in zip([ax_int, ax_box, ax_freq], names):
            sp_path = (
                f"{save_path.rsplit('.',1)[0]}_{name}.png"
                if save_path else f"{name}.png"
            )
            ax.figure.savefig(sp_path, dpi=300, bbox_inches="tight")

    plt.show()



def plot_n_inters_ts_outcome_freq_outcome_compar_seq(df, config: dict, save_path=None, save_subplots=False):
    """
    Creates 3 subplots comparing the human baseline to under AI intervention
    to under AI intervention with stuck estimation via the sequential estimator.
    Makes boxplots for n interventions (first subplot), ts to dropout *and goal* (second
    subplot).  Makes bar plots for goal/dropout frequency (third subplot).

    NOTE: this version is WITH ts to goal and is for sequential estimator only.
    
    Three-panel comparison: interventions, ts to dropout, goal/dropout frequency.

    Version markings:
      • Baseline  (// hatch)
      • Oracle    (solid)
      • Seq Est   (.. hatch)

    NOTE: config should have keys 'p_fwd_chain_1', 'p_fwd_chain_2', 'p_fwd_stuck'
    and is for selecting the right parameter configuration from the df to select
    that row and plot its statistics.
    """  
    plt.rcParams.update({
        "axes.labelsize": 16,      # Axis labels (x and y)
        "axes.titlesize": 16,      # Title font size
        "xtick.labelsize": 15,     # X-axis tick labels
        "ytick.labelsize": 15,     # Y-axis tick labels
        "legend.fontsize": 16,     # Legend
    })
    # ------------------------------------------------------------------
    # Config parsing
    # ------------------------------------------------------------------
    method_names  = config.get("method_names",  ["baseline", "inter", "seq"])
    method_labels = config.get("method_labels", {"baseline": "Baseline",
                                                 "inter":    "Oracle",
                                                 "seq":      "Seq Est"})
    method_hatch  = config.get("method_hatch",  {"baseline": "//",
                                                 "inter":    "",
                                                 "seq":      ".."})

    var_colors_freq = config.get("var_colors_freq", {"goal": "#00AA66", "dropout": "#BB2222"})
    var_colors_ts   = config.get("var_colors_ts",   {"goal": "#77D9A8", "dropout": "#D97575"})
    var_colors_interventions = config.get("var_colors_interventions", "#9B1B9BB7")

    suffix_map  = {"baseline": "_baseline", "inter": "_inter", "seq": "_seq"}
    suffix_intv = {"inter": "", "seq": "_seq"}

    # fig, axs = plt.subplots(1, 3, figsize=(13, 4.5), gridspec_kw={'wspace': 0.25})
    fig = plt.figure(figsize=(13, 4.5))
    gs = GridSpec(1, 3, width_ratios=[0.3, 1, 1], wspace=0.3)
    axs = [fig.add_subplot(gs[i]) for i in range(3)]

    # Select correct row from df based on config parameters
    row = df[
        (df["p_fwd_stuck"] == config["p_fwd_stuck"]) &
        (df["p_fwd_chain_1"] == config["p_fwd_chain_1"]) &
        (df["p_fwd_chain_2"] == config["p_fwd_chain_2"])
    ]
    if row.empty:
        raise ValueError("No matching row found in DataFrame for the given config.")

    # ==============================================================
    # 1. Interventions –– boxplots side-by-side, wider, with hatching
    # ==============================================================
    box_data, positions, pos_labels = [], [], []

    for i, method in enumerate(["inter", "seq"]):
        suf = suffix_intv[method]
        box_data.append({
            'med':    row[f"n_interventions{suf}_median"].values[0],
            'q1':     row[f"n_interventions{suf}_q1"].values[0],
            'q3':     row[f"n_interventions{suf}_q3"].values[0],
            'whislo': row[f"n_interventions{suf}_min"].values[0],
            'whishi': row[f"n_interventions{suf}_max"].values[0],
            'fliers': []})
        positions.append(i * 0.24)
        pos_labels.append(method_labels[method])

    ret = axs[0].bxp(box_data,
                     positions=positions,
                    #  widths=0.4,
                     widths=0.24,
                     patch_artist=True,
                     boxprops=dict(facecolor=var_colors_interventions, edgecolor='black'),
                     medianprops=dict(color='black', linewidth=2),
                     whiskerprops=dict(color='black'),
                     capprops=dict(color='black'))

    for idx, patch in enumerate(ret['boxes']):
        patch.set_hatch(method_hatch[["inter", "seq"][idx]])

    axs[0].set_xticks([])
    axs[0].set_xticklabels([])
    axs[0].tick_params(axis='x', bottom=False)  # Turn off x-axis ticks entirely

    # --- tighten horizontal padding, but keep boxes centred ---
    box_w   = 0.24                     # the width you pass to bxp
    pad     = 0.08                     # tiny extra breathing room
    left    = positions[0] - box_w/2 - pad
    right   = positions[-1] + box_w/2 + pad

    axs[0].set_xlim(left, right)       # symmetrical, so contents stay centered
    axs[0].margins(x=0)                # turn off the auto-padding Matplotlib adds
    axs[0].set_ylabel("# Interventions")
    axs[0].set_title("# Interventions")

    # ==============================================================
    # 2. Timesteps to Outcome –– boxplots with hatching, wider
    # ==============================================================
    outcomes     = ["goal", "dropout"]
    spacing = 0.32
    offset       = 0
    box_data_ts  = []
    positions_ts = []
    xtick_pos    = []

    for outcome in outcomes:
        for j, method in enumerate(method_names):
            suf = suffix_map[method]
            prefix = f"n_ts_to_{outcome}{suf}"
            box_data_ts.append({
                'med':    row[f"{prefix}_median"].values[0],
                'q1':     row[f"{prefix}_q1"].values[0],
                'q3':     row[f"{prefix}_q3"].values[0],
                'whislo': row[f"{prefix}_min"].values[0],
                'whishi': row[f"{prefix}_max"].values[0],
                'fliers': []})
            positions_ts.append(offset + j * spacing)
        xtick_pos.append(offset + spacing)
        offset += spacing * len(method_names) + 0.2

    for idx, box in enumerate(box_data_ts):
        outcome_idx = idx // len(method_names)
        method_idx  = idx %  len(method_names)
        method      = method_names[method_idx]

        axs[1].bxp([box],
                   positions=[positions_ts[idx]],
                   #    widths=0.25,
                   widths=0.32,
                   patch_artist=True,
                   boxprops=dict(facecolor=list(var_colors_ts.values())[outcome_idx],
                                 edgecolor='black',
                                 hatch=method_hatch[method]),
                   medianprops=dict(color='black', linewidth=2),
                   whiskerprops=dict(color='black'),
                   capprops=dict(color='black'))

    axs[1].set_xticks(xtick_pos)
    axs[1].set_xticklabels(["Goal", "Dropout"])
    axs[1].set_ylabel("Timesteps")
    axs[1].set_title("Timesteps to Outcome")

    # ==============================================================
    # 3. Outcome Frequencies –– bars, wider, tighter spacing
    # ==============================================================
    x          = np.arange(len(outcomes))
    bar_width  = 0.24

    for i, method in enumerate(method_names):
        freqs = [row[f"freq_mk_goal_{method}"].values[0],
                 row[f"freq_dropout_{method}"].values[0]]
        axs[2].bar(x + (i - 1) * (bar_width * 0.9), freqs,
                   width=bar_width,
                   label=method_labels[method],
                   color=[var_colors_freq[o] for o in outcomes],
                   hatch=method_hatch[method],
                   edgecolor='black')

    axs[2].set_xticks(x)
    axs[2].set_xticklabels(["Goal", "Dropout"])
    axs[2].set_ylabel("Frequency")
    axs[2].set_title("Outcome Frequencies")

    # --------------------------------------------------------------
    # Final formatting / saving
    # --------------------------------------------------------------
    legend_handles = [
        Patch(facecolor='white', edgecolor='black', hatch=method_hatch[m], label=method_labels[m])
        for m in method_names
    ]

    fig.legend(handles=legend_handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout(rect=[0.01, 0.06, 0.99, 0.98])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')

        if save_subplots:
            subplot_dir = os.path.splitext(save_path)[0] + "_subplots"
            os.makedirs(subplot_dir, exist_ok=True)
            for i, ax in enumerate(axs):
                ext = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(os.path.join(subplot_dir, f"subplot_{i+1}.png"), bbox_inches=ext)

    plt.show()    


def plot_n_inters_ts_outcome_freq_outcome_compar_seq_iter(df, config: dict, 
                                                          save_path: str=None, save_subplots: bool=False,
                                                          use_suptitle: bool = False):
    """
    Creates 3 subplots comparing the human baseline to under AI intervention
    to under AI intervention with stuck estimation via the sequential estimator.
    Makes boxplots for n interventions (first subplot), ts to dropout *and goal* (second
    subplot).  Makes bar plots for goal/dropout frequency (third subplot).

    NOTE: this version is WITH ts to goal and is for BOTH the sequential estimator
    AND the iterative estimator.
    
    Three-panel comparison: interventions, ts to dropout, goal/dropout frequency.

    Version markings:
      • Baseline  (// hatch)
      • Oracle    (solid)
      • Seq Est   (.. hatch)
      • Iter Est  (xx hatch)

    NOTE: config should have keys 'p_fwd_chain_1', 'p_fwd_chain_2', 'p_fwd_stuck'
    and is for selecting the right parameter configuration from the df to select
    that row and plot its statistics.

    If save_subplots: will save each subplot separately.
    If use_suptitle: will make the suptitle the config['label']
    """  
    plt.rcParams.update({
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 16,
    })

    # ------------------------------------------------------------------
    # Config parsing
    # ------------------------------------------------------------------
    method_names  = config.get("method_names",  ["baseline", "inter", "seq", "iter"])
    method_labels = config.get("method_labels", {
        "baseline": "Baseline",
        "inter":    "Oracle",
        "seq":      "Seq Est",
        "iter":     "Iter Est"
    })
    method_hatch  = config.get("method_hatch",  {
        "baseline": "//",
        "inter":    "",
        "seq":      "..",
        "iter":     "xx"
    })

    var_colors_freq = config.get("var_colors_freq", {"goal": "#00AA66", "dropout": "#BB2222"})
    var_colors_ts   = config.get("var_colors_ts",   {"goal": "#77D9A8", "dropout": "#D97575"})
    var_colors_interventions = config.get("var_colors_interventions", "#9B1B9BB7")

    suffix_map  = {
        "baseline": "_baseline",
        "inter":    "_inter",
        "seq":      "_seq",
        "iter":     "_iter"
    }

    suffix_intv = {
        "inter": "",
        "seq":   "_seq",
        "iter":  "_iter"
    }

    fig = plt.figure(figsize=(13, 4.5))
    gs = GridSpec(1, 3, width_ratios=[0.5, 1, 1], wspace=0.3)
    axs = [fig.add_subplot(gs[i]) for i in range(3)]

    # Filter to correct row
    row = df[
        (df["p_fwd_stuck"] == config["p_fwd_stuck"]) &
        (df["p_fwd_chain_1"] == config["p_fwd_chain_1"]) &
        (df["p_fwd_chain_2"] == config["p_fwd_chain_2"])
    ]
    if row.empty:
        raise ValueError("No matching row found in DataFrame for the given config.")

    # ==============================================================  
    # 1. Interventions –– boxplots side-by-side  
    # ==============================================================  
    box_data, positions, pos_labels = [], [], []

    for i, method in enumerate(["inter", "seq", "iter"]):
        suf = suffix_intv[method]
        box_data.append({
            'med':    row[f"n_interventions{suf}_median"].values[0],
            'q1':     row[f"n_interventions{suf}_q1"].values[0],
            'q3':     row[f"n_interventions{suf}_q3"].values[0],
            'whislo': row[f"n_interventions{suf}_min"].values[0],
            'whishi': row[f"n_interventions{suf}_max"].values[0],
            'fliers': []})
        positions.append(i * 0.24)
        pos_labels.append(method_labels[method])

    ret = axs[0].bxp(box_data,
                     positions=positions,
                     widths=0.24,
                     patch_artist=True,
                     boxprops=dict(facecolor=var_colors_interventions, edgecolor='black'),
                     medianprops=dict(color='black', linewidth=2),
                     whiskerprops=dict(color='black'),
                     capprops=dict(color='black'))

    for idx, patch in enumerate(ret['boxes']):
        patch.set_hatch(method_hatch[["inter", "seq", "iter"][idx]])

    axs[0].set_xticks([])
    axs[0].set_xticklabels([])
    axs[0].tick_params(axis='x', bottom=False)

    box_w   = 0.24
    pad     = 0.08
    left    = positions[0] - box_w/2 - pad
    right   = positions[-1] + box_w/2 + pad

    axs[0].set_xlim(left, right)
    axs[0].margins(x=0)
    axs[0].set_ylabel("# Interventions")
    axs[0].set_title("# Interventions")

    # ==============================================================  
    # 2. Timesteps to Outcome –– boxplots  
    # ==============================================================  
    outcomes     = ["goal", "dropout"]
    spacing = 0.32
    offset       = 0
    box_data_ts  = []
    positions_ts = []
    xtick_pos    = []

    for outcome in outcomes:
        for j, method in enumerate(method_names):
            suf = suffix_map[method]
            prefix = f"n_ts_to_{outcome}{suf}"
            box_data_ts.append({
                'med':    row[f"{prefix}_median"].values[0],
                'q1':     row[f"{prefix}_q1"].values[0],
                'q3':     row[f"{prefix}_q3"].values[0],
                'whislo': row[f"{prefix}_min"].values[0],
                'whishi': row[f"{prefix}_max"].values[0],
                'fliers': []})
            positions_ts.append(offset + j * spacing)
        xtick_pos.append(offset + spacing)
        offset += spacing * len(method_names) + 0.2

    for idx, box in enumerate(box_data_ts):
        outcome_idx = idx // len(method_names)
        method_idx  = idx %  len(method_names)
        method      = method_names[method_idx]

        axs[1].bxp([box],
                   positions=[positions_ts[idx]],
                   widths=0.32,
                   patch_artist=True,
                   boxprops=dict(facecolor=list(var_colors_ts.values())[outcome_idx],
                                 edgecolor='black',
                                 hatch=method_hatch[method]),
                   medianprops=dict(color='black', linewidth=2),
                   whiskerprops=dict(color='black'),
                   capprops=dict(color='black'))

    axs[1].set_xticks(xtick_pos)
    axs[1].set_xticklabels(["Goal", "Dropout"])
    axs[1].set_ylabel("Timesteps")
    axs[1].set_title("Timesteps to Outcome")

    # ==============================================================  
    # 3. Outcome Frequencies –– bars  
    # ==============================================================  
    x         = np.arange(len(outcomes))
    bar_width = 0.24

    for i, method in enumerate(method_names):
        freqs = [row[f"freq_mk_goal_{method}"].values[0],
                 row[f"freq_dropout_{method}"].values[0]]
        axs[2].bar(x + (i - 1) * (bar_width * 0.9), freqs,
                   width=bar_width,
                   label=method_labels[method],
                   color=[var_colors_freq[o] for o in outcomes],
                   hatch=method_hatch[method],
                   edgecolor='black')

    axs[2].set_xticks(x)
    axs[2].set_xticklabels(["Goal", "Dropout"])
    axs[2].set_ylabel("Frequency")
    axs[2].set_title("Outcome Frequencies")

    # --------------------------------------------------------------  
    # Final formatting / saving  
    # --------------------------------------------------------------  
    legend_handles = [
        Patch(facecolor='white', edgecolor='black', hatch=method_hatch[m], label=method_labels[m])
        for m in method_names
    ]

    fig.legend(handles=legend_handles, loc='lower center', ncol=len(method_names), bbox_to_anchor=(0.5, -0.08))
    if use_suptitle:
        fig.suptitle(config["label"], fontsize=18)
    plt.tight_layout(rect=[0.01, 0.06, 0.99, 0.98])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')

        if save_subplots:
            subplot_dir = os.path.splitext(save_path)[0] + "_subplots"
            os.makedirs(subplot_dir, exist_ok=True)
            for i, ax in enumerate(axs):
                ext = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(os.path.join(subplot_dir, f"subplot_{i+1}.png"), bbox_inches=ext)

    plt.show()
