"""
For plotting the policy class results on the 2D heatmaps.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def vis_vary_f1_f2_other_heatmap_1scale_pol(df: pd.DataFrame, other: str, decreasing: bool, 
                                        save_path: str, num_trajs: int, unique_other: list,
                                        title_var: str=None):
    """
    Creates sequence of heatmaps where x-axis is values of p_fwd_chain_1 and 
    y-axis is values of p_fwd_chain_2. The sequence varies another variable.

    Does this on a green to red spectrum, avging the number of times the human
    agent reaches goal versus drops out.

    Parameters:
    -----------
    df: pd.DataFrame
        Data containing 'p_fwd_chain_1', 'p_fwd_chain_2', 'freq_mk_goal', 'freq_dropout'.

    other: str
        Column variable to sequence the plots by.

    decreasing: bool
        If True, sort by decreasing order of the 'other' variable. Otherwise, increasing.

    save_path: str
        Path to save the plot (should end in '.png').

    num_trajs: int
        The number of trajectories the frequency of making the goal and dropping
        out was over.

    unique_other: list
        list of floats for the other variable to take on when determining what 
        values of it to make subplots for.  
        If None: automatically finds all its unique values.

    title_var: str or None
        what to call the other variable being varied for the title of the subplot
        indicating what value that variable is set to.
    """
    # Set font sizes
    plt.rcParams.update({
        "axes.labelsize": 16,      # Axis labels (x and y)
        "axes.titlesize": 16,      # Title font size
        "xtick.labelsize": 14,     # X-axis tick labels
        "ytick.labelsize": 15,     # Y-axis tick labels
        "legend.fontsize": 16,     # Legend
    })
    cbar_lbl_fontsize = 16 # Colorbar label

    if unique_other is None:
        unique_other = sorted(df[other].unique(), reverse=decreasing)
    num_plots = len(unique_other)

    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots + 2, 4), sharey=True)
    if num_plots == 1:
        axes = [axes]

    # Define a custom colormap: Red → White → Green
    colors = ["#8B0000", "#FFFFFF", "#006400"]  # Dark Red → White → Dark Green
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("CustomRdWhGn", colors)
    norm = mcolors.Normalize(vmin=-1, vmax=1)

    # Map unique values to indices for consistent plotting
    x_vals_sorted = sorted(df["p_fwd_chain_1"].unique())
    y_vals_sorted = sorted(df["p_fwd_chain_2"].unique())
    x_mapping = {val: i for i, val in enumerate(x_vals_sorted)}
    y_mapping = {val: i for i, val in enumerate(y_vals_sorted)}

    for ax, p_back_value in zip(axes, unique_other):
        df_subset = df[df[other] == p_back_value]
        color_matrix = np.ones((len(y_vals_sorted), len(x_vals_sorted), 4))  # RGBA (default white)

        for _, row in df_subset.iterrows():
            x_idx = x_mapping[row["p_fwd_chain_1"]]
            y_idx = y_mapping[row["p_fwd_chain_2"]]
            
            # Compute color value based on freq_mk_goal and freq_dropout
            color_value = (row["freq_mk_goal"] * num_trajs - row["freq_dropout"] * num_trajs) / num_trajs
            
            # Assign color from the custom diverging colormap
            color_matrix[y_idx, x_idx] = custom_cmap(norm(color_value))

        # Plot heatmap
        ax.imshow(color_matrix, aspect="auto", origin="lower",  
                  extent=[-0.5, len(x_vals_sorted)-0.5, -0.5, len(y_vals_sorted)-0.5])

        # Ensure square aspect ratio
        ax.set_box_aspect(1)

        # Set axis labels and ticks
        ax.set_xticks(range(len(x_vals_sorted)))
        ax.set_xticklabels([str(x) for x in x_vals_sorted], rotation=45)

        ax.set_yticks(range(len(y_vals_sorted)))
        ax.set_yticklabels([str(y) for y in y_vals_sorted])

        # Add grid lines
        ax.set_xticks(np.arange(-0.5, len(x_vals_sorted), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(y_vals_sorted), 1), minor=True)
        ax.grid(which="minor", color="black", linewidth=1.0)
        
        ax.set_xlabel("f_1: prob mv fwd chain 1")
        if num_plots > 1:
            if title_var is None:
                ax.set_title(f"{other} = {p_back_value}")
            else:
                ax.set_title(f"{title_var} = {p_back_value}")


        easy_only_pol_pts = df_subset[
            # Started and ended by working on easier chain (working on it forever)
            (   (df_subset["work_start_easier_chain"] == 1) &  
                (df_subset["work_end_easier_fwd"] == 1) &
                (df_subset["p_fwd_chain_1"] != df_subset["p_fwd_chain_2"])
            ) |
            # if 2 chains tied: see if work on the one started on forever
            (   (df_subset["p_fwd_chain_1"] == df_subset["p_fwd_chain_2"]) &
                (df_subset["work_start_chain_1"]) &
                (df_subset["work_end_chain_1"]) # see if work on chain 1 forever
            ) |
            (
                (df_subset["p_fwd_chain_1"] == df_subset["p_fwd_chain_2"]) &
                (df_subset["work_start_chain_1"] == 0) & # started by working on chain 2
                (df_subset["work_end_chain_2"]) # see if work on chain 2 forever
            )
        ]
            
        # Scatter plot of black dots (CGS points)
        if not easy_only_pol_pts.empty:
            ax.scatter(
                easy_only_pol_pts["p_fwd_chain_1"].map(x_mapping),
                easy_only_pol_pts["p_fwd_chain_2"].map(y_mapping),
                facecolors="none",  # Open circle
                edgecolors="black",  # Black border
                s=80, marker='o', label="Easy Only"
            )

        hard_prioritization = df_subset[
            # started by working on harder chain
            (   (df_subset["work_start_easier_chain"] == 0) & 
                (df_subset["p_fwd_chain_1"] != df_subset["p_fwd_chain_2"])
            ) |
            # if 2 chains tied: see if once reach end of one being worked on, 
            # work on the other no matter how stuck
            (   (df_subset["p_fwd_chain_1"] == df_subset["p_fwd_chain_2"]) & 
                (df_subset["work_start_chain_1"] == 1) & # started by working on chain 1
                (df_subset["work_end_chain_1"] == 0) & # switched to work on chain 2 at end of chain 1
                (df_subset["work_end_stuck_chain_1"] == 0) # no matter how stuck
            ) |
            (
                (df_subset["p_fwd_chain_1"] == df_subset["p_fwd_chain_2"]) & 
                (df_subset["work_start_chain_1"] == 0) & # started by working on chain 2
                (df_subset["work_end_chain_2"] == 0) & # switched to work on chain 1 at end of chain 2
                (df_subset["work_end_stuck_chain_2"] == 0) # no matter how stuck
            )
        ]
        hard_idx = hard_prioritization.index

        # Scatter plot of black dots (CGS points)
        if not hard_prioritization.empty:
            ax.scatter(
                hard_prioritization["p_fwd_chain_1"].map(x_mapping),
                hard_prioritization["p_fwd_chain_2"].map(y_mapping),
                color="black", s=100, marker='+', label="Hard Priorit."
            )

        easy_prioritization = df_subset[
            # started by working on easier chain then switched to work on
            # the harder chain (when not at all stuck)
            (   (df_subset["work_start_easier_chain"] == 1) &  
                (df_subset["work_end_easier_fwd"] == 0) & 
                (df_subset["p_fwd_chain_1"] != df_subset["p_fwd_chain_2"])
            ) | 
            # 2 chains tied: see if worked on one and then switched to the other only
            # when not stuck
            (   (df_subset["p_fwd_chain_1"] == df_subset["p_fwd_chain_2"]) & 
                (df_subset["work_start_chain_1"] == 1) & # started by working on chain 1
                (df_subset["work_end_chain_1"] == 0) & # switched to work on chain 2 at end of chain 1
                (df_subset["work_end_stuck_chain_1"] == 1) # but don't when stuck
            ) |
            (
                (df_subset["p_fwd_chain_1"] == df_subset["p_fwd_chain_2"]) & 
                (df_subset["work_start_chain_1"] == 0) & # started by working on chain 2
                (df_subset["work_end_chain_2"] == 0) & # switched to work on chain 1 at end of chain 2
                (df_subset["work_end_stuck_chain_2"] == 1) # but don't when stuck
            )
        ]
        # Exclude any points already categorized as hard (equivalent cases here by symmetry
        # of when the 2 progress chains have equal p fwds, only happens in this case)
        easy_prioritization = easy_prioritization.loc[~easy_prioritization.index.isin(hard_idx)]

        # Scatter plot of black dots (CGS points)
        if not easy_prioritization.empty:
            ax.scatter(
                easy_prioritization["p_fwd_chain_1"].map(x_mapping),
                easy_prioritization["p_fwd_chain_2"].map(y_mapping),
                color="black", s=50, marker='s', label="Easy Priorit."
            )

    axes[0].set_ylabel("f_2: prob mv fwd chain 2")

    # Adjust layout
    fig.subplots_adjust(right=0.85)

    legend_labels = {
    }
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=label) 
               for color, label in legend_labels.items()]
    
    # Add marker for each policy class
    handles.append(plt.Line2D([0], [0], marker='o',  markerfacecolor='none', markeredgecolor='black', linestyle='', markersize=8, label="Easy Only"))
    handles.append(plt.Line2D([0], [0], marker='s', color='black', linestyle='', markersize=8, label="Easy Priorit"))
    handles.append(plt.Line2D([0], [0], marker='+', color='black', linestyle='', markersize=8, label="Hard Priorit"))

    # Add colorbar with the custom colormap
    if num_plots > 1:
        cbar_ax = fig.add_axes([0.86, 0.15, 0.02, 0.7])

    else:
        # for mini
        cbar_ax = fig.add_axes([0.77, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)

    # Set vertical label
    cbar.set_label("Freq dropout vs mk goal", fontsize = cbar_lbl_fontsize)

    # Add "Always mk goal" at top and "Always dropout" at bottom
    if num_plots > 1:
        # for 5 plots
        cbar_ax.text(2.2, 1.05, "Always mk goal", ha="left", va="center", fontsize=cbar_lbl_fontsize, transform=cbar_ax.transAxes)
        cbar_ax.text(2.2, -0.1, "Always dropout", ha="left", va="center", fontsize=cbar_lbl_fontsize, transform=cbar_ax.transAxes)
    else:
        #  for 1 plot
        cbar_ax.text(5.5, 1.05, "Always make goal", ha="left", va="center", fontsize=cbar_lbl_fontsize, transform=cbar_ax.transAxes)
        cbar_ax.text(5.5, -0.1, "Always dropout", ha="left", va="center", fontsize=cbar_lbl_fontsize, transform=cbar_ax.transAxes)

    # Move legend to top-right
    if num_plots > 1:
        # for 5 plots
        # fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(1, 1.05))
        # NOTE: just describe in text
        pass
    else:
        # for mini
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.3, 0.8))

    # Save and show
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()