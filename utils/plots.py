import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

from matplotlib.patches import Rectangle, Patch
from scipy.ndimage import gaussian_filter1d


def lasagna_plot(
    df,
    values,
    plot_title,
    x_label,
    y_label,
    output_path,
    smoothing=20,
    binary_mode="topk",
    binary_tol=0.03,
    binary_topk=3,
    overlay=True,
    mode="validity",
    sort_methods="mean_value"
):
    """
    Save a lasagna plot from a long dataframe and optionally add binary plots showing
    which methods are within `binary_tol` of the best method or in the top-k at each EPV.
    If `overlay=True`, the selected binary information is drawn as white outlines on
    top of the original lasagna plot instead of being shown as separate plots.

    Parameters
    ----------
    df : pd.DataFrame
        Long dataframe with selector/method, epv, and value columns.
    values : str
        The metric of interest.
    plot_title : str
        The title for the plot.
    x_label : str
        The x-axis label.
    y_label : str
        The y-axis label.
    output_path : str
        The path for saving the results.
    smoothing : int
        The sigma parameter for the Gaussian kernel to smooth the base plot.
    binary_mode : str
        The mode to use for binarization. Choices: "threshold", "topk", or None.
    binary_tol : float
        A method is marked 1 if it is within this tolerance of the best value at that EPV.
    binary_topk : int
        A method is marked 1 if it is among the top-k methods at that EPV.
    overlay : bool
        If True, draw white outlines on the original lasagna plot instead of creating
        separate binary plots.
    sort_methods: str
        If "alphabetical", sort methods A-Z. If "mean_value", sort by mean value across EPVs.
    """

    # --- DATA CLEANING ---
    # Work on a copy to prevent modifying the original dataframe outside the function
    df = df.copy()

    if "selector" in df.columns:
        # 1. Strip "FeatureSelector" from all method names
        df["selector"] = df["selector"].str.replace("FeatureSelector", "", regex=False)

        # 2. Rename specific methods to their acronyms/new names
        df["selector"] = df["selector"].replace({
            "Accuracy": "LOCO",
            "SequentialBackwardElimination": "RFE",
            "SequentialForwardSelection": "SFS"
        })

    if binary_mode not in {None, "threshold", "topk"}:
        raise ValueError("binary_mode must be one of: None, 'threshold', 'topk'")

    lasagna_df = (
        df.pivot(index="selector", columns="epv", values=values)
        .reindex(sorted(df["epv"].unique()), axis=1)
    )

    if sort_methods == "alphabetical":
        # Sorts A to Z. Because of ax.invert_yaxis() later, 'A' goes to the top.
        lasagna_df = lasagna_df.sort_index(ascending=True)

    elif sort_methods == "mean_value":
        # Calculates the mean stability for each method across all EPVs.
        # Sorts descending, so the highest mean is placed at the top.
        row_means = lasagna_df.mean(axis=1)
        lasagna_df = lasagna_df.loc[row_means.sort_values(ascending=False).index]

    epvs = lasagna_df.columns.to_numpy()

    # plot_epv_distribution(epvs)

    epv_edges = _compute_bin_edges(epvs)
    epvs_dense_edges = np.linspace(epv_edges[0], epv_edges[-1], 1001)
    epvs_dense = 0.5 * (epvs_dense_edges[:-1] + epvs_dense_edges[1:])
    y_edges = np.arange(len(lasagna_df.index) + 1) - 0.5

    # interpolation
    Z = np.vstack([
        np.interp(epvs_dense, epvs[mask], values_per_selector[mask])
        for _, values_per_selector in lasagna_df.iterrows()
        for mask in [values_per_selector.notna()]
    ])

    # smoothing
    Z = gaussian_filter1d(Z, sigma=smoothing, axis=1, mode="nearest")

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.pcolormesh(
        epvs_dense_edges,
        y_edges,
        Z,
        cmap="Blues",
        vmin=0,
        vmax=1,
        shading="flat"
    )

    ax.set_title(plot_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    min_distance = (epvs.max() - epvs.min()) / 8.0  # Allow roughly 8 ticks max across the axis

    tick_positions = []
    tick_labels = []

    last_pos = -float('inf')

    # Iterate through all available epvs
    for val in epvs:
        if (val - last_pos) >= min_distance:
            tick_positions.append(val)
            tick_labels.append(f"{val:.1f}")  # 1 decimal place (e.g., "2799.7")
            last_pos = val

    # Force the very last EPV to be included so the right edge has a label
    if epvs[-1] not in tick_positions:
        # If adding the last one makes it overlap the second-to-last, replace the second-to-last
        if len(tick_positions) > 0 and (epvs[-1] - tick_positions[-1]) < min_distance * 0.5:
            tick_positions.pop()
            tick_labels.pop()
        tick_positions.append(epvs[-1])
        tick_labels.append(f"{epvs[-1]:.1f}")

    # Apply them to the axis
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")

    ax.set_yticks(np.arange(len(lasagna_df.index)))
    ax.set_yticklabels(lasagna_df.index)
    ax.invert_yaxis()
    ax.invert_xaxis()

    if overlay:
        if binary_mode == "threshold":
            _add_discrete_overlay(
                ax,
                _compute_binary_threshold(lasagna_df, binary_tol),
                epv_edges
            )
        elif binary_mode == "topk":
            _add_discrete_overlay(
                ax,
                _compute_binary_topk(lasagna_df, binary_topk),
                epv_edges
            )

    cmap = cm.get_cmap("Blues")
    color_low = cmap(0.1)
    color_high = cmap(0.9)

    # Create the patch elements for the legend
    legend_elements = [
        Patch(facecolor=color_high, edgecolor='black', label='High Stability'),
        Patch(facecolor=color_low, edgecolor='black', label='Low Stability')
    ]

    # Add the legend
    if mode == "stability":
        """cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, shrink=0.6)
        cbar.set_ticks([])
        cbar.ax.set_title("stable", pad=10, fontsize=10)
        cbar.ax.text(
            0.5, -0.05,
            "unstable",
            transform=cbar.ax.transAxes,
            ha="center",
            va="top",
            fontsize=10
        )"""
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.ax.set_title("stable", pad=12, fontsize=11)
        cbar.set_ticks([0.0, 1.0])
        cbar.ax.tick_params(length=0, labelsize=10)
        cbar.ax.text(
            0.5, -0.05,
            "unstable",
            transform=cbar.ax.transAxes,
            ha="center",
            va="top",
            fontsize=12
        )
    else:
        """cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, shrink=0.6, aspect=30)
        cbar.ax.set_title("valid", pad=10, fontsize=10)
        cbar.set_ticks([])
        cbar.ax.text(
            0.5, -0.05,
            "unvalid",
            transform=cbar.ax.transAxes,
            ha="center",
            va="top",
            fontsize=10
        )"""
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, shrink=0.6)
        cbar.ax.set_title("valid", pad=12, fontsize=11)
        cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar.ax.tick_params(length=0, labelsize=10)
        cbar.ax.text(
            0.5, -0.05,
            "unvalid",
            transform=cbar.ax.transAxes,
            ha="center",
            va="top",
            fontsize=12
        )

    suffix = _build_suffix(
        overlay=overlay,
        binary_mode=binary_mode if overlay else None,
        binary_tol=binary_tol,
        binary_topk=binary_topk
    )

    plt.tight_layout()
    plt.savefig(f"{output_path}_{suffix}.svg", dpi=300, bbox_inches="tight")
    plt.close()

    if binary_mode == "threshold" and not overlay:
        binary_df = _compute_binary_threshold(lasagna_df, binary_tol)
        binary_Z = binary_df.to_numpy()

        fig, ax = plt.subplots(figsize=(12, 6))

        im = ax.pcolormesh(
            epv_edges,
            y_edges,
            binary_Z,
            cmap="Blues",
            vmin=0,
            vmax=1,
            shading="flat"
        )

        ax.set_title(plot_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        ax.set_yticks(np.arange(len(binary_df.index)))
        ax.set_yticklabels(binary_df.index)
        min_distance = (epvs.max() - epvs.min()) / 8.0  # Allow roughly 8 ticks max across the axis

        tick_positions = []
        tick_labels = []

        last_pos = -float('inf')

        # Iterate through all available epvs
        for val in epvs:
            if (val - last_pos) >= min_distance:
                tick_positions.append(val)
                tick_labels.append(f"{val:.1f}")  # 1 decimal place (e.g., "2799.7")
                last_pos = val

        # Force the very last EPV to be included so the right edge has a label
        if epvs[-1] not in tick_positions:
            # If adding the last one makes it overlap the second-to-last, replace the second-to-last
            if len(tick_positions) > 0 and (epvs[-1] - tick_positions[-1]) < min_distance * 0.5:
                tick_positions.pop()
                tick_labels.pop()
            tick_positions.append(epvs[-1])
            tick_labels.append(f"{epvs[-1]:.1f}")

        # Apply them to the axis
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=0, ha="center")
        ax.invert_yaxis()
        ax.invert_xaxis()

        binary_legend_elements = [
            Patch(facecolor=cmap(1.0), edgecolor='black', label='Within Threshold'),
            Patch(facecolor=cmap(0.0), edgecolor='black', label='Outside Threshold')
        ]
        if mode == "validity":
            ax.legend(handles=binary_legend_elements, loc='upper right', bbox_to_anchor=(1.0, 0.11))
        else:
            ax.legend(handles=binary_legend_elements, loc='upper right', bbox_to_anchor=(0.15, 0.0))


        suffix = _build_suffix(
            overlay=False,
            binary_mode="threshold",
            binary_tol=binary_tol,
            binary_topk=binary_topk
        )

        plt.tight_layout()
        plt.savefig(f"{output_path}_{suffix}.svg", dpi=300, bbox_inches="tight")
        plt.close()

    if binary_mode == "topk" and not overlay:
        binary_df = _compute_binary_topk(lasagna_df, binary_topk)
        binary_Z = binary_df.to_numpy()

        fig, ax = plt.subplots(figsize=(12, 6))

        im = ax.pcolormesh(
            epv_edges,
            y_edges,
            binary_Z,
            cmap="Blues",
            vmin=0,
            vmax=1,
            shading="flat"
        )

        ax.set_title(plot_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        ax.set_yticks(np.arange(len(binary_df.index)))
        ax.set_yticklabels(binary_df.index)
        min_distance = (epvs.max() - epvs.min()) / 8.0  # Allow roughly 8 ticks max across the axis

        tick_positions = []
        tick_labels = []

        last_pos = -float('inf')

        # Iterate through all available epvs
        for val in epvs:
            if (val - last_pos) >= min_distance:
                tick_positions.append(val)
                tick_labels.append(f"{val:.1f}")  # 1 decimal place (e.g., "2799.7")
                last_pos = val

        # Force the very last EPV to be included so the right edge has a label
        if epvs[-1] not in tick_positions:
            # If adding the last one makes it overlap the second-to-last, replace the second-to-last
            if len(tick_positions) > 0 and (epvs[-1] - tick_positions[-1]) < min_distance * 0.5:
                tick_positions.pop()
                tick_labels.pop()
            tick_positions.append(epvs[-1])
            tick_labels.append(f"{epvs[-1]:.1f}")

        # Apply them to the axis
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=0, ha="center")
        ax.invert_yaxis()
        ax.invert_xaxis()

        binary_legend_elements = [
            Patch(facecolor=cmap(1.0), edgecolor='black', label='Within Top-3'),
            Patch(facecolor=cmap(0.0), edgecolor='black', label='Outside Top-3')
        ]
        if mode =="validity":
            ax.legend(handles=binary_legend_elements, loc='upper right', bbox_to_anchor=(0.15, 0.11))
        else:
            ax.legend(handles=binary_legend_elements, loc='upper right', bbox_to_anchor=(0.15, 0.0))

        suffix = _build_suffix(
            overlay=False,
            binary_mode="topk",
            binary_tol=binary_tol,
            binary_topk=binary_topk
        )

        plt.tight_layout()
        plt.savefig(f"{output_path}_{suffix}.svg", dpi=300, bbox_inches="tight")
        plt.close()


def _build_suffix(overlay, binary_mode, binary_tol, binary_topk):
    if binary_mode is None:
        return "plain"

    parts = ["overlay" if overlay else "binary"]

    if binary_mode == "threshold":
        parts.append(f"thr_{binary_tol:.2f}".replace(".", ""))
    elif binary_mode == "topk":
        parts.append(f"top_{binary_topk}")

    return "_".join(parts)


def _compute_binary_threshold(df, binary_tol):
    best_per_epv = df.max(axis=0)
    binary_df = df.ge(best_per_epv - binary_tol, axis=1).astype(int)
    return binary_df.reindex(df.index)


def _compute_binary_topk(df, binary_topk):
    binary_df = pd.DataFrame(0, index=df.index, columns=df.columns)
    for epv in df.columns:
        col = df[epv].dropna()
        top_idx = col.nlargest(min(binary_topk, len(col))).index
        binary_df.loc[top_idx, epv] = 1
    return binary_df.reindex(df.index)


def _compute_bin_edges(x):
    x = np.asarray(x, dtype=float)

    if len(x) == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5], dtype=float)

    edges = np.empty(len(x) + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    return edges


def _add_discrete_overlay(ax, df, x_edges, linewidth=1, linestyle="-"):
    for row_idx, (_, row) in enumerate(df.iterrows()):
        vals = row.to_numpy(dtype=int)

        start = None
        for col_idx, v in enumerate(vals):
            if v == 1 and start is None:
                start = col_idx

            if (v == 0 or col_idx == len(vals) - 1) and start is not None:
                end = col_idx if v == 0 else col_idx + 1

                x0 = x_edges[start]
                x1 = x_edges[end]

                rect = Rectangle(
                    (x0, row_idx - 0.5),
                    x1 - x0,
                    1.0,
                    fill=False,
                    edgecolor="white",
                    linewidth=linewidth,
                    linestyle=linestyle
                )
                ax.add_patch(rect)
                start = None


def plot_epv_distribution(epvs):
    fig, ax = plt.subplots(figsize=(12, 8))  # Short and wide figure

    # Create an array of zeros the same length as epvs for the y-axis
    y_zeros = np.zeros_like(epvs)

    # Plot the 1D scatter
    ax.scatter(epvs, y_zeros, color="blue", alpha=0.5, s=50, edgecolors="black")

    # Clean up the plot
    ax.set_yticks([])  # Hide the y-axis ticks since they are all 0
    ax.set_xlabel("EPV (Selection Difficulty)")
    ax.set_title("Distribution of EPV Values")

    # Hide the top, left, and right spines so it looks like a single axis line
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    plt.show()
