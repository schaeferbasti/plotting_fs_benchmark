from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

""" 
Description:
Performance per dataset ID, grouped by FS family
(one bar for feature ranking methods and one bar for subset selection methods)
"""

FILE_NAME = "dummy_performance_results.csv"
PLOT_NAME = "performance_per_scoring_type_v2.png"

PLOT_TITLE = "Performance by Dataset ID and FS Family"
X_LABEL = "Mean Metric Error"
Y_LABEL = "Dataset ID"


def plot(df):
    # Dataset identifier column
    df["dataset_id_plot"] = df["tid"]

    # Map boolean column to readable family names
    df["fs_family"] = df["feature_selection_is_scoring_method"].map({
        True: "Feature ranking",
        False: "Subset selection"
    }).dropna()  # Remove Unknown rows entirely

    # Keep only valid rows
    groups = df.dropna(subset=["dataset_id_plot", "metric_error", "fs_family"])

    # Aggregate mean performance per dataset and family
    pivot = groups.pivot_table(
        values="metric_error",
        index="dataset_id_plot",
        columns="fs_family",
        aggfunc="mean"
    ).fillna(np.nan)

    # Sort datasets by average performance
    pivot["__mean__"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("__mean__", ascending=True).drop(columns="__mean__")

    dataset_ids = pivot.index
    families = pivot.columns

    fig, ax = plt.subplots(figsize=(max(12, len(dataset_ids) * 0.6), 8))

    colors = {
        "Feature ranking": "#4C72B0",
        "Subset selection": "#DD8452",
    }

    x = np.arange(len(dataset_ids))  # Now x positions for dataset IDs
    width = 0.35

    for j, family in enumerate(families):
        values = pivot[family].reindex(dataset_ids).values
        ax.bar(
            x + (j - (len(families) - 1) / 2) * width,
            values,
            width=width,
            color=colors.get(family, "#999999"),
            label=family,
            alpha=0.9,
            edgecolor="black"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_ids, rotation=45, ha="right")
    ax.set_title(PLOT_TITLE)
    ax.set_xlabel(Y_LABEL)  # Dataset IDs now on x-axis
    ax.set_ylabel(X_LABEL)  # Metric error on y-axis
    ax.grid(True, alpha=0.3, axis="y")

    legend_elements = [
        Patch(facecolor=colors.get(f, "#999999"), edgecolor="black", label=f)
        for f in families
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    out = OUTPUT_DIR / PLOT_NAME
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


SCRIPT_DIR = Path(__file__).parent / "../../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/performance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    plot(df)


if __name__ == "__main__":
    main()