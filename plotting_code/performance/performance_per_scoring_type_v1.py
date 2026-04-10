import ast
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

""" 
Description:
Performance per dataset size, grouped by FS family
(one bar for scoring methods, one bar for subset methods)
"""

FILE_NAME = "dummy_performance_results.csv"
PLOT_NAME = "performance_per_scoring_type_v1.png"

PLOT_TITLE = "Performance by Dataset Size and FS Family"
X_LABEL = "Dataset Size (# Features)"
Y_LABEL = "Mean Metric Error"


def plot(df):
    # Dataset size from original_feature_names
    df["dataset_size"] = df.apply(
        lambda row: len(ast.literal_eval(row["original_feature_names"]))
        if pd.notna(row["original_feature_names"]) else np.nan,
        axis=1
    )

    # Bin sizes
    bins = [0, 10, 50, 100, 500, np.inf]
    labels = ["<10F", "10-50F", "50-100F", "100-500F", ">500F"]
    df["size_bin"] = pd.cut(df["dataset_size"], bins=bins, labels=labels, include_lowest=True)

    # Map boolean column to readable family names
    df["fs_family"] = df["feature_selection_is_scoring_method"].map({
        True: "Feature ranking",
        False: "Subset selection"
    }).fillna("Unknown")

    # Filter valid rows
    groups = df.dropna(subset=["size_bin", "metric_error", "fs_family"])

    # Aggregate mean error per dataset size and FS family
    pivot = groups.pivot_table(
        values="metric_error",
        index="size_bin",
        columns="fs_family",
        aggfunc="mean"
    ).fillna(np.nan)

    sizes = pivot.index
    families = pivot.columns

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        "Feature ranking": "#4C72B0",
        "Subset selection": "#DD8452",
        "Unknown": "#999999"
    }

    x = np.arange(len(sizes))
    width = 0.35

    for j, family in enumerate(families):
        values = pivot[family].reindex(sizes).values
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
    ax.set_xticklabels(sizes, rotation=0)
    ax.set_title(PLOT_TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
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