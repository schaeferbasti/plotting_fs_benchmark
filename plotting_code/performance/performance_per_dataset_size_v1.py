import ast
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

""" 
Description:
Performance per dataset size, tight point clusters
"""

FILE_NAME = "dummy_performance_results.csv"
PLOT_NAME = "performance_per_dataset_size_v1.png"

PLOT_TITLE = "Performance by Dataset Size (FS Methods)"
X_LABEL = "Dataset Size (# Features)"
Y_LABEL = "Mean Metric Error"


def plot(df):
    df["fs_method"] = df.get("featureselectionmethod", df.get("feature_selection_method", "unknown"))

    df['dataset_size'] = df.apply(
        lambda row: len(ast.literal_eval(row['original_feature_names']))
        if pd.notna(row['original_feature_names']) else np.nan, axis=1
    )

    bins = [0, 10, 50, 100, 500, np.inf]
    labels = ['<10F', '10-50F', '50-100F', '100-500F', '>500F']
    df['size_bin'] = pd.cut(df['dataset_size'], bins=bins, labels=labels, include_lowest=True)

    groups = df.dropna(subset=["fs_method", "size_bin", "metric_error"])

    pivot = groups.pivot_table(
        values="metric_error",
        index="size_bin",
        columns="fs_method",
        aggfunc="mean"
    ).fillna(np.nan)

    sizes = pivot.index
    methods = pivot.columns

    fig, ax = plt.subplots(figsize=(14, 8))

    cmap = plt.get_cmap("tab10", len(methods))
    colors = {m: cmap(i) for i, m in enumerate(methods)}

    x = np.arange(len(sizes))

    # TIGHT ALIGNMENT: minimal spread 0.05 width
    spread = 0.05
    for j, method in enumerate(methods):
        values = pivot[method].reindex(sizes).values

        # Tiny offsets: all points nearly aligned vertically
        offsets = (j - len(methods) / 2) * spread / len(methods)
        x_pos = x + offsets

        ax.scatter(x_pos, values, color=colors[method], label=method,
                   s=120, alpha=0.9, edgecolors='black', linewidth=1.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(sizes, rotation=0)
    ax.set_title(PLOT_TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.grid(True, alpha=0.3, axis="y")

    legend_elements = [Patch(facecolor=colors[m], edgecolor='black', label=m)
                       for m in methods]
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