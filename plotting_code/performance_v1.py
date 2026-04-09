from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

""" 
Description:
This plot shows the 
"""

# TODO: Adapt file and plot name
FILE_NAME = "dummy_results.csv"
PLOT_NAME = "performance_v1.png"

# TODO: Adapt title and labels
PLOT_TITLE = "Performance"
X_LABEL = "Feature Selection Method"
Y_LABEL = "Metric Error"

# TODO: Adapt plotting function
def plot(df):
    methods = sorted(df["feature_selection_method"].dropna().unique())

    # When using this part, we get the bars for only the first, the last and the middle value in max_features
    groups = df.dropna(subset=["feature_selection_method", "time_train_s", "metric_error", "max_features"])
    all_max_feats = sorted(groups["max_features"].unique())
    n_steps = len(all_max_feats)
    step_indices = sorted(set([0, n_steps // 2, n_steps - 1]))
    selected_max_feats = [all_max_feats[idx] for idx in step_indices]
    groups = groups[groups["max_features"].isin(selected_max_feats)]

    # When using this part, we get the bars for all different steps in max_features
    """groups = df.dropna(subset=["feature_selection_method", "time_train_s", "metric_error", "max_features"])
    """

    pivot = groups.pivot_table(values='metric_error', index='feature_selection_method',
                               columns='max_features', aggfunc='mean').fillna(np.nan)

    fig, ax = plt.subplots(figsize=(16, 7))

    max_feats = sorted(pivot.columns)  # e.g. [5, 10, 20] or [50, 100, 200, 500]
    step_to_color = plt.get_cmap("Set3", len(max_feats))
    colors = {mf: step_to_color(i) for i, mf in enumerate(max_feats)}

    for i, method in enumerate(methods):
        method_cols = pivot.loc[method].dropna().index  # Available max_features values

        # Width inversely proportional to max_features (smaller #features = narrower bar)
        widths = np.linspace(1.5, 0.5, len(method_cols))

        means = pivot.loc[method].dropna().values
        bar_colors = [colors[mf] for mf in method_cols]

        bars = ax.bar(i, means, width=widths * 0.12, color=bar_colors, label=f'{method} ({len(method_cols)} budgets)' if i == 0 else "")

    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_title(PLOT_TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.grid(True, alpha=0.3, axis='y')
    max_feat_labels = ["Lowest #max_features", "Middle #max_features", "Highest #max_features"]
    max_feat_colors = [colors[mf] for mf in sorted(selected_max_feats)]
    legend_elements = [Patch(facecolor=c, label=l)
                       for c, l in zip(max_feat_colors, max_feat_labels)]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 0.75),
              loc='upper left', title="max_features")

    plt.tight_layout()
    out = OUTPUT_DIR / PLOT_NAME
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()

# Do nothing below
SCRIPT_DIR = Path(__file__).parent / "../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    plot(df)


if __name__ == "__main__":
    main()
