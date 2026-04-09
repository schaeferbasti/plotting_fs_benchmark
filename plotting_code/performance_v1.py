from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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
    data = [
        df.loc[df["feature_selection_method"] == method, "metric_error"].dropna().values
        for method in methods
    ]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Calculate mean and std for error bars
    means = [np.mean(d) for d in data]
    stds = [np.std(d) for d in data]

    # Bar chart with error bars
    bars = ax.bar(range(len(methods)), means, yerr=stds, capsize=5,
                  color='steelblue', alpha=0.8, edgecolor='navy', linewidth=1.2)

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        # ax.text(bar.get_x() + bar.get_width() / 2., height + max(stds) * 0.01, f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_title(PLOT_TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.grid(True, alpha=0.3, axis='y')

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
