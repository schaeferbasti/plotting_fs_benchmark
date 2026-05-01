import ast
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from utils.beautify import beautify_names, remove_jmi_random, add_model_name
from utils.scaling import min_max_scale, z_score_scale, median_max_scale

# TODO: Adapt file and plot name
FILE_NAME = "results_per_split.csv"
PLOT_NAME = "performance_per_budget_v3.png"

# TODO: Adapt title and labels
PLOT_TITLE = "Performance per Budget Stage (median-max scaling)"
X_LABEL = "Feature Selection Method"
Y_LABEL = "Mean Metric Error"


def plot(df):
    df = df.copy()

    df = beautify_names(df)
    df = remove_jmi_random(df)
    df = add_model_name(df)

    # Rank the budgets per dataset globally (1 = smallest max_features, 2 = next, etc.)
    df["budget_stage"] = df.groupby("tid")["max_features"].rank(method="dense").astype(int)

    # FIX: Strictly enforce a maximum of 5 budget stages.
    # This filters out any rogue data anomalies or extra splits that caused ranks 6-10.
    df = df[df["budget_stage"] <= 5]

    df = median_max_scale(df)

    # Pivot table based on the budget_stage (1, 2, 3, 4, 5)
    pivot = df.pivot_table(
        values='scaled_score',
        index='feature_selection_method',
        columns='budget_stage',
        aggfunc='mean'
    ).fillna(np.nan)

    methods = sorted(pivot.index)
    fig, ax = plt.subplots(figsize=(16, 7))

    # We now strictly generate exactly 5 colors for the 5 stages
    stages = [1, 2, 3, 4, 5]
    step_to_color = plt.get_cmap("Set3", len(stages))
    colors = {stage: step_to_color(i) for i, stage in enumerate(stages)}

    for i, method in enumerate(methods):
        method_cols = pivot.loc[method].dropna().index  # Available stages for this method (up to 5)

        # Widths: Scale from 0.8 (base width) down to 0.2 (top bar width)
        widths = np.linspace(0.8, 0.2, len(method_cols))

        means = pivot.loc[method].dropna().values
        bar_colors = [colors[stage] for stage in method_cols]

        bars = ax.bar([i] * len(method_cols), means, width=widths,
                      color=bar_colors, edgecolor="black", linewidth=0.5)

    # Formatting axes
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
    ax.set_title(PLOT_TITLE, fontsize=14, weight='bold')
    ax.set_xlabel(X_LABEL, fontsize=12)
    ax.set_ylabel(Y_LABEL, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    # Clean, hardcoded legend for exactly 5 budgets
    legend_labels = {
        1: "Budget 1 (Smallest)",
        2: "Budget 2",
        3: "Budget 3",
        4: "Budget 4",
        5: "Budget 5 (Largest)"
    }

    legend_elements = [
        Patch(facecolor=colors[stage], edgecolor="black", linewidth=0.5, label=legend_labels[stage])
        for stage in stages
    ]

    ax.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1),
              loc='upper left', title="Budget Stages")

    plt.tight_layout()
    out = OUTPUT_DIR / PLOT_NAME
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Budget Stage plot saved to {out}")


# Do nothing below
SCRIPT_DIR = Path(__file__).parent / "../../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/performance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    plot(df)


if __name__ == "__main__":
    main()