import ast
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

# TODO: Adapt file and plot name
FILE_NAME = "results_per_split.csv"
PLOT_NAME = "performance_per_budget_v1.png"

# TODO: Adapt title and labels
PLOT_TITLE = "Performance per Budget Stage"
X_LABEL = "Feature Selection Method"
Y_LABEL = "Mean Metric Error"


def plot(df):
    df = df.copy()

    # --- DATA CLEANING (Consistency with previous plots) ---
    if "feature_selection_method" in df.columns:
        df["feature_selection_method"] = df["feature_selection_method"].str.replace("FeatureSelector", "", regex=False)

        df["feature_selection_method"] = df["feature_selection_method"].replace({
            "Accuracy": "LOCO",
            "SequentialBackwardElimination": "SBE",
            "SequentialForwardSelection": "SFS"
        })

        df = df[~df["feature_selection_method"].isin(["JMI", "Random"])]
    # -------------------------------------------------------

    # Filter to only rows with the necessary data
    groups = df.dropna(subset=["feature_selection_method", "metric_error", "max_features", "tid"])

    # Rank the budgets per dataset globally (1 = smallest max_features, 2 = next, etc.)
    groups["budget_stage"] = groups.groupby("tid")["max_features"].rank(method="dense").astype(int)

    # FIX: Strictly enforce a maximum of 5 budget stages.
    # This filters out any rogue data anomalies or extra splits that caused ranks 6-10.
    groups = groups[groups["budget_stage"] <= 5]

    # Pivot table based on the budget_stage (1, 2, 3, 4, 5)
    pivot = groups.pivot_table(
        values='metric_error',
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