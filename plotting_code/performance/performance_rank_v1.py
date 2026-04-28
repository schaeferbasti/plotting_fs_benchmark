from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

# TODO: Adapt file and plot name
FILE_NAME = "results_per_split.csv"
PLOT_NAME = "performance_rank_v1.png"
# TODO: Adapt title and labels
PLOT_TITLE = "Performance Distribution (Global Rank)"
X_LABEL = "Feature Selection Method"
Y_LABEL = "Rank (1 = Best)"

# Define which metrics mean "lower is better" vs "higher is better"
METRIC_DIRECTIONS = {
    "log_loss": True,  # Lower error is better
    "rmse": True,  # Lower error is better
    "roc_auc": False,  # Higher performance is better
    "accuracy": False,
    "f1": False
}


def calculate_raw_ranks(df):
    required_cols = ["metric", "dataset", "feature_selection_method", "metric_error"]

    if "dataset" not in df.columns:
        required_cols.remove("dataset")
        group_cols = ["metric"]
    else:
        group_cols = ["metric", "dataset"]

    df_clean = df.dropna(subset=required_cols).copy()

    # 1. Adjust metric values so "Lower is ALWAYS Better"
    def adjust_direction(row):
        is_lower_better = METRIC_DIRECTIONS.get(row["metric"], True)
        if not is_lower_better:
            return -row["metric_error"]
        return row["metric_error"]

    df_clean["adjusted_metric"] = df_clean.apply(adjust_direction, axis=1)

    # 2. Average the adjusted metric per method, per dataset, per metric
    df_collapsed = df_clean.groupby(
        group_cols + ["feature_selection_method"]
    )["adjusted_metric"].mean().reset_index()

    # 3. Calculate the Rank across datasets
    df_collapsed["rank"] = df_collapsed.groupby(group_cols)["adjusted_metric"].rank(
        method="min",
        ascending=True,
        na_option="bottom"
    )

    return df_collapsed


def plot_boxplot(df):
    ranked_df = calculate_raw_ranks(df)

    # Sort the boxes from best (left) to worst (right) by MEAN rank
    mean_ranks = ranked_df.groupby("feature_selection_method")["rank"].mean().reset_index()
    mean_ranks = mean_ranks.sort_values("rank", ascending=False)
    sorted_methods = mean_ranks["feature_selection_method"].tolist()

    data_to_plot = []
    for method in sorted_methods:
        method_data = ranked_df[ranked_df["feature_selection_method"] == method]["rank"].values
        data_to_plot.append(method_data)

    fig, ax = plt.subplots(figsize=(16, 8))

    # ---- CUSTOMIZE BOXPLOT PROPS ----
    boxprops = dict(linewidth=1.5, color="black", facecolor="#4C72B0", alpha=0.7)

    # Hide the median line entirely
    medianprops = dict(linewidth=0, visible=False)

    # Make the mean look exactly like the traditional median line
    meanprops = dict(linestyle='-', linewidth=2, color='orange')

    # Create the boxplot
    bp = ax.boxplot(
        data_to_plot,
        patch_artist=True,
        showmeans=True,  # Enable showing the mean
        meanline=True,  # Force the mean to be a line, not a marker
        meanprops=meanprops,  # Apply the solid orange styling
        medianprops=medianprops,  # Apply the hidden median styling
        flierprops=dict(marker='o', color='black', alpha=0.3, markersize=5),
        widths=0.6
    )

    # Set colors for the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('#4C72B0')

    # Formatting
    ax.set_xticks(np.arange(1, len(sorted_methods) + 1))
    ax.set_xticklabels(sorted_methods, rotation=45, ha="right")
    ax.set_title(PLOT_TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)

    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.invert_yaxis()  # 1 is best, so put it at the top
    ax.grid(True, alpha=0.3, axis="y")

    # Add a custom legend to explicitly state the line is the mean
    ax.plot([], [], color='orange', linestyle='-', linewidth=2, label='Mean')
    ax.legend(loc="upper right")

    plt.tight_layout()
    out = OUTPUT_DIR / PLOT_NAME
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ Boxplot saved to {out}")


# Do nothing below
SCRIPT_DIR = Path(__file__).parent / "../../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/performance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    plot_boxplot(df)


if __name__ == "__main__":
    main()