from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

FILE_NAME = "results_per_split.csv"
PLOT_NAME = "scaled_performance_v1.png"
PLOT_TITLE = "Scaled Absolute Performance (100 = Best Method on Dataset)"
X_LABEL = "Feature Selection Method"
Y_LABEL = "Mean Scaled Performance Score (Higher is Better)"

METRIC_DIRECTIONS = {
    "log_loss": True,  # Lower error is better
    "rmse": True,  # Lower error is better
    "roc_auc": False,  # Higher performance is better
}


def calculate_scaled_performance(df):
    required_cols = ["metric", "feature_selection_method", "metric_error"]
    df_clean = df.dropna(subset=required_cols).copy()

    # 1. Adjust metric values so "Higher is ALWAYS Better" (Performance Score)
    def adjust_direction(row):
        is_lower_better = METRIC_DIRECTIONS.get(row["metric"], True)
        if is_lower_better:
            return -row["metric_error"]  # Invert error so higher is better
        return row["metric_error"]

    df_clean["performance"] = df_clean.apply(adjust_direction, axis=1)

    # 2. Average the performance per method, per dataset, per metric
    df_collapsed = df_clean.groupby(
        ["metric", "feature_selection_method"]
    )["performance"].mean().reset_index()

    # 3. Min-Max Scale the performance per dataset/metric
    # Best method gets 100, Worst method gets 0
    def min_max_scale(group):
        min_val = group.min()
        max_val = group.max()
        if max_val == min_val:
            return pd.Series(100.0, index=group.index)  # If all tied, give 100
        return ((group - min_val) / (max_val - min_val)) * 100.0

    df_collapsed["scaled_score"] = df_collapsed.groupby(["metric"])["performance"].transform(
        min_max_scale)

    # 4. Aggregate mean score and std per feature selection method
    agg_df = df_collapsed.groupby("feature_selection_method")["scaled_score"].agg(["mean", "std"]).reset_index()
    agg_df.columns = ["feature_selection_method", "mean_score", "std_score"]
    agg_df["std_score"] = agg_df["std_score"].fillna(0)

    return agg_df


def plot_absolute(df):
    agg_df = calculate_scaled_performance(df)

    # Sort by score (best first: higher score = better)
    agg_df = agg_df.sort_values("mean_score", ascending=False)

    methods = agg_df["feature_selection_method"].values
    scores = agg_df["mean_score"].values
    stds = agg_df["std_score"].values

    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(methods))

    bars = ax.bar(
        x,
        scores,
        yerr=stds,
        capsize=5,
        alpha=0.85,
        edgecolor="black",
        color="#4C72B0"  # Switched to a nice green color
    )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_title(PLOT_TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.grid(True, alpha=0.3, axis="y")

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
    plot_absolute(df)


if __name__ == "__main__":
    main()
