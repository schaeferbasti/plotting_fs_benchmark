from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# TODO: Adapt file and plot name
FILE_NAME = "results_per_split.csv"
PLOT_NAME = "pareto_performance_time_v1.png"

# TODO: Adapt title and labels
PLOT_TITLE = "Pareto Front: Performance vs. Training Time"
X_LABEL = "Mean Training Time (s)"
Y_LABEL = "Mean Scaled Performance Score (Higher is Better)"

METRIC_DIRECTIONS = {
    "log_loss": True,  # Lower error is better
    "rmse": True,  # Lower error is better
    "roc_auc": False,  # Higher performance is better
    "accuracy": False,  # Higher performance is better
    "f1": False  # Higher performance is better
}


def calculate_scaled_performance_and_time(df):
    # Ensure dataset, metric, method, error, and time columns are present
    required_cols = ["metric", "feature_selection_method", "metric_error", "feature_selection_fit_time"]

    df_clean = df.dropna(subset=required_cols).copy()

    # 1. Adjust metric values so "Higher is ALWAYS Better"
    def adjust_direction(row):
        is_lower_better = METRIC_DIRECTIONS.get(row["metric"], True)
        if is_lower_better:
            return -row["metric_error"]  # Invert error so higher is better
        return row["metric_error"]

    df_clean["performance"] = df_clean.apply(adjust_direction, axis=1)

    # 2. Average the performance and time per method, per dataset, per metric
    # This collapses CV splits/models
    df_collapsed = df_clean.groupby(
        ["metric", "feature_selection_method"]
    ).agg(
        performance=("performance", "mean"),
        feature_selection_fit_time=("feature_selection_fit_time", "mean")
    ).reset_index()

    # 3. Min-Max Scale the performance per dataset/metric
    def min_max_scale(group):
        min_val = group.min()
        max_val = group.max()
        if max_val == min_val:
            return pd.Series(100.0, index=group.index)
        return ((group - min_val) / (max_val - min_val)) * 100.0

    df_collapsed["scaled_score"] = df_collapsed.groupby(["metric"])["performance"].transform(
        min_max_scale)

    # 4. Aggregate mean score and mean time per feature selection method across all datasets
    agg_df = df_collapsed.groupby("feature_selection_method").agg(
        mean_score=("scaled_score", "mean"),
        mean_time=("feature_selection_fit_time", "mean")
    ).reset_index()

    return agg_df


def plot(df):
    agg_df = calculate_scaled_performance_and_time(df)

    # Compute Pareto front: MAXIMIZE score and MINIMIZE time
    pareto_idx = []
    for i, row_i in agg_df.iterrows():
        dominated = False
        for j, row_j in agg_df.iterrows():
            if i == j:
                continue

            # row_j dominates row_i if it is better or equal on BOTH objectives,
            # and strictly better on AT LEAST ONE.
            better_or_equal_score = row_j["mean_score"] >= row_i["mean_score"]
            better_or_equal_time = row_j["mean_time"] <= row_i["mean_time"]

            strictly_better_score = row_j["mean_score"] > row_i["mean_score"]
            strictly_better_time = row_j["mean_time"] < row_i["mean_time"]

            if (better_or_equal_score and better_or_equal_time) and (strictly_better_score or strictly_better_time):
                dominated = True
                break

        if not dominated:
            pareto_idx.append(i)

    # Extract Pareto points and sort by time to draw a clean line
    pareto = agg_df.loc[pareto_idx].sort_values("mean_time")

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot all methods (Non-Pareto + Pareto)
    ax.scatter(
        agg_df["mean_time"],
        agg_df["mean_score"],
        s=60,
        color="lightgray",
        edgecolor="gray",
        alpha=0.8,
        label="Methods",
    )

    # Highlight Pareto front
    ax.scatter(
        pareto["mean_time"],
        pareto["mean_score"],
        s=80,
        color="tab:red",
        label="Pareto front",
        zorder=3,
    )

    # Draw line connecting Pareto points
    ax.plot(
        pareto["mean_time"],
        pareto["mean_score"],
        color="tab:red",
        linewidth=2,
        zorder=2,
    )

    # Labels on Pareto points
    for _, row in pareto.iterrows():
        ax.annotate(
            row["feature_selection_method"],
            (row["mean_time"], row["mean_score"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
            weight='bold'
        )

    ax.set_title(PLOT_TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)

    # Optional: If time varies wildly (e.g. 0.1s vs 1000s), uncomment the line below:
    # ax.set_xscale("log")

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = OUTPUT_DIR / PLOT_NAME
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Pareto plot saved to {out}")


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