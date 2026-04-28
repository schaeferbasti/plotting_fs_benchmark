from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# TODO: Adapt file and plot name
FILE_NAME = "results_per_split.csv"
PLOT_NAME = "performance_v1.png"
# TODO: Adapt title and labels
PLOT_TITLE = "Performance (Global Rank)"
X_LABEL = "Feature Selection Method"
Y_LABEL = "Mean Rank (1 = Best)"

# Define which metrics mean "lower is better" vs "higher is better"
# Adapt this dictionary to match the exact 'metric_name' strings in your CSV
METRIC_DIRECTIONS = {
    "log_loss": True,  # Lower error is better
    "rmse": True,  # Lower error is better
    "roc_auc": False,  # Higher performance is better
}


def calculate_global_ranks(df):
    required_cols = ["metric", "feature_selection_method", "metric_error"]
    df_clean = df.dropna(subset=required_cols).copy()

    # 1. Adjust metric values so "Lower is ALWAYS Better"
    def adjust_direction(row):
        is_lower_better = METRIC_DIRECTIONS.get(row["metric"], True)
        if not is_lower_better:
            return -row["metric_error"]
        return row["metric_error"]

    df_clean["adjusted_metric"] = df_clean.apply(adjust_direction, axis=1)

    # 2. Average the adjusted metric per method, per dataset, per metric
    # This collapses cross-validation splits and models so we only have ONE row per method
    df_collapsed = df_clean.groupby(
        ["metric", "feature_selection_method"]
    )["adjusted_metric"].mean().reset_index()

    # 3. Calculate the Rank
    df_collapsed["rank"] = df_collapsed.groupby(["metric"])["adjusted_metric"].rank(
        method="min",
        ascending=True,
        na_option="bottom"
    )

    # 4. Aggregate mean rank and std of rank per feature selection method
    agg_df = df_collapsed.groupby("feature_selection_method")["rank"].agg(["mean", "std"]).reset_index()
    agg_df.columns = ["feature_selection_method", "mean_rank", "std_rank"]
    agg_df["std_rank"] = agg_df["std_rank"].fillna(0)

    return agg_df


# TODO: Adapt plotting function
def plot(df):
    # Get the globally ranked dataframe
    agg_df = calculate_global_ranks(df)

    # Sort by rank (best first: lower mean_rank = better)
    agg_df = agg_df.sort_values("mean_rank", ascending=True)

    methods = agg_df["feature_selection_method"].values
    ranks = agg_df["mean_rank"].values
    stds = agg_df["std_rank"].values

    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(len(methods))

    bars = ax.bar(
        x,
        ranks,
        yerr=stds,
        capsize=5,
        alpha=0.85,
        edgecolor="black",
        color="#4C72B0"
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


# Do nothing below
SCRIPT_DIR = Path(__file__).parent / "../../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/performance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_model_name(model_details):
    """Clean model name from dict string"""
    if pd.isna(model_details):
        return "Unknown"
    try:
        details = eval(model_details)  # Safe for your format
        return f"{details['model_cls']} ({details['model_type']})"
    except:
        return str(model_details)[:20] + "..."


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    plot(df)


if __name__ == "__main__":
    main()