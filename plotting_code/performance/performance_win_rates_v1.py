from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

# TODO: Adapt file and plot name
FILE_NAME = "results_per_split.csv"
PLOT_NAME = "performance_win_rates_v1.png"
PLOT_TITLE = "Method Win Rates"
X_LABEL = "Feature Selection Method"
Y_LABEL = "Win Rate (%)"

METRIC_DIRECTIONS = {
    "log_loss": True,  # Lower error is better
    "rmse": True,  # Lower error is better
    "roc_auc": False,  # Higher performance is better
}


def calculate_win_rates(df):
    """
    Calculate win rates: proportion of datasets where each method has the best performance.
    """
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

    # 2. Average CV splits per dataset/method
    df_collapsed = df_clean.groupby(
        group_cols + ["feature_selection_method"]
    )["adjusted_metric"].mean().reset_index()

    # 3. Identify the winner in each dataset (lowest adjusted_metric)
    df_collapsed["is_winner"] = df_collapsed.groupby(group_cols)["adjusted_metric"].transform("min") == df_collapsed[
        "adjusted_metric"]

    # 4. Calculate win rates per method
    win_rates = df_collapsed.groupby("feature_selection_method")["is_winner"].agg([
        "sum", "count", "mean"
    ]).reset_index()
    win_rates.columns = ["feature_selection_method", "wins", "total_datasets", "win_rate"]
    win_rates["win_rate_pct"] = win_rates["win_rate"] * 100

    # Also calculate 95% CI for binomial proportion (Wilson score interval)
    n = win_rates["total_datasets"]
    p = win_rates["win_rate"]
    z = 1.96  # 95% CI
    center = (p + z ** 2 / (2 * n)) / (1 + z ** 2 / n)
    margin = z * np.sqrt((p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / (1 + z ** 2 / n)
    win_rates["ci_lower"] = (center - margin) * 100
    win_rates["ci_upper"] = (center + margin) * 100

    return win_rates.sort_values("win_rate", ascending=False)


def plot_win_rates(df):
    win_rates = calculate_win_rates(df)

    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot bars with error bars (95% CI)
    y_pos = np.arange(len(win_rates))
    bars = ax.bar(
        y_pos,
        win_rates["win_rate_pct"],
        yerr=[win_rates["win_rate_pct"] - win_rates["ci_lower"], win_rates["ci_upper"] - win_rates["win_rate_pct"]],
        capsize=8,
        alpha=0.85,
        edgecolor="black",
        color="#4C72B0",
        linewidth=1.2
    )

    # Add win counts as text labels on bars
    for i, row in win_rates.iterrows():
        ax.text(i, row["win_rate_pct"] + 0.5,
                f'{int(row["wins"])}/{int(row["total_datasets"])}',
                ha='center', va='bottom', fontsize=9, weight='bold')

    # Formatting
    ax.set_xticks(y_pos)
    ax.set_xticklabels(win_rates["feature_selection_method"], rotation=45, ha="right", fontsize=10)
    ax.set_title(PLOT_TITLE, fontsize=16, weight='bold', pad=20)
    ax.set_xlabel(X_LABEL, fontsize=12)
    ax.set_ylabel(Y_LABEL, fontsize=12)

    # Format y-axis as percentages
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylim(0, max(win_rates["ci_upper"]) * 1.1)

    # Add grid and legend
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / PLOT_NAME
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Win rates plot saved to {out}")
    print("\n📊 Win Rate Summary:")
    print(win_rates[["feature_selection_method", "wins", "total_datasets", "win_rate_pct"]].round(1))


# Do nothing below
SCRIPT_DIR = Path(__file__).parent / "../../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/performance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    plot_win_rates(df)


if __name__ == "__main__":
    main()