import ast
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

FILE_NAME = "results_per_split.csv"
PLOT_NAME = "performance_rank_v1.png"

PLOT_TITLE = "Performance Distribution (Global Rank)"
X_LABEL = "Feature Selection Method"
Y_LABEL = "Rank (1 = Best)"


def calculate_raw_ranks(df):
    df = df.copy()

    def extract_model_cls(model_details):
        details_dict = ast.literal_eval(str(model_details))
        return details_dict.get('model_cls', "Unknown")

    df["model_cls"] = df["model_details"].apply(extract_model_cls)

    # 3. AVERAGE PHASE
    # Group ONLY by Dataset, Metric, and Method.
    # Pandas averages over all splits, models, and budgets automatically.
    df_collapsed = df.groupby(
        ["tid", "metric", "feature_selection_method"]
    )["metric_error"].mean().reset_index()

    # 4. RANKING PHASE
    # Group ONLY by Dataset and Metric.
    # Ranks the methods against each other for that specific dataset.
    df_collapsed["rank"] = df_collapsed.groupby(
        ["tid"]
    )["metric_error"].rank(
        method="average",
        ascending=True,
        na_option="keep"
    )

    return df_collapsed


def plot_boxplot(df):
    ranked_df = calculate_raw_ranks(df)

    # Sort the boxes from best (left) to worst (right) by MEDIAN rank
    median_ranks = ranked_df.groupby("feature_selection_method")["rank"].median().reset_index()

    # Sort ascending so the smallest/best median rank is on the left of the x-axis
    median_ranks = median_ranks.sort_values("rank", ascending=True)
    sorted_methods = median_ranks["feature_selection_method"].tolist()

    data_to_plot = []
    for method in sorted_methods:
        method_data = ranked_df[ranked_df["feature_selection_method"] == method]["rank"].values
        data_to_plot.append(method_data)

    fig, ax = plt.subplots(figsize=(16, 8))

    # --- ADD BACKGROUND SCATTER POINTS ---
    np.random.seed(42)  # For reproducible jitter
    for i, method_data in enumerate(data_to_plot):
        # Drop NaNs so scatter doesn't complain
        y = method_data[~np.isnan(method_data)]
        # Add a little horizontal jitter so points don't completely overlap
        x = np.random.normal(i + 1, 0.08, size=len(y))

        # Plot points. zorder=1 ensures they stay behind the boxes
        ax.scatter(x, y, alpha=0.3, s=15, color='gray', zorder=1, edgecolors='none')

    # ---- CUSTOMIZE BOXPLOT PROPS ----
    boxprops = dict(linewidth=1.5, color="black", facecolor="#4C72B0", alpha=0.8)

    # We show BOTH the median (as the green line) and the mean (as the orange line)
    medianprops = dict(linewidth=2, visible=True, color="green")
    meanprops = dict(linestyle='-', linewidth=2, color='orange')

    # Create the boxplot
    bp = ax.boxplot(
        data_to_plot,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        meanprops=meanprops,
        medianprops=medianprops,
        showfliers=False,  # <-- Turned off fliers since we draw ALL points via scatter
        zorder=2,  # <-- Forces the boxes to draw on top of the scatter
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
    ax.invert_yaxis()  # Puts Rank 1 at the top of the Y-axis
    ax.grid(True, alpha=0.3, axis="y")

    # Add a custom legend to explicitly state the lines
    ax.plot([], [], color='orange', linestyle='-', linewidth=2, label='Mean Rank')
    ax.plot([], [], color='green', linestyle='-', linewidth=2, label='Median Rank')
    # Add scatter proxy to legend
    ax.scatter([], [], color='gray', alpha=0.5, s=30, label='Individual Datasets')
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