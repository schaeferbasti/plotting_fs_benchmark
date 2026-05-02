from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from utils.average import average_per_dataset_and_method
from utils.beautify import add_model_name, remove_jmi, beautify_names

FILE_NAME = "results_per_split.csv"
PLOT_NAME = "rank_v2.png"

PLOT_TITLE = "Rank per Dataset"
X_LABEL = "Feature Selection Method"
Y_LABEL = "Rank (1 = Best)"


def calculate_raw_ranks(df):
    df = df.copy()

    df = beautify_names(df)
    df = remove_jmi(df)
    df = add_model_name(df)

    df = average_per_dataset_and_method(df)

    df["rank"] = df.groupby(
        ["tid"]
    )["metric_error"].rank(
        method="average",
        ascending=True,
        na_option="keep"
    )
    return df


def plot_boxplot(df):
    ranked_df = calculate_raw_ranks(df)

    # Sort the boxes from best (left) to worst (right) by MEDIAN rank
    median_ranks = ranked_df.groupby("feature_selection_method")["rank"].median().reset_index()
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
        y = method_data[~np.isnan(method_data)]
        x = np.random.normal(i + 1, 0.08, size=len(y))
        ax.scatter(x, y, alpha=0.6, s=20, color='gray', zorder=1, edgecolors='black', linewidth=0.3)

    # ---- CUSTOMIZE BOXPLOT PROPS ----
    # FIXED: Added alpha=0.4 for transparency!
    boxprops = dict(linewidth=1.5, color="black", facecolor="#4C72B0", alpha=0.4)

    medianprops = dict(linewidth=2, visible=True, color="green")
    meanprops = dict(linestyle='-', linewidth=2, color='orange')

    bp = ax.boxplot(
        data_to_plot,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        meanprops=meanprops,
        medianprops=medianprops,
        showfliers=False,
        zorder=2,
        widths=0.6,
    )

    # Set colors for the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('#4C72B0')
        patch.set_alpha(0.4)  # <-- Double-check that transparency is applied

    # Formatting
    ax.set_xticks(np.arange(1, len(sorted_methods) + 1))
    ax.set_xticklabels(sorted_methods, rotation=45, ha="right")
    ax.set_title(PLOT_TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)

    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="y")

    # Legend
    ax.plot([], [], color='orange', linestyle='-', linewidth=2, label='Mean Rank')
    ax.plot([], [], color='green', linestyle='-', linewidth=2, label='Median Rank')
    ax.scatter([], [], color='gray', alpha=0.6, s=20, edgecolors='black', linewidth=0.3, label='Individual Datasets')
    ax.legend(loc="upper right")

    plt.tight_layout()
    out = OUTPUT_DIR / PLOT_NAME
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ Boxplot saved to {out}")


# Do nothing below
SCRIPT_DIR = Path(__file__).parent / "../../../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/performance/ranking"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    plot_boxplot(df)


if __name__ == "__main__":
    main()