import ast
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils.average import average_per_method, average_per_dataset_and_method
from utils.beautify import beautify_names, remove_jmi, add_model_name
from utils.scaling import min_max_scale

FILE_NAME = "results_per_split.csv"
PLOT_NAME = "relative_performance_v1.png"
PLOT_TITLE = "Error Reduction compared to Random"
X_LABEL = "Feature Selection Method"
Y_LABEL = "Reduction Error"


def calculate_relative_performance(df):
    df = df.copy()

    df = beautify_names(df)
    df = remove_jmi(df)
    df = add_model_name(df)

    # 3. Extract the RandomFeatureSelector baseline errors
    random_errors = df[df["feature_selection_method"] == "Random"][
        ["tid", "metric", "metric_error"]
    ]
    random_errors = random_errors.rename(columns={"metric_error": "random_error"})

    # 4. Merge the baseline back
    df = df.merge(random_errors, on=["tid", "metric"], how="left")

    # FIX: Calculate Error Reduction (Positive value = less error than Random)
    df["error_reduction"] = df["random_error"] - df["metric_error"]

    # 5. Aggregate mean improvement and std per feature selection method
    agg_df = df.groupby("feature_selection_method")["error_reduction"].agg(["mean", "std"]).reset_index()
    agg_df.columns = ["feature_selection_method", "mean_reduction", "std_reduction"]
    agg_df["std_reduction"] = agg_df["std_reduction"].fillna(0)

    # 6. Drop the Random baseline
    agg_df = agg_df[agg_df["feature_selection_method"] != "Random"]

    return agg_df


def plot_relative(df):
    agg_df = calculate_relative_performance(df)

    # Sort by score (best first: higher score = better)
    agg_df = agg_df.sort_values("mean_reduction", ascending=False)

    methods = agg_df["feature_selection_method"].values
    scores = agg_df["mean_reduction"].values
    stds = agg_df["std_reduction"].values

    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(methods))

    # Color code: Green if positive (better than random), Red if negative (worse)
    colors = ["#55A868" if score >= 0 else "#C44E52" for score in scores]

    bars = ax.bar(
        x,
        scores,
        yerr=stds,
        capsize=5,
        alpha=0.85,
        edgecolor="black",
        color=colors
    )

    # Draw a solid horizontal line at Y=0 (The Random Baseline)
    ax.axhline(0, color="black", linewidth=1.5, linestyle="--", zorder=0)

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

    print(f"✅ Relative Performance plot saved to {out}")


SCRIPT_DIR = Path(__file__).parent / "../../../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/performance/baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    plot_relative(df)


if __name__ == "__main__":
    main()