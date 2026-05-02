from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils.beautify import beautify_names, add_model_name, remove_jmi
from utils.scaling import tabarena_normalization

FILE_NAME = "results_per_split.csv"
PLOT_NAME = "relative_performance_v3.png"
PLOT_TITLE = "Improvability to Random Baseline"
X_LABEL = ""
Y_LABEL = "Improvability (Percentage points better/worse than Random)"


def calculate_relative_performance(df):
    df = df.copy()

    df = beautify_names(df)
    df = remove_jmi(df)
    df = add_model_name(df)

    # 1. NORMALIZE FIRST! This ensures RMSE and Accuracy are on the same scale
    # so that a difference of "0.1" means the same thing across all datasets.
    df = tabarena_normalization(df)

    # 2. Average out the noise (models, budgets, splits) to get 1 score per method/dataset/metric
    df_avg = df.groupby(["tid", "feature_selection_method"])["normalized_score"].mean().reset_index()

    # 3. Extract Random baseline
    random_scores = df_avg[df_avg["feature_selection_method"] == "Random"][
        ["tid", "normalized_score"]
    ]
    random_scores = random_scores.rename(columns={"normalized_score": "random_score"})

    # 4. Merge back
    df_merged = df_avg.merge(random_scores, on=["tid"], how="left")

    # 5. Calculate Absolute Percentage Point Improvement for SCORES (Higher is better)
    # Positive result = Model score is higher than Random score
    df_merged["improvability"] = (df_merged["normalized_score"] - df_merged["random_score"]) * 100

    # 6. Aggregate across all datasets
    agg_df = df_merged.groupby("feature_selection_method")["improvability"].agg(["mean", "std"]).reset_index()
    agg_df.columns = ["feature_selection_method", "mean_improvability", "std_improvability"]
    agg_df = agg_df[agg_df["feature_selection_method"] != "Random"]

    return agg_df


def plot_relative(df):
    agg_df = calculate_relative_performance(df)

    # FIX 1: Sort by improvement (best first: most POSITIVE value = best performance)
    # ascending=False puts the highest positive values at the beginning
    agg_df = agg_df.sort_values("mean_improvability", ascending=True)

    methods = agg_df["feature_selection_method"].values
    mean_improv = agg_df["mean_improvability"].values
    std_improv = agg_df["std_improvability"].values

    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(methods))

    # FIX 2: Color code: Green if positive (better than random), Red if negative (worse)
    colors = ["#55A868" if score >= 0 else "#C44E52" for score in mean_improv]

    bars = ax.bar(
        x,
        mean_improv,
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