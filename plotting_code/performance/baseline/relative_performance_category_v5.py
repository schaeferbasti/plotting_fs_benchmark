from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils.beautify import beautify_names, add_model_name, remove_jmi

FILE_NAME = "results_per_split.csv"
PLOT_NAME = "relative_performance_category_v5.png"
PLOT_TITLE = "LightGBM Max"
X_LABEL = ""
Y_LABEL = "Improvement over Random (%)"


def calculate_relative_performance(df):
    df = df.copy()

    df = beautify_names(df)
    df = remove_jmi(df)
    df = add_model_name(df)

    df = df[df["model_cls"] == "LGBModel"].copy()

    category_mapping = {
        # Filters
        "ANOVA": "correlation",
        "ElasticNet": "regularization",
        "LaplacianScore": "distance",
        "MutualInformation": "info-theory",
        "Random": "random",
        "RFImportance": "tree",
        "RFE": "backward-search",
        "SFS": "forward-search",
    }

    df["feature_selection_method_category"] = df["feature_selection_method"].map(category_mapping)

    random_df = df[df["feature_selection_method_category"] == "random"]
    random_baseline = (
        random_df.groupby("tid")["metric_error"]
        .mean()
        .reset_index()
        .rename(columns={"metric_error": "random_error"})
    )

    df_merged = df.merge(random_baseline, on="tid", how="left")

    df_merged["improvability"] = (
         (df_merged["random_error"] - df_merged["metric_error"])
         / df_merged["random_error"]
    ) * 100



    agg_df = (
        df_merged.groupby("feature_selection_method_category")["improvability"]
        .agg(["mean", "std"])
        .reset_index()
    )

    agg_df.columns = [
        "feature_selection_method_category",
        "mean_improvability",
        "std_improvability",
    ]

    agg_df = agg_df[agg_df["feature_selection_method_category"] != "random"]

    return agg_df


def plot_relative(df):
    agg_df = calculate_relative_performance(df)

    if agg_df.empty:
        return

    # FIX 1: Sort by improvement (best first: most POSITIVE value = best performance)
    agg_df = agg_df.sort_values("mean_improvability", ascending=True)

    methods = agg_df["feature_selection_method_category"].values
    mean_improv = agg_df["mean_improvability"].values
    std_improv = agg_df["std_improvability"].values

    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(methods))

    # FIX 2: Color code: Green if positive (better than random), Red if negative (worse)
    # I updated the hex codes here so it actually plots Green (#55A868) and Red (#C44E52)
    colors = ["#4C72B0" if score >= 0 else "#4C72B0" for score in mean_improv]

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