from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils.beautify import beautify_names, add_model_name, remove_jmi
from utils.scaling import tabarena_normalization

FILE_NAME = "results_per_split.csv"
PLOT_NAME = "relative_performance_v3.png"
PLOT_TITLE = ""
X_LABEL = ""
Y_LABEL = "Improvability (%)"


def calculate_relative_performance(df):
    df = df.copy()

    df = beautify_names(df)
    df = remove_jmi(df)
    df = add_model_name(df)

    # 1. Average budget, models (and splits)
    df_avg = (
        df.groupby(["tid", "feature_selection_method"])["metric_error"]
        .mean()
        .reset_index()
    )

    # 2. Best error per dataset
    best_errors = (
        df_avg.groupby("tid")["metric_error"]
        .min()
        .reset_index()
        .rename(columns={"metric_error": "best_error"})
    )

    # 3. Merge best error back
    df_merged = df_avg.merge(best_errors, on="tid", how="left")

    # 4. Improvability = how much error remains until best, relative to own error
    df_merged["improvability"] = (
        (df_merged["metric_error"] - df_merged["best_error"])
        / (df_merged["metric_error"])
    ) * 100

    # 5. Aggregate across datasets
    agg_df = (
        df_merged.groupby("feature_selection_method")["improvability"]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg_df.columns = [
        "feature_selection_method",
        "mean_improvability",
        "std_improvability",
    ]

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