from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils.beautify import beautify_names, add_model_name, remove_jmi

FILE_NAME = "results_per_split.csv"
PLOT_NAME = "time_category_v1.png"
PLOT_TITLE = ""
X_LABEL = ""
# Updated wording for time comparison
Y_LABEL = "Relative Time Overhead (%)"

def calculate_relative_performance(df):
    df = df.copy()

    df = beautify_names(df)
    df = remove_jmi(df)
    df = add_model_name(df)

    category_mapping = {
        # Filters
        "ANOVA": "Filter",
        "CART": "Embedded",
        "ElasticNet": "Embedded",
        "GainRatio": "Filter",
        "LaplacianScore": "Filter",
        "Lasso": "Embedded",
        "LOCO": "Wrapper",
        "MarkovBlanket": "Filter",
        "mRMR": "Filter",
        "MutualInformation": "Filter",
        "Random": "random",
        "ReliefF": "Filter",
        "RFImportance": "Embedded",
        "RFE": "Wrapper",
        "SFS": "Wrapper",
    }

    df["feature_selection_method_category"] = df["feature_selection_method"].map(category_mapping)

    df_avg = (
        df.groupby(["tid", "feature_selection_method_category"])["feature_selection_fit_time"]
        .mean()
        .reset_index()
    )

    random_baseline = df_avg[df_avg["feature_selection_method_category"] == "random"].copy()
    random_baseline = random_baseline[["tid", "feature_selection_fit_time"]].rename(columns={"feature_selection_fit_time": "random_time"})

    df_merged = df_avg.merge(random_baseline, on="tid", how="left")

    # FIX 1: Flipped the math.
    # (Method Time - Random Time) / Random Time.
    # Now, a positive number means it is SLOWER than Random (which is an overhead).
    df_merged["relative_delay"] = (
         (df_merged["feature_selection_fit_time"] - df_merged["random_time"])
         / df_merged["random_time"]
    ) * 100

    # Aggregate across datasets
    agg_df = (
        df_merged.groupby("feature_selection_method_category")["relative_delay"]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg_df.columns = [
        "feature_selection_method_category",
        "mean_relative_delay",
        "std_relative_delay",
    ]

    # Drop the random bar since it's the 0% baseline
    agg_df = agg_df[agg_df["feature_selection_method_category"] != "random"]

    return agg_df

def plot_relative(df):
    agg_df = calculate_relative_performance(df)

    # FIX 2: Sort ascending. Smallest overhead (fastest methods) will be on the left.
    agg_df = agg_df.sort_values("mean_relative_delay", ascending=True)

    methods = agg_df["feature_selection_method_category"].values
    mean_delay = agg_df["mean_relative_delay"].values
    std_delay = agg_df["std_relative_delay"].values

    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(methods))

    # FIX 3: Color code: Red if > 0 (slower than random, a penalty), Green if <= 0 (faster than random)
    colors = ["#4C72B0" if score > 0 else "#4C72B0" for score in mean_delay]

    bars = ax.bar(
        x,
        mean_delay,
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

    print(f"✅ Time Overhead plot saved to {out}")

SCRIPT_DIR = Path(__file__).parent / "../../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/time"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    plot_relative(df)

if __name__ == "__main__":
    main()