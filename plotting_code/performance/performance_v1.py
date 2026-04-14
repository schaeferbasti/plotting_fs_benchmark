from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

""" 
Description:
This plot shows the 
"""

# TODO: Adapt file and plot name
FILE_NAME = "dummy_performance_results.csv"
PLOT_NAME = "performance_v1.png"

# TODO: Adapt title and labels
PLOT_TITLE = "Performance"
X_LABEL = "Feature Selection Method"
Y_LABEL = "Metric Error"


# TODO: Adapt plotting function
def plot(df):
    # Group by feature selection method only
    groups = df.dropna(subset=["feature_selection_method", "metric_error"])

    # Aggregate mean performance across all models per method
    agg_df = groups.groupby("feature_selection_method")["metric_error"].agg(["mean", "std"]).reset_index()
    agg_df.columns = ["feature_selection_method", "mean_error", "std_error"]

    # Sort by mean performance (best first: lower error = better)
    agg_df = agg_df.sort_values("mean_error", ascending=True)

    methods = agg_df["feature_selection_method"].values
    means = agg_df["mean_error"].values
    stds = agg_df["std_error"].values

    fig, ax = plt.subplots(figsize=(16, 8))

    x = np.arange(len(methods))

    bars = ax.bar(
        x,
        means,
        yerr=stds,
        capsize=5,
        alpha=0.85,
        edgecolor="black",
        color="#4C72B0"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_title(PLOT_TITLE)
    ax.set_xlabel("Feature Selection Method")
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
