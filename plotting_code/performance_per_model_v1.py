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
PLOT_NAME = "performance_per_model_v1.png"

# TODO: Adapt title and labels
PLOT_TITLE = "Performance"
X_LABEL = "Feature Selection Method"
Y_LABEL = "Metric Error"

# TODO: Adapt plotting function
def plot(df):
    df["model_name"] = df["model_details"].apply(extract_model_name)

    groups = df.dropna(subset=["feature_selection_method", "metric_error", "model_name"])

    pivot = groups.pivot_table(
        values="metric_error",
        index="feature_selection_method",
        columns="model_name",
        aggfunc="mean"
    ).fillna(np.nan)

    methods = sorted(pivot.index)
    model_names = sorted(pivot.columns)

    fig, ax = plt.subplots(figsize=(16, 7))

    cmap = plt.get_cmap("Set3", len(model_names))
    colors = {m: cmap(i) for i, m in enumerate(model_names)}

    x = np.arange(len(methods))
    width = 0.8 / len(model_names)

    for j, model in enumerate(model_names):
        values = pivot[model].reindex(methods).values
        ax.bar(x + j * width, values, width=width, color=colors[model], label=model)

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_title(PLOT_TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.grid(True, alpha=0.3, axis="y")

    legend_elements = [Patch(facecolor=colors[m], label=m) for m in model_names]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 0.75),
              loc="upper left", title="Model")

    plt.tight_layout()
    out = OUTPUT_DIR / PLOT_NAME
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

# Do nothing below
SCRIPT_DIR = Path(__file__).parent / "../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots"
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
