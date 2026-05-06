from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import textwrap  # <-- NEW: Import textwrap

# TODO: Adapt file and plot name
FILE_NAME = "data_foundry.csv"
PLOT_NAME = "dataset_task_v1.pdf"

# TODO: Adapt title and labels
PLOT_TITLE = "Dummy Plot"
X_LABEL = "Feature Selection Method"
Y_LABEL = "Metric Error"


# TODO: Adapt plotting function
def plot(df):
    # Use task column instead of year
    tasks = df["Problem Type"].dropna().unique()

    # Count datasets per task
    task_counts = df["Problem Type"].value_counts()

    fig, ax = plt.subplots(figsize=(3, 3))

    # Bar plot of counts per task
    bins_pos = np.arange(len(task_counts))
    ax.bar(bins_pos, task_counts.values, color="#0072B2", alpha=0.8, edgecolor="black")

    # --- NEW CODE: Use textwrap to format labels ---
    # Wraps text so that no single line is longer than ~12 characters
    multi_line_labels = [label.replace(" Classification", "") for label in task_counts.index]

    ax.set_xticks(bins_pos)
    ax.set_xticklabels(multi_line_labels, rotation=45, ha="center", va="top", fontsize=12)  # Slightly smaller font

    ax.set_title("Datasets by Problem Type")
    ax.set_ylabel("Number of Datasets")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = OUTPUT_DIR / PLOT_NAME
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


# Do nothing below
SCRIPT_DIR = Path(__file__).parent / "../../"
RESULTS_FILE = SCRIPT_DIR / "result_files/curation" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/datasets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    plot(df)


if __name__ == "__main__":
    main()