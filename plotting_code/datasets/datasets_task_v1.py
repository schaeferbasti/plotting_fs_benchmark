from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# TODO: Adapt file and plot name
FILE_NAME = "data_foundry.csv"
PLOT_NAME = "dataset_task_v1.png"

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

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plot of counts per task
    bins_pos = np.arange(len(task_counts))
    ax.bar(bins_pos, task_counts.values, color="#4C72B0", alpha=0.8, edgecolor="black")

    ax.set_xticks(bins_pos)
    ax.set_xticklabels(task_counts.index, rotation=45, ha="right")
    ax.set_title("Datasets by Task")
    ax.set_xlabel("Task")
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
