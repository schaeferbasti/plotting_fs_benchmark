from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# TODO: Adapt file and plot name
FILE_NAME = "data_foundry.csv"
PLOT_NAME = "dataset_age_v1.png"

# TODO: Adapt title and labels
PLOT_TITLE = "Dummy Plot"
X_LABEL = "Feature Selection Method"
Y_LABEL = "Metric Error"

# TODO: Adapt plotting function
def plot(df):
    df["dataset_year"] = pd.to_numeric(df["Year"], errors="coerce")

    # Filter valid years
    years = df["dataset_year"].dropna()

    # Bin into decades (adjust bins as needed for your data)
    bins = [1980, 1990, 2000, 2010, 2020, 2030]
    labels = ["1980s", "1990s", "2000s", "2010s", "2020s"]
    year_bins = pd.cut(years, bins=bins, labels=labels, right=False, include_lowest=True)

    # Count datasets per bin
    bin_counts = year_bins.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plot of counts per decade
    bins_pos = np.arange(len(bin_counts))
    ax.bar(bins_pos, bin_counts.values, color="#4C72B0", alpha=0.8, edgecolor="black")

    ax.set_xticks(bins_pos)
    ax.set_xticklabels(bin_counts.index, rotation=0)
    ax.set_title("Dataset Publication Decade")
    ax.set_xlabel("Publication Decade")
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
