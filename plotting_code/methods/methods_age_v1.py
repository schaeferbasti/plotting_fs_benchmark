from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# TODO: Adapt file and plot name
FILE_NAME = "method_curation.csv"
PLOT_NAME = "method_age_v1.png"

# TODO: Adapt title and labels
PLOT_TITLE = "Dummy Plot"
X_LABEL = "Feature Selection Method"
Y_LABEL = "Metric Error"

# TODO: Adapt plotting function
def plot(df):
    # Parse Year columns (try both Paper and Code years)
    df = df[df["Final Decision "] == "Yes"].copy()

    df["year_paper"] = pd.to_numeric(df["Year (Paper)"], errors="coerce")
    df["year_code"] = pd.to_numeric(df["Year (Code)"], errors="coerce")

    # Use paper year if available, otherwise code year
    df["dataset_year"] = df["year_paper"].fillna(df["year_code"])

    # Filter valid years
    years = df["dataset_year"].dropna()

    # Bin into decades (adjust bins as needed for your data)
    bins = [1900, 1980, 1990, 2000, 2010, 2020]
    labels = ["earlier", "1980s", "1990s", "2000s", "2010s"]
    year_bins = pd.cut(years, bins=bins, labels=labels, right=False, include_lowest=True)

    # Count datasets per bin
    bin_counts = year_bins.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plot of counts per decade
    bins_pos = np.arange(len(bin_counts))
    ax.bar(bins_pos, bin_counts.values, color="#4C72B0", alpha=0.8, edgecolor="black")

    ax.set_xticks(bins_pos)
    ax.set_xticklabels(bin_counts.index, rotation=0)
    ax.set_title("Method Publication Decade")
    ax.set_xlabel("Publication Decade")
    ax.set_ylabel("Number of Methods")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = OUTPUT_DIR / PLOT_NAME
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


# Do nothing below
SCRIPT_DIR = Path(__file__).parent / "../../"
RESULTS_FILE = SCRIPT_DIR / "result_files/curation" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/methods"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    plot(df)


if __name__ == "__main__":
    main()
