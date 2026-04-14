from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# TODO: Adapt file and plot name
FILE_NAME = "data_foundry.csv"
PLOT_NAME = "dataset_features_samples_v1.png"

# TODO: Adapt title and labels
PLOT_TITLE = "Dummy Plot"
X_LABEL = "Feature Selection Method"
Y_LABEL = "Metric Error"

# TODO: Adapt plotting function
def plot(df):
    # Parse number of features and samples (adjust column names as needed)
    df["n_features"] = pd.to_numeric(df["# features"], errors="coerce")
    df["n_samples"] = pd.to_numeric(df["samples"], errors="coerce")

    # Filter valid data
    dataset_data = df.dropna(subset=["n_features", "n_samples"])

    fig, ax = plt.subplots(figsize=(12, 8))

    # Scatter plot: samples (x) vs features (y)
    scatter = ax.scatter(
        dataset_data["n_samples"],
        dataset_data["n_features"],
        alpha=0.7,
        s=80,
        edgecolors="black",
        linewidth=1,
        c="tab:blue"
    )

    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Number of Features")
    ax.set_title("Dataset Characteristics (Samples vs Features)")
    ax.grid(True, alpha=0.3)

    # Log scale for better visualization (common for dataset characteristics)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Add dataset count
    ax.text(0.02, 0.98, f"N = {len(dataset_data)} datasets",
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

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
