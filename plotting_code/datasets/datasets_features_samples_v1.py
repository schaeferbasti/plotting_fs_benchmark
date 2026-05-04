from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# TODO: Adapt file and plot name
FILE_NAME = "data_foundry.csv"
PLOT_NAME = "dataset_features_samples_v1.png"

# TODO: Adapt title and labels
PLOT_TITLE = ""
X_LABEL = "Number of Samples"
Y_LABEL = "Number of Features"


def plot(df):
    # Parse number of features, samples, and classes
    df["n_features"] = pd.to_numeric(df["# features"], errors="coerce")
    df["n_samples"] = pd.to_numeric(df["samples"], errors="coerce")
    df["n_classes"] = pd.to_numeric(df["# classes"], errors="coerce")

    # Filter valid data for the axes
    dataset_data = df.dropna(subset=["n_features", "n_samples"])

    # Split into Classification (has classes) and Regression (NaN classes)
    df_class = dataset_data[dataset_data["n_classes"].notna()]
    df_reg = dataset_data[dataset_data["n_classes"].isna()]

    fig, ax = plt.subplots(figsize=(8, 7))

    # 1. Plot Classification datasets (Color-coded by num_classes)
    if not df_class.empty:
        scatter_class = ax.scatter(
            df_class["n_samples"],
            df_class["n_features"],
            c=df_class["n_classes"],
            cmap="Blues",
            vmin=0,  # Set to 0 so '2' isn't purely white/invisible, making it light blue
            vmax=df_class["n_classes"].max(),
            alpha=0.8,
            s=80,
            edgecolors="black",
            linewidth=1,
            label="Classification"
        )

        # Add a colorbar specifically for the classification scatter
        cbar = fig.colorbar(scatter_class, ax=ax, pad=0.02, shrink=0.6)
        cbar.set_label("Number of Classes", rotation=270, labelpad=15)

    # 2. Plot Regression datasets (Solid Green)
    if not df_reg.empty:
        scatter_reg = ax.scatter(
            df_reg["n_samples"],
            df_reg["n_features"],
            color="#55A868",  # A nice visible green
            alpha=0.8,
            s=80,
            edgecolors="black",
            linewidth=1,
            label="Regression"
        )

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title(PLOT_TITLE)
    ax.grid(True, alpha=0.3)

    # Log scale for better visualization
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Display the standard legend (shows Classification vs Regression dots)
    ax.legend(loc="upper right")

    plt.tight_layout()
    out = OUTPUT_DIR / PLOT_NAME
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ Plot saved to {out}")


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