from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from utils.beautify import beautify_names, remove_jmi
from utils.scaling import tabarena_normalization

FILE_NAME = "results_per_split_with_tuned.csv"
PLOT_NAME = "performance_per_budget_v6.pdf"
PLOT_TITLE = ""
X_LABEL = ""
Y_LABEL = "Normalized Score"


def calculate_relative_performance(df):
    df = df.copy()


    # Identify if a row is a tuned version using regex
    df["is_tuned"] = df["model_details"].str.contains(r"_r\d{1,2}(?!\d)", regex=True)

    # Extract the exact config identifier (e.g., "_r2" or "_c1") so we can group by it
    # If no config is found, default to "_c1"
    df["config_id"] = df["model_details"].str.extract(r"(_[rc]\d{1,2}(?!\d))")[0].fillna("_c1")

    df = beautify_names(df)
    df = remove_jmi(df)

    df = tabarena_normalization(df)

    df = df[df["feature_selection_method"] != "AccuracyLinear"]

    df["budget_stage"] = df.groupby("tid")["max_features"].rank(method="dense").astype(int)
    df = df[df["budget_stage"] <= 5]

    # 1. Average across datasets/splits for EACH specific configuration (e.g., _r2, _r3)
    df_config_avg = (
        df.groupby(["feature_selection_method", "budget_stage", "config_id"])["normalized_score"]
        .mean()
        .reset_index()
    )

    # 2. For each method and model variant, pick the BEST configuration score
    # For untuned (_c1) this does nothing, but for Tuned it selects the best _rX
    df_best = (
        df_config_avg.groupby(["feature_selection_method", "budget_stage"])["normalized_score"]
        .max()
        .reset_index()
    )

    # 3. Pivot the table for plotting
    pivot = df_best.pivot_table(
        values="normalized_score",
        index="feature_selection_method",
        columns="budget_stage",
        aggfunc="first" # It's already aggregated, so "first" or "mean" does the same here
    ).fillna(np.nan)


    return pivot


def plot_relative(df):
    pivot = calculate_relative_performance(df)

    methods = sorted(pivot.index)
    budgets = sorted(pivot.columns)

    fig, ax = plt.subplots(figsize=(10, 4))

    # Color per method — colorblind-friendly palette (Wong)
    method_colors = [
        "#0072B2", "#E69F00", "#009E73", "#D55E00",
        "#CC79A7", "#56B4E9", "#F0E442", "#000000",
        "#999999", "#882255", "#44AA99", "#DDCC77",
    ]
    colors = {method: method_colors[i % len(method_colors)] for i, method in enumerate(methods)}

    x = np.arange(len(budgets), dtype=float)
    total_width = 0.85
    width_per_method = total_width / len(methods)

    for j, method in enumerate(methods):
        values = [pivot.loc[method, b] if b in pivot.columns else np.nan for b in budgets]
        offset = -(total_width / 2) + (j + 0.5) * width_per_method

        ax.bar(
            x + offset,
            values,
            width=width_per_method * 0.9,
            color=colors[method],
            edgecolor="black",
            linewidth=0.5,
            label=method
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Budget {b}" for b in budgets], rotation=0, ha="center")
    ax.axhline(y=0.0, color="black", linestyle="-", alpha=0.2, linewidth=1)

    ax.set_title(PLOT_TITLE)
    ax.set_xlabel("Budget Stage")
    ax.set_ylabel(Y_LABEL)
    ax.grid(True, alpha=0.3, axis="y")

    legend_elements = [
        Patch(facecolor=colors[m], edgecolor="black", label=m)
        for m in methods
    ]

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        title="Method",
        framealpha=0.9,
        edgecolor="black",
        ncol=1
    )

    for side in ['left', 'bottom', 'right', 'top']:
        ax.spines[side].set_color("black")
        ax.spines[side].set_alpha(0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / PLOT_NAME
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ Relative Performance plot saved to {out}")


SCRIPT_DIR = Path(__file__).parent / "../../../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/performance/normalization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    plot_relative(df)


if __name__ == "__main__":
    main()