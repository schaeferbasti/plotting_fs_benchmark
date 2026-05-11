from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from utils.beautify import beautify_names, add_model_name, remove_jmi
from utils.scaling import tabarena_normalization

FILE_NAME = "results_per_split_with_tuned.csv"
PLOT_NAME = "performance_per_model_tuned_v5.pdf"
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
    df = add_model_name(df)

    df = tabarena_normalization(df)

    df = df[df["feature_selection_method"] != "AccuracyLinear"]

    # Label the base model variant
    df["model_variant"] = df.apply(
        lambda row: f"{row['model_cls']} (Tuned)" if row["is_tuned"] else row["model_cls"],
        axis=1
    )

    # 1. Average across datasets/splits for EACH specific configuration (e.g., _r2, _r3)
    df_config_avg = (
        df.groupby(["feature_selection_method", "model_variant", "config_id"])["normalized_score"]
        .mean()
        .reset_index()
    )

    # 2. For each method and model variant, pick the BEST configuration score
    # For untuned (_c1) this does nothing, but for Tuned it selects the best _rX
    df_best = (
        df_config_avg.groupby(["feature_selection_method", "model_variant"])["normalized_score"]
        .max()
        .reset_index()
    )

    # 3. Pivot the table for plotting
    pivot = df_best.pivot_table(
        values="normalized_score",
        index="feature_selection_method",
        columns="model_variant",
        aggfunc="first" # It's already aggregated, so "first" or "mean" does the same here
    ).fillna(np.nan)


    return pivot


def plot_relative(df):
    pivot = calculate_relative_performance(df)

    # Rename base models, keeping the " (Tuned)" suffix intact if present
    model_rename_map = {
        "LGBModel": "LGBM",
        "LinearModel": "LM",
        "RFModel": "RF",
        "TabICLv2Model": "TabICLv2"
    }

    new_columns = {}
    for col in pivot.columns:
        base_name = col.replace(" (Tuned)", "")
        new_base = model_rename_map.get(base_name, base_name)
        new_columns[col] = f"{new_base} (Tuned)" if " (Tuned)" in col else new_base

    pivot = pivot.rename(columns=new_columns)

    # ==========================================================
    # NEW: Calculate and print the LightGBM tuning improvement
    # ==========================================================
    if "LGBM" in pivot.columns and "LGBM (Tuned)" in pivot.columns:
        # Calculate the absolute improvement for each method
        lgbm_improvement = pivot["LGBM (Tuned)"] - pivot["LGBM"]

        # Print detailed breakdown
        print("\n" + "=" * 60)
        print("📈 LightGBM Tuning Improvement (Normalized Score Delta)")
        print("=" * 60)
        for method, diff in lgbm_improvement.dropna().items():
            print(f"{method:25}: {'+' if diff > 0 else ''}{diff:.4f} pp")

        # Print the overall averages across all methods
        avg_untuned = pivot["LGBM"].mean()
        avg_tuned = pivot["LGBM (Tuned)"].mean()
        avg_improvement = lgbm_improvement.mean()

        # Calculate the relative improvement in percent
        # Handle division by zero just in case the untuned average is exactly 0
        if avg_untuned != 0:
            relative_improvement_pct = (avg_improvement / abs(avg_untuned)) * 100
        else:
            relative_improvement_pct = 0.0

        print("-" * 60)
        print(f"Average Default LGBM:       {avg_untuned:.4f}%")
        print(f"Average Tuned LGBM:         {avg_tuned:.4f}%")
        print(f"Absolute Improvement:       {'+' if avg_improvement > 0 else ''}{avg_improvement:.4f} pp")
        print(
            f"Relative Improvement:       {'+' if relative_improvement_pct > 0 else ''}{relative_improvement_pct:.2f}%")
        print("=" * 60 + "\n")
    # ==========================================================

    methods = sorted(pivot.index)
    model_names = sorted(pivot.columns)

    fig, ax = plt.subplots(figsize=(10, 4))

    # Define base colors for untuned, and lighter/hashed variants for tuned
    colors = {
        "LGBM": "#0072B2",
        "LGBM (Tuned)": "#56B4E9",  # Lighter blue
        "LM": "#F0E442",
        "RF": "#009E73",
        "TabICLv2": "#D55E00",
    }

    # Standard x coordinates
    x = np.arange(len(methods), dtype=float)

    total_width = 0.85
    width_per_model = total_width / len(model_names)

    # Plot the divergent bars extending from 0.0
    for j, model in enumerate(model_names):
        values = pivot[model].reindex(methods).values

        offset = -(total_width / 2) + (j + 0.5) * width_per_model

        is_tuned = " (Tuned)" in model
        hatch_pattern = "////" if is_tuned else ""

        ax.bar(
            x + offset,
            values,
            width=width_per_model * 0.9,
            color=colors.get(model, "#333333"),
            edgecolor="black" if is_tuned else "none",
            linewidth=0.5,
            hatch=hatch_pattern,
            label=model
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.axhline(y=0.0, color="black", linestyle="-", alpha=0.2, linewidth=1)

    ax.set_title(PLOT_TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.grid(True, alpha=0.3, axis="y")

    legend_elements = []
    for m in model_names:
        is_tuned = " (Tuned)" in m
        legend_elements.append(
            Patch(
                facecolor=colors.get(m, "#333333"),
                edgecolor="black" if is_tuned else "none",
                hatch="////" if is_tuned else "",
                label=m
            )
        )

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        title="Model",
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