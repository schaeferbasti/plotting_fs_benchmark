from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from utils.beautify import beautify_names, add_model_name, remove_jmi

FILE_NAME = "results_per_split.csv"
PLOT_NAME = "relative_performance_per_model_v5.pdf"
PLOT_TITLE = ""
X_LABEL = ""
Y_LABEL = "Improvement from Method Average (%)" # Normalized by the Avg. Improvement of each Method


def calculate_relative_performance(df):
    df = df.copy()

    df = beautify_names(df)
    df = remove_jmi(df)
    df = add_model_name(df)

    # 1. Average budget, models (and splits)
    df_avg = (
        df.groupby(["tid", "feature_selection_method", "model_cls"])["metric_error"]
        .mean()
        .reset_index()
    )

    # 2. Best error per dataset
    random_errors = df_avg[df_avg["feature_selection_method"] == "Random"].copy()

    random_baseline = (
        random_errors.groupby(["tid", "model_cls"])["metric_error"]
        .mean()
        .reset_index()
        .rename(columns={"metric_error": "random_error"})
    )

    # 3. Merge best error back using BOTH tid and model_cls
    df_merged = df_avg.merge(random_baseline, on=["tid", "model_cls"], how="left")

    # 4. Improvability = how much error remains until best, relative to own error
    df_merged["improvability"] = (
                                         (df_merged["random_error"] - df_merged["metric_error"])
                                         / (df_merged["random_error"])
                                 ) * 100

    pivot = df_merged.pivot_table(
        values="improvability",
        index="feature_selection_method",
        columns="model_cls",
        aggfunc="mean"
    ).fillna(np.nan)

    if "Random" in pivot.index:
        pivot = pivot.drop(index="Random")

    # We no longer calculate or append an "All Methods" row here,
    # because the baseline is now 0.0 by definition!

    # Normalization: SUBTRACT the method average instead of dividing by it
    method_avg = pivot.mean(axis=1)
    pivot = pivot.sub(method_avg, axis="index")

    return pivot


def plot_relative(df):
    pivot = calculate_relative_performance(df)

    model_rename_map = {
        "LGBModel": "LGBM",
        "LinearModel": "LM",
        "RFModel": "RF",
        "TabICLv2Model": "TabICLv2"
    }

    # Apply the renaming to the columns of the pivot table
    pivot = pivot.rename(columns=model_rename_map)

    methods = sorted(pivot.index)
    model_names = sorted(pivot.columns)

    fig, ax = plt.subplots(figsize=(8, 4.2))

    colors = {
        "LGBM": "#0072B2",  # Light Blue
        "LM": "#F0E442",  # Yellow
        "RF": "#009E73",  # Green
        "TabICLv2": "#D55E00",  # Red
    }

    # Standard x coordinates (no gap needed anymore since "All Methods" is gone)
    x = np.arange(len(methods), dtype=float)

    total_width = 0.8
    width_per_model = total_width / len(model_names)

    # Plot the divergent bars extending from 0.0
    for j, model in enumerate(model_names):
        values = pivot[model].reindex(methods).values

        # Calculate exactly where this bar should sit on the x-axis
        offset = -(total_width / 2) + (j + 0.5) * width_per_model

        # The bottom of the bar is 0.0 (the default for ax.bar)
        ax.bar(
            x + offset,
            values,  # The value is already the delta from the average!
            width=width_per_model * 0.9,
            color=colors.get(model, "#333333"),
            linewidth=0.5,
            label=model
        )

    # Align the ticks perfectly with the x-coordinates
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")

    # Draw a prominent horizontal line at 0.0 to act as the divergent baseline
    ax.axhline(y=0.0, color="black", linestyle="-", alpha=0.8, linewidth=1)

    ax.set_title(PLOT_TITLE)
    ax.set_xlabel(X_LABEL)

    # Update ylabel to reflect the new math
    ax.set_ylabel(Y_LABEL)
    ax.grid(True, alpha=0.3, axis="y")

    # Create Legend
    legend_elements = [Patch(facecolor=colors.get(m, "#333333"), label=m) for m in model_names]
    ax.legend(
        handles=legend_elements,
        loc="upper right",  # Anchor to the bottom-left corner
        title="Model",
        framealpha=0.5,  # Make the legend background slightly transparent
        edgecolor="black"  # Give it a clean border
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
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/performance/baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    plot_relative(df)


if __name__ == "__main__":
    main()