from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from utils.beautify import beautify_names, add_model_name, remove_jmi
from utils.scaling import tabarena_normalization

FILE_NAME = "results_per_split.csv"
PLOT_NAME = "relative_performance_per_model_v4.pdf"
PLOT_TITLE = ""
X_LABEL = ""
Y_LABEL = "Improvability over Random (%) Normalized by the Avg. Improvement per Model"


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
    pivot.loc["All Methods"] = pivot.mean(axis=0)
    all_method_avg_per_model = pivot.loc["All Methods"]
    pivot = pivot.div(all_method_avg_per_model, axis="columns")

    return pivot


def plot_relative(df):
    pivot = calculate_relative_performance(df)

    # Sort methods, ensuring "All Methods" is at the end
    methods = sorted([m for m in pivot.index if m != "All Methods"])
    methods.append("All Methods")

    model_names = sorted(pivot.columns)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {
        "LGBModel": "#0072B2",  # Light Blue
        "LinearModel": "#F0E442",  # Yellow
        "RFModel": "#009E73",  # Green
        "TabICLv2Model": "#D55E00",  # Red
    }

    # --- NEW: Create x coordinates, but add a gap of 0.5 for the last item ---
    x = np.arange(len(methods), dtype=float)
    x[-1] += 0.5

    width = 0.8 / len(model_names)

    for j, model in enumerate(model_names):
        values = pivot[model].reindex(methods).values

        # Plot the bars and save them to a variable
        bars = ax.bar(x + j * width, values, width=width, color=colors[model], label=model)

        # --- NEW: Add a bounding box around the "All Methods" bar (which is the last one) ---
        #bars[-1].set_edgecolor("black")
        #bars[-1].set_linewidth(2)
        # Optional: You can even add a hatch pattern to make it pop more
        # bars[-1].set_hatch("//")

    # Align the ticks perfectly with the shifted x-coordinates
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(methods, rotation=45, ha="right")

    # --- NEW: Add a subtle vertical dashed line to separate the average ---
    # Placed right between the last individual method and the shifted "All Methods"
    separator_x = x[-2] + 1
    ax.axvline(x=separator_x, color="gray", linestyle="--", alpha=0.7)

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