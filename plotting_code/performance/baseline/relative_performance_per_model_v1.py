from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from utils.beautify import beautify_names, add_model_name, remove_jmi
from utils.scaling import tabarena_normalization

FILE_NAME = "results_per_split.csv"
PLOT_NAME = "relative_performance_per_model_v3.png"
PLOT_TITLE = ""
X_LABEL = ""
Y_LABEL = "Improvability (%)"


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
        random_errors.groupby("tid")["metric_error"]
        .mean()
        .reset_index()
        .rename(columns={"metric_error": "random_error"})
    )

    # 3. Merge best error back
    df_merged = df_avg.merge(random_baseline, on="tid", how="left")

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

    return pivot

def plot_relative(df):
    pivot = calculate_relative_performance(df)

    # FIX 1: Sort by improvement (best first: most POSITIVE value = best performance)
    # ascending=False puts the highest positive values at the beginning
    methods = sorted(pivot.index)
    model_names = sorted(pivot.columns)

    fig, ax = plt.subplots(figsize=(16, 7))

    cmap = plt.get_cmap("Set3", len(model_names))
    colors = {m: cmap(i) for i, m in enumerate(model_names)}

    x = np.arange(len(methods))
    width = 0.8 / len(model_names)

    for j, model in enumerate(model_names):
        values = pivot[model].reindex(methods).values
        ax.bar(x + j * width, values, width=width, color=colors[model], label=model)

    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(methods, rotation=45, ha="right")
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