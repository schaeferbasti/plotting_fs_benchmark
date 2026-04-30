from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

# TODO: Adapt file and plot name
FILE_NAME = "results_per_split.csv"
PLOT_NAME = "performance_per_model_v3.png"

# TODO: Adapt title and labels
PLOT_TITLE = "Performance per Downstream Model (z-score)"
X_LABEL = "Feature Selection Method"
Y_LABEL = "Z-Score (0 = Mean Performance, >0 is Better)"

# Define which metrics mean "lower is better" vs "higher is better"
METRIC_DIRECTIONS = {
    "log_loss": True,
    "rmse": True,
    "roc_auc": False
}


def extract_model_name(model_details):
    """Clean model name from dict string"""
    if pd.isna(model_details):
        return "Unknown"
    try:
        details = eval(model_details)  # Safe for your format
        return f"{details['model_cls']} ({details['model_type']})"
    except:
        return str(model_details)[:20] + "..."


def calculate_zscore_performance(df):
    df["model_name"] = df["model_details"].apply(extract_model_name)

    group_cols = [c for c in ["metric", "dataset"] if c in df.columns]
    if not group_cols:
        group_cols = ["metric"]

    required_cols = group_cols + ["feature_selection_method", "metric_error", "model_name"]
    df_clean = df.dropna(subset=required_cols).copy()

    # 1. Adjust metric values so "Higher is ALWAYS Better"
    def adjust_direction(row):
        is_lower_better = METRIC_DIRECTIONS.get(row["metric"], True)
        if is_lower_better:
            return -row["metric_error"]
        return row["metric_error"]

    df_clean["performance"] = df_clean.apply(adjust_direction, axis=1)

    # 2. Average CV splits (one row per dataset, metric, method, and model)
    df_collapsed = df_clean.groupby(
        group_cols + ["feature_selection_method", "model_name"]
    )["performance"].mean().reset_index()

    # 3. Z-Score Scale the performance per dataset/metric
    def z_score_scale(group):
        mean_val = group.mean()
        std_val = group.std()

        # If all methods perform identically, standard deviation is 0
        if std_val == 0 or pd.isna(std_val):
            return pd.Series(0.0, index=group.index)

        return (group - mean_val) / std_val

    # Apply the Z-score transformation group by group
    df_collapsed["scaled_score"] = df_collapsed.groupby(group_cols)["performance"].transform(z_score_scale)

    return df_collapsed


def plot(df):
    df_scaled = calculate_zscore_performance(df)

    # Create pivot table for plotting (Mean scaled score per method and model)
    pivot = df_scaled.pivot_table(
        values="scaled_score",
        index="feature_selection_method",
        columns="model_name",
        aggfunc="mean"
    ).fillna(np.nan)

    # Sort methods by their overall average across all models (Best method on the left)
    method_means = pivot.mean(axis=1).sort_values(ascending=False)
    methods = method_means.index.tolist()

    # Reorder pivot table based on sorted methods
    pivot = pivot.loc[methods]
    model_names = sorted(pivot.columns)

    fig, ax = plt.subplots(figsize=(16, 8))

    # Color map
    cmap = plt.get_cmap("Set3", len(model_names))
    colors = {m: cmap(i) for i, m in enumerate(model_names)}

    x = np.arange(len(methods))
    width = 0.8 / len(model_names)

    # Draw a baseline at 0 (The Mean Performance)
    ax.axhline(0, color="black", linewidth=1.5, linestyle="--", zorder=0)

    for j, model in enumerate(model_names):
        values = pivot[model].values
        # Shift bars so they sit side-by-side
        ax.bar(
            x + j * width,
            values,
            width=width,
            color=colors[model],
            label=model,
            edgecolor="black",
            linewidth=0.5,
            zorder=3
        )

    # Formatting
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=10)
    ax.set_title(PLOT_TITLE, fontsize=14, weight='bold', pad=15)
    ax.set_xlabel(X_LABEL, fontsize=12)
    ax.set_ylabel(Y_LABEL, fontsize=12)

    # Grid lines
    ax.grid(True, alpha=0.4, axis="y", linestyle="--", zorder=0)

    # Legend
    legend_elements = [Patch(facecolor=colors[m], edgecolor="black", label=m) for m in model_names]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1),
              loc="upper left", title="Downstream Model", title_fontsize=11)

    plt.tight_layout()
    out = OUTPUT_DIR / PLOT_NAME
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Z-Score performance per model plot saved to {out}")


# Do nothing below
SCRIPT_DIR = Path(__file__).parent / "../../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/performance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    plot(df)


if __name__ == "__main__":
    main()