from pathlib import Path
import ast
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

FILE_NAME = "dummy_performance_results.csv"
PLOT_NAME = "performance_per_model_per_max_features_v1.png"

PLOT_TITLE = "Performance per Model and max_features"
X_LABEL = "Feature Selection Method"
Y_LABEL = "Mean Metric Error"


def extract_model_name(model_details):
    if pd.isna(model_details):
        return "Unknown"
    try:
        d = ast.literal_eval(model_details)
        model_cls = d.get("model_cls", "Unknown")
        model_type = d.get("model_type", "")
        return f"{model_cls} ({model_type})" if model_type else model_cls
    except (ValueError, SyntaxError):
        return "Unknown"


def plot(df):
    df = df.copy()
    df["model_name"] = df["model_details"].apply(extract_model_name)

    groups = df.dropna(
        subset=["feature_selection_method", "metric_error", "model_name", "max_features"]
    )

    agg = (
        groups.groupby(["feature_selection_method", "model_name", "max_features"], as_index=False)
        .agg(mean_metric_error=("metric_error", "mean"))
    )

    methods = sorted(agg["feature_selection_method"].unique())
    models = sorted(agg["model_name"].unique())
    max_feats = sorted(agg["max_features"].unique())

    color_map = plt.get_cmap("Set2", len(models))
    model_colors = {model: color_map(i) for i, model in enumerate(models)}

    hatches = ["", "//", "\\\\", "xx", "..", "++", "--", "oo"]
    hatch_map = {mf: hatches[i % len(hatches)] for i, mf in enumerate(max_feats)}

    n_models = len(models)
    n_max = len(max_feats)
    total_bars_per_method = n_models * n_max
    group_width = 0.85
    bar_width = group_width / total_bars_per_method

    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(max(14, len(methods) * 0.8), 7))

    for m_idx, model in enumerate(models):
        for f_idx, mf in enumerate(max_feats):
            offset_idx = m_idx * n_max + f_idx
            offsets = x - group_width / 2 + offset_idx * bar_width + bar_width / 2

            vals = []
            for method in methods:
                row = agg[
                    (agg["feature_selection_method"] == method)
                    & (agg["model_name"] == model)
                    & (agg["max_features"] == mf)
                ]
                vals.append(row["mean_metric_error"].iloc[0] if not row.empty else np.nan)

            ax.bar(
                offsets,
                vals,
                width=bar_width,
                color=model_colors[model],
                hatch=hatch_map[mf],
                edgecolor="black",
                linewidth=0.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_title(PLOT_TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.grid(True, axis="y", alpha=0.3)

    model_legend = [
        Patch(facecolor=model_colors[m], edgecolor="black", label=m) for m in models
    ]
    maxfeat_legend = [
        Patch(facecolor="white", edgecolor="black", hatch=hatch_map[mf], label=f"max_features={mf}")
        for mf in max_feats
    ]

    leg1 = ax.legend(
        handles=model_legend,
        title="Model",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=maxfeat_legend,
        title="max_features",
        bbox_to_anchor=(1.02, 0.45),
        loc="upper left",
    )

    plt.tight_layout()
    out = OUTPUT_DIR / PLOT_NAME
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


SCRIPT_DIR = Path(__file__).parent / "../../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/performance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    plot(df)


if __name__ == "__main__":
    main()