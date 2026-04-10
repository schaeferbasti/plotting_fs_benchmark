from colorsys import rgb_to_hls, hls_to_rgb
from pathlib import Path
import ast
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cmx

FILE_NAME = "dummy_performance_results.csv"
PLOT_NAME = "performance_per_model_per_max_features_v4.png"

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
        .agg(
            mean_metric_error=("metric_error", "mean"),
            std_metric_error=("metric_error", "std"),
        )
    )

    methods = sorted(agg["feature_selection_method"].unique())
    models = sorted(agg["model_name"].unique())
    max_feats = sorted(agg["max_features"].unique())

    base_colors = plt.get_cmap("Set2", len(models))
    model_base_colors = {model: base_colors(i) for i, model in enumerate(models)}

    maxfeat_norm = plt.Normalize(vmin=min(max_feats), vmax=max(max_feats))
    model_colormaps = {}

    for model in models:
        base_color = np.array(model_base_colors[model])

        # Brighter low end, still model-tinted
        light_tint = base_color[:3] * 0.4 + 0.6
        light_tint = np.clip(light_tint, 0, 1)

        # Dark end stays model-colored
        r, g, b = base_color[:3]
        h, l, s = rgb_to_hls(r, g, b)
        dark_color = np.array(hls_to_rgb(h, max(0.08, l * 0.35), s))

        cmap = LinearSegmentedColormap.from_list(
            f"{model}_gradient",
            [light_tint, base_color[:3], dark_color]
        )
        model_colormaps[model] = cmap

    x_base = np.arange(len(methods))

    # Small offsets: max_features drives most of the spread, model only a little
    n_max = len(max_feats)
    n_models = len(models)

    maxfeat_spread = 0.10  # main compact cloud width
    model_spread = -0.2  # tiny model separation only

    maxfeat_offsets = (
        np.linspace(-maxfeat_spread / 2, maxfeat_spread / 2, n_max)
        if n_max > 1 else np.array([0.0])
    )
    model_offsets = (
        np.linspace(-model_spread / 2, model_spread / 2, n_models)
        if n_models > 1 else np.array([0.0])
    )

    maxfeat_to_offset = {mf: maxfeat_offsets[i] for i, mf in enumerate(max_feats)}
    model_to_offset = {model: model_offsets[i] for i, model in enumerate(models)}

    fig, ax = plt.subplots(figsize=(max(16, len(methods) * 0.55), 7))

    for model in models:
        for mf in max_feats:
            subset = agg[
                (agg["model_name"] == model) &
                (agg["max_features"] == mf)
            ]

            x_vals = []
            y_vals = []
            y_err = []

            for i, method in enumerate(methods):
                row = subset[subset["feature_selection_method"] == method]
                if not row.empty:
                    x_vals.append(x_base[i] + maxfeat_to_offset[mf] + model_to_offset[model])
                    y_vals.append(row["mean_metric_error"].iloc[0])
                    y_err.append(row["std_metric_error"].iloc[0])

            if x_vals:
                color_intensity = model_colormaps[model](maxfeat_norm(mf))

                ax.scatter(
                    x_vals,
                    y_vals,
                    s=60,  # smaller points for many methods
                    c=[color_intensity] * len(x_vals),
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.9,
                    zorder=3 + models.index(model),
                )

                """ax.errorbar(
                    x_vals,
                    y_vals,
                    yerr=y_err,
                    fmt="none",
                    ecolor=color_intensity,
                    elinewidth=0.8,
                    alpha=0.45,
                    capsize=1.5,
                    zorder=2,
                )"""

    ax.set_xticks(x_base)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_title(PLOT_TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)

    model_legend = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            linestyle="None",
            markerfacecolor=model_colormaps[model](0.5),
            markeredgecolor="black",
            markersize=8,
            label=model
        )
        for model in models
    ]

    bw_cmap = LinearSegmentedColormap.from_list("bw", ["#F5F5F5", "#555555"])
    sm = plt.cm.ScalarMappable(cmap=bw_cmap, norm=maxfeat_norm)
    sm.set_array([])

    leg1 = ax.legend(
        handles=model_legend,
        title="Model",
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
    )
    ax.add_artist(leg1)

    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, label="max_features")
    cbar.set_ticks([])
    cbar.ax.set_ylabel("max_features", rotation=270, labelpad=20)

    plt.subplots_adjust(right=0.80)
    out = OUTPUT_DIR / PLOT_NAME
    plt.savefig(out, dpi=150, bbox_inches="tight", pad_inches=1)
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