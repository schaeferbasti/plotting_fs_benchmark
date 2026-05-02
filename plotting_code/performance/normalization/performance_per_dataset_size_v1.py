import ast
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from utils.beautify import beautify_names, remove_jmi, add_model_name
from utils.scaling import tabarena_normalization

""" 
Description:
Performance per dataset size, tight point clusters
"""

FILE_NAME = "results_per_split.csv"
PLOT_NAME = "performance_per_dataset_size_v1.png"

PLOT_TITLE = "Performance by Dataset Size (FS Methods)"
X_LABEL = "Dataset Size (# Features)"
Y_LABEL = "Mean Score"



def parse_feature_size(x):
    """Safely extracts the number of features, whether it's a list, a string, or an integer."""
    if pd.isna(x):
        return np.nan

    # If Pandas already loaded it as a number
    if isinstance(x, (int, float)):
        return int(x)

    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x.strip())
            if isinstance(parsed, list):
                return len(parsed)
            elif isinstance(parsed, (int, float)):
                return int(parsed)
        except (ValueError, SyntaxError):
            # Fallback if literal_eval fails (e.g., plain comma-separated string)
            if ',' in x:
                return len(x.split(','))

    # If Pandas already loaded it as a list object
    if isinstance(x, list):
        return len(x)

    return np.nan


def add_dataset_size_from_validity(df_performance, df_validity):
    df_performance = df_performance.copy()
    df_val = df_validity.copy()

    # 1. Safely calculate the dataset size using the bulletproof parser
    df_val['dataset_size'] = df_val['original_features'].apply(parse_feature_size)

    # 2. Extract the clean 'tid' from the long 'data_foundry_task_id' string
    df_val["tid"] = df_val["data_foundry_task_id"].apply(
        lambda x: int(str(x).split("|")[1]) if pd.notna(x) and "|" in str(x) else np.nan
    )

    # 3. Create a clean mapping dictionary: {tid: dataset_size}
    val_dedup = df_val.dropna(subset=["tid", "dataset_size"]).drop_duplicates(subset=["tid"], keep="first")
    size_mapping = val_dedup.set_index("tid")["dataset_size"].to_dict()

    # 4. Map the dataset sizes onto the performance dataframe
    df_performance["tid"] = df_performance["tid"].astype(int)
    df_performance["dataset_size"] = df_performance["tid"].map(size_mapping)

    # (Optional) Print a warning if any datasets couldn't be matched
    missing_sizes = df_performance["dataset_size"].isna().sum()
    if missing_sizes > 0:
        print(f"⚠️ Warning: Could not find original dataset size for {missing_sizes} rows.")

    return df_performance


def plot(df):
    df = df.copy()
    df = beautify_names(df)
    df = remove_jmi(df)
    df = add_model_name(df)

    df = tabarena_normalization(df)

    df_val = pd.read_csv(SCRIPT_DIR / "result_files" / "validity_results.csv", low_memory=False)
    df = add_dataset_size_from_validity(df, df_val)

    bins = [50, 100, 500, 1000, 5000, 10000, np.inf]
    labels = ['10-50F', '50-100F', '100-500F', '500-1000F', '1000-5000F', '5000-10000F' '>10000F']
    df['size_bin'] = pd.cut(df['dataset_size'], bins=bins, labels=labels, include_lowest=True)

    groups = df.dropna(subset=["feature_selection_method", "size_bin", "normalized_score"])

    pivot = groups.pivot_table(
        values="metric_error",
        index="size_bin",
        columns="feature_selection_method",   # ["feature_selection_method", "tid"],
        aggfunc="mean",
        observed=False
    ).fillna(np.nan)

    sizes = pivot.index
    methods = pivot.columns

    fig, ax = plt.subplots(figsize=(14, 8))

    cmap = plt.get_cmap("tab10", len(methods))
    colors = {m: cmap(i) for i, m in enumerate(methods)}

    x = np.arange(len(sizes))

    # TIGHT ALIGNMENT: minimal spread 0.05 width
    spread = 0.05
    for j, method in enumerate(methods):
        values = pivot[method].reindex(sizes).values

        # Tiny offsets: all points nearly aligned vertically
        offsets = (j - len(methods) / 2) * spread / len(methods)
        x_pos = x + offsets

        ax.scatter(x_pos, values, color=colors[method], label=method,
                   s=120, alpha=0.9, edgecolors='black', linewidth=1.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(sizes, rotation=0)
    ax.set_title(PLOT_TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.grid(True, alpha=0.3, axis="y")

    legend_elements = [Patch(facecolor=colors[m], edgecolor='black', label=m)
                       for m in methods]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    out = OUTPUT_DIR / PLOT_NAME
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


SCRIPT_DIR = Path(__file__).parent / "../../../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/performance/normalization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    plot(df)


if __name__ == "__main__":
    main()