import ast
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Set styling to match the light, clean look of the reference
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

FILE_NAME = "results_per_split.csv"
PLOT_NAME = "performance_rank_v2.png"  # PDF is usually better for papers!

PLOT_TITLE = ""  # The reference image has no title, just the legend
X_LABEL = ""
Y_LABEL = "Rank (1 = best)"

# Dictionary to map methods to their categories (Filter, Wrapper, Embedded, etc.)
# You may need to adjust these names to exactly match your feature_selection_method column
METHOD_CATEGORIES = {
    "RFE": "Wrapper",
    "RFImportance": "Embedded",
    "ElasticNet": "Embedded",
    "Lasso": "Embedded",
    "CART": "Embedded",
    "MI": "Filter",
    "ANOVA": "Filter",
    "mRMR": "Filter",
    "GainRatio": "Filter",
    "ReliefF": "Filter",
    "SFS": "Wrapper",
    "LOCO": "Wrapper",
    "MarkovBlanket": "Filter",
    "Random": "Filter",
    "LaplacianScore": "Filter"
}

# Styling mapping for categories
CATEGORY_STYLES = {
    "Filter": {"color": "#C6D8EB", "marker": "D", "label": "Filter"},  # Light Blue, Diamond
    "Wrapper": {"color": "#FBE2C4", "marker": "*", "label": "Wrapper"},  # Light Orange, Star
    "Embedded": {"color": "#C6D8EB", "marker": "X", "label": "Embedded"}  # Light Blue, Cross
}


def calculate_raw_ranks(df):
    df = df.copy()

    def extract_model_cls(model_details):
        if pd.isna(model_details):
            return "Unknown"
        details_dict = ast.literal_eval(str(model_details))
        return details_dict.get('model_cls', "Unknown")

    df["model_cls"] = df["model_details"].apply(extract_model_cls)

    # 3. AVERAGE PHASE
    df_collapsed = df.groupby(
        ["tid", "metric", "feature_selection_method"]
    )["metric_error"].mean().reset_index()

    # 4. RANKING PHASE
    df_collapsed["rank"] = df_collapsed.groupby(
        ["tid"]
    )["metric_error"].rank(
        method="average",
        ascending=True,
        na_option="keep"
    )

    return df_collapsed


def plot_boxplot(df):
    ranked_df = calculate_raw_ranks(df)

    # Clean method names if they have "FeatureSelector" at the end to match the dictionary
    ranked_df["clean_method"] = ranked_df["feature_selection_method"].str.replace("FeatureSelector", "")

    # Calculate mean and std for the x-axis labels (like "RFE \n 5.44 ± 2.91")
    stats = ranked_df.groupby("clean_method")["rank"].agg(["mean", "std", "median"]).reset_index()
    stats = stats.sort_values("median", ascending=True)

    sorted_methods = stats["clean_method"].tolist()

    fig, ax = plt.subplots(figsize=(18, 6))

    # Collect data and colors in the sorted order
    data_to_plot = []
    box_colors = []
    mean_markers = []
    xtick_labels = []

    for i, row in stats.iterrows():
        method = row["clean_method"]
        m_mean = row["mean"]
        m_std = row["std"]

        # Format the label to look exactly like the reference
        xtick_labels.append(f"{method}\n{m_mean:.2f} $\pm$ {m_std:.2f}")

        method_data = ranked_df[ranked_df["clean_method"] == method]["rank"].dropna().values
        data_to_plot.append(method_data)

        category = METHOD_CATEGORIES.get(method, "Filter")  # Default to filter if not found
        style = CATEGORY_STYLES[category]
        box_colors.append(style["color"])
        mean_markers.append(style["marker"])

    # 1. ADD JITTERED SCATTER POINTS (IN BACKGROUND)
    np.random.seed(42)
    for i, (method_data, color) in enumerate(zip(data_to_plot, box_colors)):
        # Make the scatter dots a slightly darker, highly transparent version of the box color
        # Or just match the exact light blue/orange with low alpha
        x = np.random.normal(i + 1, 0.06, size=len(method_data))
        ax.scatter(x, method_data, alpha=0.3, s=20, color=color, edgecolors='#555555', linewidths=0.5, zorder=1)

    # 2. CREATE TRANSPARENT BOXPLOTS
    bp = ax.boxplot(
        data_to_plot,
        patch_artist=True,
        showmeans=False,  # We will plot custom mean markers manually
        showfliers=False,  # Fliers are already drawn by the scatter
        medianprops=dict(linewidth=1.5, color='black'),
        boxprops=dict(linewidth=1.5, color='black'),
        whiskerprops=dict(linewidth=1.5, color='black'),
        capprops=dict(linewidth=1.5, color='black'),
        widths=0.4,
        zorder=2
    )

    # Apply transparency and color to the boxes
    for patch, color in zip(bp['boxes'], box_colors):
        # Convert hex to RGBA to make the facecolor transparent (alpha=0.6), but keep border solid
        import matplotlib.colors as mcolors
        rgba_color = mcolors.to_rgba(color, alpha=0.6)
        patch.set_facecolor(rgba_color)

    # 3. ADD CUSTOM MEAN MARKERS
    for i, (method_data, marker) in enumerate(zip(data_to_plot, mean_markers)):
        mean_val = np.mean(method_data)
        ax.plot(i + 1, mean_val, marker=marker, markersize=12, markerfacecolor="None",
                markeredgecolor="black", markeredgewidth=2, zorder=3)

    # 4. FORMATTING AND GRID
    ax.set_xticks(np.arange(1, len(sorted_methods) + 1))
    ax.set_xticklabels(xtick_labels, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel(Y_LABEL, fontsize=14)

    # Invert Y axis and set locator
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.invert_yaxis()

    # Subtle dashed grid lines on Y axis only
    ax.grid(True, axis='y', linestyle='--', alpha=0.7, color='#CCCCCC')
    ax.grid(False, axis='x')

    # 5. CUSTOM TOP LEGEND
    # Create proxy artists for the legend
    leg_non_search = mpatches.Patch(facecolor=mcolors.to_rgba('#C6D8EB', alpha=0.6), edgecolor='black', linewidth=1.5,
                                    label='Non-search-based')
    leg_search = mpatches.Patch(facecolor=mcolors.to_rgba('#FBE2C4', alpha=0.6), edgecolor='black', linewidth=1.5,
                                label='Search-based')

    leg_filter = mlines.Line2D([], [], color='none', marker='D', markersize=10, markeredgecolor='black',
                               markeredgewidth=1.5, label='Filter')
    leg_wrapper = mlines.Line2D([], [], color='none', marker='*', markersize=12, markeredgecolor='black',
                                markeredgewidth=1.5, label='Wrapper')
    leg_embedded = mlines.Line2D([], [], color='none', marker='X', markersize=10, markeredgecolor='black',
                                 markeredgewidth=1.5, label='Embedded')

    # Place legend above the plot
    ax.legend(handles=[leg_non_search, leg_search, leg_filter, leg_wrapper, leg_embedded],
              loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=5, fontsize=12, frameon=True,
              edgecolor='#AAAAAA', fancybox=True)

    plt.tight_layout()
    out = OUTPUT_DIR / PLOT_NAME
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Styled Boxplot saved to {out}")


# Do nothing below
SCRIPT_DIR = Path(__file__).parent / "../../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/performance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    plot_boxplot(df)


if __name__ == "__main__":
    main()