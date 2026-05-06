from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
from autorank import autorank, plot_stats, create_report

from utils.average import average_per_dataset_model_budget_and_method
from utils.beautify import beautify_names, remove_jmi, add_model_name

FILE_NAME = "results_per_split.csv"
PLOT_NAME = "critical_difference_v3.pdf"
PLOT_TITLE = ""

def prepare_data_for_autorank(df):
    """
    Transforms the long-format dataframe into the wide-format block design
    required by Autorank: Rows = Datasets, Columns = Feature Selection Methods.
    """

    df = beautify_names(df)
    df = remove_jmi(df)
    df = add_model_name(df)

    df = average_per_dataset_model_budget_and_method(df)

    df = df.pivot(
        index="tid_model_budget",
        columns="feature_selection_method",
        values="metric_error"
    )

    return df


def plot_autorank(df):
    # 1. Prepare data
    data = prepare_data_for_autorank(df)

    # 2. Run statistical analysis
    result = autorank(data, alpha=0.05, verbose=False, order='ascending')
    create_report(result)

    # 3. Create the figure with desired dimensions: (width, height)
    # Using (4, 8) makes it narrow (4 inches) and tall (8 inches)
    fig, ax = plt.subplots(figsize=(3, 12))

    # 4. Plot stats into the pre-defined ax
    # Autorank will draw the plot inside this specific axis
    plot_stats(result, ax=ax)

    for text in ax.texts:
        if text.get_text() not in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14",
                                   "15"]:
            x, y = text.get_position()
            text.set_position((x - 0, y))
            text.set_fontsize(20)  # Augment size here

    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin + 0.15, xmax - 0.15)

    # Add title and layout adjustments
    ax.set_title(PLOT_TITLE, pad=20, fontsize=14, weight='bold')

    out = OUTPUT_DIR / PLOT_NAME
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Autorank CD plot saved to {out}")


SCRIPT_DIR = Path(__file__).parent / "../../../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/performance/cd"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    plot_autorank(df)


if __name__ == "__main__":
    main()