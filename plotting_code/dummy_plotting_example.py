from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt

# TODO: Adapt file and plot name
FILE_NAME = "dummy_performance_results.csv"
PLOT_NAME = "dummy_plot.png"

# TODO: Adapt title and labels
PLOT_TITLE = "Dummy Plot"
X_LABEL = "Feature Selection Method"
Y_LABEL = "Metric Error"

# TODO: Adapt plotting function
def plot(df):
    methods = sorted(df["feature_selection_method"].dropna().unique())
    data = [
        df.loc[df["feature_selection_method"] == method, "metric_error"].dropna().values
        for method in methods
    ]
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.boxplot(
        data,
        tick_labels=methods,
        showmeans=True,
        patch_artist=True
    )

    ax.set_title(PLOT_TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    out = OUTPUT_DIR / PLOT_NAME
    plt.savefig(out)
    plt.close()


# Do nothing below
SCRIPT_DIR = Path(__file__).parent / "../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    plot(df)


if __name__ == "__main__":
    main()
