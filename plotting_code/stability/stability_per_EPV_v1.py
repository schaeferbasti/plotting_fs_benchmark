import ast
import pandas as pd
import sys

from pathlib import Path


# TODO: Adapt file and plot name
FILE_NAME = "extended_dummy_stability_results.csv"
PLOT_NAME = "stability_per_EPV_v1"

# TODO: Adapt title and labels
PLOT_TITLE = "Stability vs. Selection Difficulty"
X_LABEL = "Selection Difficulty"
Y_LABEL = ""

# TODO: Adapt lasagna plot metric and smoothing
METRIC = "stability"
SMOOTHING = 20

# Do nothing below
SCRIPT_DIR = Path(__file__).parent / "../../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/stability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / PLOT_NAME


def main():
    
    sys.path.append(str(SCRIPT_DIR.resolve()))
    from utils.metrics import compute_epv, compute_stability
    from utils.plots import lasagna_plot

    df = pd.read_csv(RESULTS_FILE, low_memory=False)

    df["selected_features_parsed"] = df["selected_features"].apply(ast.literal_eval)
    df["original_features_parsed"] = df["original_features"].apply(ast.literal_eval)

    # TODO: check if pattern holds for non-dummy data
    df["selector"] = df["method"].str.split("__").str[1]
    df["dataset"] = df["data_foundry_task_id"].str.split("|").str[2].str.split("/").str[0]

    # TODO: check if min_samples_per_class is nan for regression and if num_samples is for the whole dataset or per split
    df["epv"] = compute_epv(df, df["original_features_parsed"])

    # TODO: adapt if stability estimation is not over all repeats
    df_plot = (
        df.groupby(["dataset", "selector", "epv", "max_features"])
        .apply(
            lambda g: compute_stability(
                g["selected_features_parsed"],
                g["original_features_parsed"]
            )
        )
        .reset_index(name="stability")
    )

    # mean stability over max_features (cardinality)
    df_plot = (
        df_plot.groupby(["dataset", "selector", "epv"], as_index=False)["stability"]
        .mean()
    )

    for binary_mode in ["threshold", "topk"]:
        for overlay in [True, False]:
            lasagna_plot(
                df_plot,
                values=METRIC,
                plot_title=PLOT_TITLE,
                x_label=X_LABEL,
                y_label=Y_LABEL,
                output_path=OUTPUT_PATH,
                smoothing=SMOOTHING,
                binary_mode=binary_mode,
                overlay=overlay
            )


if __name__ == "__main__":
    main()
