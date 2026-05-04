import ast
import pandas as pd
import sys

from pathlib import Path


# TODO: Adapt file and plot name
FILE_NAME = "validity_results.csv"
PLOT_NAME = "validity_per_EPV_v1"

# TODO: Adapt title and labels
PLOT_TITLE = ""
X_LABEL = "Selection Difficulty (EPV)"
Y_LABEL = ""

# TODO: Adapt lasagna plot metric and smoothing
METRIC = "validity"
SMOOTHING = 10

# Do nothing below
SCRIPT_DIR = Path(__file__).parent / "../../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/validity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / PLOT_NAME


def extract_noise(val):
    try:
        # If it is a string like "{'noise': 0.1, ...}", parse it
        if isinstance(val, str):
            parsed_dict = ast.literal_eval(val)
        elif isinstance(val, dict):
            parsed_dict = val
        else:
            return "unknown"
        return parsed_dict.get("noise", "unknown")
    except (ValueError, SyntaxError, AttributeError):
        return "unknown"

def main():
    
    sys.path.append(str(SCRIPT_DIR.resolve()))
    from utils.metrics import compute_epv, compute_validity
    from utils.plots import lasagna_plot
    
    # TODO: adapt if not all rows, but only rows where repeat == 0
    df = pd.read_csv(RESULTS_FILE, low_memory=False)

    df = df[df["method"].astype(str).str.startswith("FSBench")]

    df["selected_features_parsed"] = df["selected_features"].apply(ast.literal_eval)
    df["original_features_parsed"] = df["original_features"].apply(ast.literal_eval)

    df["validity"] = compute_validity(df["selected_features_parsed"], df["max_features"])

    # TODO: check if pattern holds for non-dummy data
    df["selector"] = df["method"].str.split("__").str[1]
    df["dataset"] = df["data_foundry_task_id"].str.split("|").str[2].str.split("/").str[0]

    # TODO: check if min_samples_per_class is nan for regression and if num_samples is for the whole dataset or per split
    df["epv"] = compute_epv(df, df["original_features_parsed"])

    df["noise_level"] = df["mode_kwargs"].apply(extract_noise)

    # mean selection precision over max_features (cardinality)
    df_plot = (
        df.groupby(["noise_level", "selector", "epv"], as_index=False)["validity"]
        .mean()
    )

    for noise_level, df_noise in df_plot.groupby("noise_level"):

        # Format the filename so floats like 0.1 become "0_1"
        noise_str = str(noise_level).replace(".", "_")
        noise_output_path = OUTPUT_DIR / f"{PLOT_NAME}_noise_{noise_str}"
        noise_label = f"{X_LABEL} - Noise: {noise_level}"

        for binary_mode in ["threshold", "topk"]:
            for overlay in [True, False]:
                lasagna_plot(
                    df_noise,  # Pass only the chunk for this noise level
                    values=METRIC,
                    plot_title=PLOT_TITLE,
                    x_label=noise_label,
                    y_label=Y_LABEL,
                    output_path=noise_output_path,
                    smoothing=SMOOTHING,
                    binary_mode=binary_mode,
                    overlay=overlay,
                    mode="validity",
                    sort_methods="mean_value"
                )


if __name__ == "__main__":
    main()
