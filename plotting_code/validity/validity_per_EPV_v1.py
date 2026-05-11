import ast
import pandas as pd
import sys

from pathlib import Path


# TODO: Adapt file and plot name
FILE_NAME = "validity_results.csv"
PLOT_NAME = "validity_per_EPV_v1"

# TODO: Adapt title and labels
PLOT_TITLE = ""
X_LABEL = "Events per Variable"
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


def diagnose_validity_data(df_validity):
    """
    Comprehensive missing data analysis for validity dataframe.
    Pass your df_validity directly - no file loading needed.
    """
    print("\n" + "=" * 60)
    print("🔍 VALIDITY DATA DIAGNOSTICS")
    print("=" * 60)

    # Extract method names
    df_validity = df_validity.copy()

    # Handle potentially missing method column safely
    df_validity['selector'] = df_validity['method'].str.split("__").str[1]

    # 1. BASIC SHAPE
    print(f"📊 Shape: {df_validity.shape[0]:,} rows, {df_validity.shape[1]} cols")
    print(f"📋 Supposed Shape: {71*15*5*3}, rows, {df_validity.shape[1]} cols\n")

    if 'extracted_noise_level' not in df_validity.columns:
        df_validity['extracted_noise_level'] = df_validity['mode_kwargs'].apply(extract_noise)

    combo_noise_counts = df_validity.groupby(
        ['data_foundry_task_id', 'selector']
    )['extracted_noise_level'].nunique().reset_index(name='completed_noise_levels')

    # 2. Filter for combinations that have exactly the number you are looking for
    # For example: 2 out of 3, or 4 out of 5
    target_completed = 1  # Change this to whatever number you are looking for (e.g., 4)

    matches = combo_noise_counts[combo_noise_counts['completed_noise_levels'] == target_completed]

    print(f"\n🎯 Found {len(matches)} dataset/method combinations with exactly {target_completed} noise levels:")

    if not matches.empty:
        # Print the first 15 as an example
        print(matches.head(15).to_string(index=False))

        # If you want to see which methods are most prone to finishing "almost" all budgets
        print("\nBreakdown of these 'almost complete' combinations per method:")
        print(matches['selector'].value_counts().to_string())
    else:
        print("None found.")

    print("=" * 60)



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

    diagnose_validity_data(df)

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

        for binary_mode in ["threshold", "topk"]:
            for overlay in [True, False]:
                lasagna_plot(
                    df_noise,  # Pass only the chunk for this noise level
                    values=METRIC,
                    plot_title=PLOT_TITLE,
                    x_label=X_LABEL,
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
