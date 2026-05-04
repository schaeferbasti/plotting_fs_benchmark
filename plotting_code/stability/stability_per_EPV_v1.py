import ast

import numpy as np
import pandas as pd
import sys

from pathlib import Path


# TODO: Adapt file and plot name
FILE_NAME = "stability_results.csv"
PLOT_NAME = "stability_per_EPV_v1"

# TODO: Adapt title and labels
PLOT_TITLE = ""
X_LABEL = "Selection Difficulty"
Y_LABEL = ""

# TODO: Adapt lasagna plot metric and smoothing
METRIC = "stability"
SMOOTHING = 10

# Do nothing below
SCRIPT_DIR = Path(__file__).parent / "../../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/stability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / PLOT_NAME

def overwrite_min_samples_from_validity(df_stability, df_validity, df_results):
    """
    Overwrites 'min_samples_per_class' in df_stability using the true values
    from df_validity, matched solely on 'data_foundry_task_id'.
    """
    df_stab = df_stability.copy()
    df_val = df_validity.copy()

    # 1. Filter validity to keep only one row per data_foundry_task_id
    val_dedup = df_val.drop_duplicates(subset=["data_foundry_task_id"], keep="first")

    # 2. Create a simple dictionary mapping: {task_id: true_min_samples}
    true_values_dict = val_dedup.set_index("data_foundry_task_id")["min_samples_per_class"].to_dict()

    # 3. Map these true values directly onto the stability dataframe using the task_id column
    new_min_samples = df_stab["data_foundry_task_id"].map(true_values_dict)

    # 4. Overwrite the column.
    # fillna ensures that if a task_id is somehow missing in validity,
    # it safely falls back to whatever value it originally had in stability.
    df_stab["min_samples_per_class"] = new_min_samples.fillna(df_stab["min_samples_per_class"])

    # 5. Extract the clean 'tid' from the 'data_foundry_task_id' string
    df_stab["temp_tid"] = df_stab["data_foundry_task_id"].apply(
        lambda x: int(str(x).split("|")[1]) if pd.notna(x) and "|" in str(x) else np.nan
    )

    # 6. Create a mapping dictionary for {tid: problem_type} from results_per_split.csv
    # We drop duplicates to ensure we have exactly one problem_type per tid
    res_dedup = df_results.dropna(subset=["tid", "problem_type"]).drop_duplicates(subset=["tid"], keep="first")
    problem_type_mapping = res_dedup.set_index("tid")["problem_type"].to_dict()

    # 7. Map the problem_type onto our stability dataframe
    df_stab["temp_problem_type"] = df_stab["temp_tid"].map(problem_type_mapping)

    # 8. Find all rows where the problem type is regression and force min_samples_per_class to NaN
    # We use .str.lower() to make the check case-insensitive (handles 'Regression' or 'regression')
    is_regression = df_stab["temp_problem_type"].str.lower() == "regression"
    df_stab.loc[is_regression, "min_samples_per_class"] = np.nan

    # 9. Clean up the temporary helper columns, so we return the dataframe in its original shape
    df_stab = df_stab.drop(columns=["temp_tid", "temp_problem_type"])

    return df_stab


def main():
    sys.path.append(str(SCRIPT_DIR.resolve()))
    from utils.metrics import compute_epv, compute_stability
    from utils.plots import lasagna_plot

    df_validity = pd.read_csv(SCRIPT_DIR / "result_files/validity_results.csv", low_memory=False)
    df_stability = pd.read_csv(RESULTS_FILE, low_memory=False)
    df_results = pd.read_csv(SCRIPT_DIR / "result_files/results_per_split.csv", low_memory=False)

    stab_datasets = set(df_stability["data_foundry_task_id"].dropna().unique())
    val_datasets = set(df_validity["data_foundry_task_id"].dropna().unique())

    # Find the difference (in stability, but NOT in validity)
    missing_in_validity = stab_datasets - val_datasets

    print(f"Total datasets in stability: {len(stab_datasets)}")
    print(f"Total datasets in validity: {len(val_datasets)}")
    print(f"Datasets in stability but NOT in validity ({len(missing_in_validity)} total):")

    df = overwrite_min_samples_from_validity(df_stability, df_validity, df_results)

    unique_counts = df.groupby("data_foundry_task_id")["min_samples_per_class"].nunique(dropna=False)

    # Filter for datasets that have more than 1 unique value
    inconsistent_datasets = unique_counts[unique_counts > 1]

    if inconsistent_datasets.empty:
        print("✅ All datasets have exactly one consistent 'min_samples_per_class' value.")
    else:
        print(f"⚠️ Found {len(inconsistent_datasets)} dataset(s) with varying 'min_samples_per_class' values:")
        for ds_id in inconsistent_datasets.index:
            # Extract the exact unique values for this specific dataset to see the conflict
            conflicting_vals = df[df["data_foundry_task_id"] == ds_id]["min_samples_per_class"].unique()
            print(f"  - Dataset {ds_id}: {conflicting_vals}")

    df["selected_features_parsed"] = df["selected_features"].apply(ast.literal_eval)
    df["original_features_parsed"] = df["original_features"].apply(ast.literal_eval)

    # TODO: check if pattern holds for non-dummy data
    df["selector"] = df["method"].str.split("__").str[1]
    df["dataset"] = df["data_foundry_task_id"].str.split("|").str[2].str.split("/").str[0]

    # TODO: check if min_samples_per_class is nan for regression and if num_samples is for the whole dataset or per split
    df["epv"] = compute_epv(df, df["original_features_parsed"])

    unique_epv_counts = df.groupby("data_foundry_task_id")["epv"].nunique(dropna=False)
    inconsistent_datasets = unique_epv_counts[unique_epv_counts > 1]

    if not inconsistent_datasets.empty:
        print(f"⚠️ Found {len(inconsistent_datasets)} dataset(s) with fluctuating EPV values. Fixing them now...")
    else:
        print("✅ All datasets have consistent EPV values.")

    # TODO: adapt if stability estimation is not over all repeats
    df_plot = (
        df.groupby(["dataset", "selector", "epv", "max_features"])
        .apply(
            lambda g: compute_stability(
                g["selected_features_parsed"],
                g["original_features_parsed"],
                method_name=g["selector"].iloc[0]
            )
        )
        .reset_index(name="stability")
    )

    df_plot = (
        df_plot.groupby(["selector", "epv"], as_index=False)["stability"]
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
                overlay=overlay,
                mode="stability",
                sort_methods="mean_value"
            )


if __name__ == "__main__":
    main()
