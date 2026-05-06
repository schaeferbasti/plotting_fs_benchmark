import ast
from pathlib import Path

import pandas as pd


def safe_parse_check(x):
    if pd.isna(x):
        return False
    x_str = str(x).strip().strip('"').strip("'")

    try:
        parsed = ast.literal_eval(x_str)
        return isinstance(parsed, dict)
    except Exception:
        return False


def check_validity_completeness(csv_path):
    df = pd.read_csv(csv_path)

    original_count = len(df)

    mask1 = df['method'].astype(str).str.startswith("FSBench", na=False)
    mask2 = df['mode_kwargs'].apply(safe_parse_check)

    final_valid_mask = mask1 & mask2
    wrong_entries_count = (~final_valid_mask).sum()

    df = df[final_valid_mask].copy()

    df['mode_kwargs_dict'] = df['mode_kwargs'].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else {}
    )

    df['noise_val'] = df['mode_kwargs_dict'].apply(lambda d: d.get('noise'))

    group_cols = ['method', 'data_foundry_task_id', 'repeat']
    grouped = df.groupby(group_cols)

    expected_noise = {0.5, 0.75, 1.0}

    missing_tallies = {0.5: 0, 0.75: 0, 1.0: 0}
    incomplete_combinations = []

    for combination, group in grouped:
        existing_noise = set(group['noise_val'].dropna().unique())

        missing_noise = expected_noise - existing_noise

        if missing_noise:
            for val in missing_noise:
                missing_tallies[val] += 1

            incomplete_combinations.append({
                'method': combination[0],
                'task_id': combination[1],
                'repeat': combination[2],
                'existing_noise': list(existing_noise),
                'missing_noise': list(missing_noise)
            })

    if not incomplete_combinations:
        print("✅ All combinations are complete! Every group has noise = 0.5, 0.75, and 1.0.")
    else:
        for item in incomplete_combinations:
            print(f"Method: {item['method']} | Task ID: {item['task_id']} | Repeat: {item['repeat']}")
            print(f"  -> Missing noise levels: {item['missing_noise']}")
            print(f"  -> Existing noise levels: {item['existing_noise']}\n")

        print(f"⚠️ Found {len(incomplete_combinations)} incomplete combinations out of {len(grouped)}.\n")

        print(f"Total original rows: {original_count}")
        print(f"❌ Removed {wrong_entries_count} wrong entries (method didn't start with 'FSBench').")
        print(f"✅ Remaining valid rows: {len(df)}")

        print("📊 Global Missing Tallies:")
        print(f"  - Runs missing noise=0.5 : {missing_tallies[0.5]}")
        print(f"  - Runs missing noise=0.75: {missing_tallies[0.75]}")
        print(f"  - Runs missing noise=1.0 : {missing_tallies[1.0]}")


def check_stability_completeness(csv_path):
    df = pd.read_csv(csv_path)
    df['mode_kwargs'] = df['mode_kwargs'].astype(str)

    group_cols = ['method', 'data_foundry_task_id', 'repeat']
    grouped = df.groupby(group_cols)
    incomplete_combinations = []

    complete_100_count = 0

    for combination, group in grouped:
        existing_kwargs = group['mode_kwargs'].unique().tolist()

        if len(existing_kwargs) != 100:
            incomplete_combinations.append({
                'method': combination[0],
                'task_id': combination[1],
                'repeat': combination[2],
                'count': len(existing_kwargs),
            })
        else:
            complete_100_count += 1

    # Output the results
    if not incomplete_combinations:
        print("✅ All combinations are complete! Every group has exactly 100 unique bootstrapped entries.")
    else:
        for item in incomplete_combinations:
            print(f"Method: {item['method']} | Task ID: {item['task_id']} | Repeat: {item['repeat']}")
            print(f"  -> Expected 100 entries, but found {item['count']}.")
        print(f"⚠️ Found {len(incomplete_combinations)} incomplete combinations out of {len(grouped)}\n")
    print(f"Found {complete_100_count} method-dataset-repeat combinations that have EXACTLY 100 entries.")


def calculate_total_runtime(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)

    if "elapsed_time_fs" in df.columns:
        runtime_col = "elapsed_time_fs"
    elif "time_train_s" in df.columns:
        runtime_col = "time_train_s"
    else:
        raise ValueError(
            "Neither 'elapsed_time_fs' nor 'feature_selection_fit_time' found in the CSV."
        )

    runtime = pd.to_numeric(df[runtime_col], errors="coerce").fillna(0)
    return runtime.sum()


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    validity_file = script_dir / "validity_results.csv"
    validity_time = calculate_total_runtime(validity_file)
    check_validity_completeness(validity_file)

    stability_file = script_dir / "stability_results.csv"
    stability_time = calculate_total_runtime(stability_file)
    #check_stability_completeness(stability_file)


    performance_file = script_dir / "results_per_split.csv"
    performance_time = calculate_total_runtime(performance_file)

    print(f"Total runtime for validity results: {validity_time:.2f} seconds")
    print(f"Total runtime for stability results: {stability_time:.2f} seconds")
    print(f"Total runtime for performance results: {performance_time:.2f} seconds")
    total_time = validity_time + stability_time + performance_time
    print(f"Total runtime across all files: {total_time:.2f} seconds")
    total_time_hours = total_time / 3600
    print(f"Total runtime across all files: {total_time_hours:.2f} hours")
    total_time_days = total_time_hours / 24
    print(f"Total runtime across all files: {total_time_days:.2f} days")
    total_time_years = total_time_days / 365
    print(f"Total runtime across all files: {total_time_years:.2f} years")

