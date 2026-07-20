from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.beautify import remove_jmi, beautify_names, add_model_name

# TODO: Adapt file and plot name
FILE_NAME = "results_per_split.csv"
PLOT_NAME = "pareto_improvability_time_budget_v1.pdf"

# TODO: Adapt title and labels
PLOT_TITLE = ""
X_LABEL = "Feature Selection Time (s) per 1k Features"
Y_LABEL = "Improvement over Random (%)"

import ast


def add_num_features_from_validity(df_performance, df_validity):
    df_performance = df_performance.copy()
    df_val = df_validity.copy()

    # 2. Extract the clean 'tid'
    df_val["tid"] = df_val["data_foundry_task_id"].apply(
        lambda x: int(str(x).split("|")[1]) if pd.notna(x) and "|" in str(x) else np.nan
    )

    # 3. Create a clean mapping dictionary
    val_dedup = df_val.dropna(subset=["tid", "original_features"]).drop_duplicates(subset=["tid"], keep="first")

    # NEW: Safely evaluate the string to a list, then get the length
    def get_list_length(val):
        if isinstance(val, str):
            try:
                # Converts "[1, 2, 3]" to a real python list [1, 2, 3]
                parsed_list = ast.literal_eval(val)
                return len(parsed_list)
            except (ValueError, SyntaxError):
                # Fallback if it fails to parse
                return np.nan
        elif isinstance(val, list):
            # Just in case it is already a list
            return len(val)
        return np.nan

    val_dedup["num_features"] = val_dedup["original_features"].apply(get_list_length)

    # Create mapping, dropping any NaNs where parsing failed
    val_dedup = val_dedup.dropna(subset=["num_features"])
    size_mapping = val_dedup.set_index("tid")["num_features"].to_dict()

    # 4. Map the dataset sizes onto the performance dataframe
    df_performance["tid"] = df_performance["tid"].astype(int)
    df_performance["num_features"] = df_performance["tid"].map(size_mapping)

    # (Optional) Print a warning if any datasets couldn't be matched
    missing_sizes = df_performance["num_features"].isna().sum()
    if missing_sizes > 0:
        print(f"⚠️ Warning: Could not find original number of features for {missing_sizes} rows.")

    return df_performance


def calculate_relative_performance(df):
    df = df.copy()

    df = beautify_names(df)
    df = remove_jmi(df)
    df = add_model_name(df)

    df_val = pd.read_csv(SCRIPT_DIR / "result_files" / "validity_results.csv", low_memory=False)
    df = add_num_features_from_validity(df, df_val)

    df["time_per_1k"] = (df["feature_selection_fit_time"] / df["num_features"]) * 1000

    df_avg = (
        df.groupby(["tid", "feature_selection_method", "feature_selection_budget_index"])[["metric_error", "time_per_1k"]]
        .mean()
        .reset_index()
    )

    # 2. Best error per dataset
    random_errors = df_avg[df_avg["feature_selection_method"] == "Random"].copy()

    random_baseline = (
        random_errors.groupby("tid")["metric_error"]
        .mean()
        .reset_index()
        .rename(columns={"metric_error": "random_error"})
    )

    # 3. Merge best error back
    df_merged = df_avg.merge(random_baseline, on="tid", how="left")

    # 4. Improvability
    df_merged["improvability"] = (
                                         (df_merged["random_error"] - df_merged["metric_error"])
                                         / (df_merged["random_error"])
                                 ) * 100

    # Final aggregation across all datasets, NOW GROUPED BY METHOD AND BUDGET
    agg_df = df_merged.groupby(["feature_selection_method", "feature_selection_budget_index"]).agg(
        mean_score=("improvability", "mean"),
        mean_time=("time_per_1k", "mean")
    ).reset_index()

    # Drop specific methods
    agg_df = agg_df[agg_df["feature_selection_method"] != "LaplacianScore"].reset_index(drop=True)

    agg_df.to_csv(OUTPUT_DIR / "pareto_data_per_budget.csv", index=False)

    return agg_df


import matplotlib.cm as cm
from adjustText import adjust_text


def plot(df):
    agg_df = calculate_relative_performance(df)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_yscale("linear")
    ax.set_xscale("log")

    # Define budgets and colors
    budgets = sorted(agg_df["feature_selection_budget_index"].unique())
    colors = ["#0072B2", "#009E73", "#F0E442", "#E69F00", "#D55E00"]

    texts = []

    # Process each budget independently
    for idx_budget, budget in enumerate(budgets):
        budget_df = agg_df[agg_df["feature_selection_budget_index"] == budget].reset_index(drop=True)
        color = colors[idx_budget]

        times = budget_df["mean_time"].to_numpy()
        scores = budget_df["mean_score"].to_numpy()

        pareto_idx_list = []
        n_points = len(budget_df)

        # Calculate Pareto Front for THIS budget
        for i in range(n_points):
            dominated = False
            for j in range(n_points):
                if i == j:
                    continue

                time_j_better_or_eq = times[j] <= times[i]
                score_j_better_or_eq = scores[j] >= scores[i]  # Assuming Score is Improvability (Higher is better)

                time_j_strictly_better = times[j] < times[i]
                score_j_strictly_better = scores[j] > scores[i]

                if (time_j_better_or_eq and score_j_better_or_eq) and (
                        time_j_strictly_better or score_j_strictly_better):
                    dominated = True
                    break

            if not dominated:
                pareto_idx_list.append(i)

        # Separate Pareto and Non-Pareto for THIS budget
        pareto = budget_df.iloc[pareto_idx_list].sort_values("mean_time", ascending=False)
        non_pareto = budget_df.drop(budget_df.index[pareto_idx_list])

        # 1. Plot Dominated points
        ax.scatter(
            non_pareto["mean_time"],
            non_pareto["mean_score"],
            s=40,
            color=color,
            alpha=0.3,
            edgecolor="white",
            zorder=1
        )

        # 2. Plot Pareto points
        ax.scatter(
            pareto["mean_time"],
            pareto["mean_score"],
            s=80,
            color=color,
            edgecolors="black",
            label=f"Budget {budget}",
            zorder=3,
        )

        # 3. Draw connecting line for Pareto front
        ax.plot(
            pareto["mean_time"],
            pareto["mean_score"],
            color=color,
            linewidth=2,
            zorder=2,
        )

        # 4. Prepare text labels for THIS budget
        for idx, row in budget_df.iterrows():
            method_name = row["feature_selection_method"].replace("FeatureSelector", "")
            x, y = row["mean_time"], row["mean_score"]

            if idx in pareto_idx_list:
                t = ax.text(x, y, method_name, fontsize=10, weight='bold', color=color, ha='center', va='center')
            else:
                t = ax.text(x, y, method_name, fontsize=8, color="dimgray", alpha=0.6, ha='center', va='center')
            texts.append(t)

    # Adjust all texts together to prevent overlap globally
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color='black', lw=0.5, alpha=0.5),
        expand=(1.5, 1.5),
        min_arrow_len=9, zorder=3
    )

    ax.set_title(PLOT_TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)

    ax.legend(title="Budgets", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    for side in ['left', 'bottom', 'right', 'top']:
        ax.spines[side].set_color("black")
        ax.spines[side].set_alpha(0.3)

    plt.tight_layout()

    out = OUTPUT_DIR / PLOT_NAME
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Pareto plot saved to {out}")


# Do nothing below
SCRIPT_DIR = Path(__file__).parent / "../../../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/performance/baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    plot(df)


if __name__ == "__main__":
    main()