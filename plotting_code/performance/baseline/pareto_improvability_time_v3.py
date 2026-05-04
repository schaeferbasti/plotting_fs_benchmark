from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.beautify import remove_jmi, beautify_names, add_model_name

# TODO: Adapt file and plot name
FILE_NAME = "results_per_split.csv"
PLOT_NAME = "pareto_improvability_time_v3.png"

# TODO: Adapt title and labels
PLOT_TITLE = ""
X_LABEL = "Feature Selection Time (s) per 1k samples"
Y_LABEL = "Improvement over Random (%)"


def add_num_samples_from_validity(df_performance, df_validity):
    df_performance = df_performance.copy()
    df_val = df_validity.copy()


    # 2. Extract the clean 'tid' from the long 'data_foundry_task_id' string
    df_val["tid"] = df_val["data_foundry_task_id"].apply(
        lambda x: int(str(x).split("|")[1]) if pd.notna(x) and "|" in str(x) else np.nan
    )

    # 3. Create a clean mapping dictionary: {tid: dataset_size}
    val_dedup = df_val.dropna(subset=["tid", "num_samples"]).drop_duplicates(subset=["tid"], keep="first")
    size_mapping = val_dedup.set_index("tid")["num_samples"].to_dict()

    # 4. Map the dataset sizes onto the performance dataframe
    df_performance["tid"] = df_performance["tid"].astype(int)
    df_performance["num_samples"] = df_performance["tid"].map(size_mapping)

    # (Optional) Print a warning if any datasets couldn't be matched
    missing_sizes = df_performance["num_samples"].isna().sum()
    if missing_sizes > 0:
        print(f"⚠️ Warning: Could not find original number of samples for {missing_sizes} rows.")

    return df_performance

def calculate_relative_performance(df):
    df = df.copy()

    df = beautify_names(df)
    df = remove_jmi(df)
    df = add_model_name(df)

    df_val = pd.read_csv(SCRIPT_DIR / "result_files" / "validity_results.csv", low_memory=False)
    df = add_num_samples_from_validity(df, df_val)

    df["time_per_1k"] = (df["feature_selection_fit_time"] / df["num_samples"]) * 1000

    # 1. Average budget, models (and splits)
    df_avg = (
        df.groupby(["tid", "feature_selection_method"])[["metric_error", "time_per_1k"]]
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

    # 4. Improvability = how much error remains until best, relative to own error
    df_merged["improvability"] = (
        (df_merged["random_error"] - df_merged["metric_error"])
        / (df_merged["random_error"])
    ) * 100

    # Final aggregation across all datasets
    agg_df = df_merged.groupby("feature_selection_method").agg(
        mean_score=("improvability", "mean"),
        mean_time=("time_per_1k", "mean")
    ).reset_index()

    # Drop 'Random' from the final aggregated dataframe so it isn't plotted
    agg_df = agg_df[agg_df["feature_selection_method"] != "LaplacianScore"].reset_index(drop=True)
    # agg_df = agg_df[agg_df["feature_selection_method"] != "Random"].reset_index(drop=True)


    return agg_df


def plot(df):
    agg_df = calculate_relative_performance(df)

    # Extract values as plain numpy arrays to avoid any pandas index bugs
    times = agg_df["mean_time"].to_numpy()
    scores = agg_df["mean_score"].to_numpy()

    pareto_idx_list = []
    n_points = len(agg_df)

    for i in range(n_points):
        dominated = False
        for j in range(n_points):
            if i == j:
                continue

            # To DOMINATE, point j must be at least as good in BOTH,
            # and strictly better in AT LEAST ONE.
            # Assuming MINIMIZE Time (Lower is Better)
            # Assuming MINIMIZE Score (Lower is Better - e.g. Error)
            # --> IF SCORE IS ACCURACY (Higher is Better), FLIP the score signs below!

            time_j_better_or_eq = times[j] <= times[i]
            score_j_better_or_eq = scores[j] >= scores[i]

            time_j_strictly_better = times[j] < times[i]
            score_j_strictly_better = scores[j] > scores[i]

            if (time_j_better_or_eq and score_j_better_or_eq) and (time_j_strictly_better or score_j_strictly_better):
                dominated = True
                break  # Point i is dominated by point j, stop checking

        if not dominated:
            pareto_idx_list.append(i)

    # -------------------------------------------------------------
    # 2. EXTRACT DATAFRAMES
    # -------------------------------------------------------------
    # We use .iloc because pareto_idx_list contains absolute integer positions (0, 1, 2...)
    pareto = agg_df.iloc[pareto_idx_list].sort_values("mean_time", ascending=False)

    # Non-pareto is everything else
    non_pareto = agg_df.drop(agg_df.index[pareto_idx_list])

    # Now you can print to debug!
    print(f"Total methods: {len(agg_df)}")
    print(f"Pareto methods: {len(pareto)}")
    print(f"Dominated methods: {len(non_pareto)}")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_yscale("linear")  # or "log" if needed
    ax.set_xscale("log")  # or "log" if needed

    # 1. Plot Non-Pareto methods in gray (Background)
    ax.scatter(
        non_pareto["mean_time"],
        non_pareto["mean_score"],
        s=60,
        color="lightgray",
        edgecolor="gray",
        alpha=0.8,
        label="Dominated Methods",
        zorder=1
    )

    # 2. Highlight Pareto front points in Red (Foreground)
    ax.scatter(
        pareto["mean_time"],
        pareto["mean_score"],
        s=80,
        color="tab:red",
        edgecolors="black",  # <--- Add this!
        label="Pareto Front",
        zorder=3,
    )

    # 3. Draw line connecting Pareto points
    ax.plot(
        pareto["mean_time"],
        pareto["mean_score"],
        color="tab:red",
        linewidth=2,
        zorder=2,
    )

    # 4. Label ALL points, shifting labels to prevent overlap
    labeled_positions = []

    for idx, row in agg_df.iterrows():
        method_name = row["feature_selection_method"].replace("FeatureSelector", "")
        x, y = row["mean_time"], row["mean_score"]

        # Shift overlapping labels
        y_offset = 5
        for (lx, ly) in labeled_positions:
            if abs(x - lx) < (agg_df["mean_time"].max() * 0.05) and abs(y - ly) < (agg_df["mean_score"].max() * 0.05):
                y_offset -= 12

        labeled_positions.append((x, y))

    ax.annotate(
        "ideal",
        xy=(0.0, 1.0),  # arrow tip near top-left
        xytext=(0.03, 0.96),  # text a bit inside the plot
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->",
            color="green",
            lw=1.5
        ),
        fontsize=14,
        fontweight="bold",
        color="green",
        ha="left",
        va="top"
    )

    # 4. Label ALL points automatically using adjustText
    from adjustText import adjust_text

    texts = []

    for idx, row in agg_df.iterrows():
        method_name = row["feature_selection_method"].replace("FeatureSelector", "")
        x, y = row["mean_time"], row["mean_score"]

        # 1. Add center alignment (ha='center', va='center') so arrows stop at the edge
        if idx in pareto_idx_list:
            t = ax.text(
                x, y, method_name,
                fontsize=14, weight='bold', color="black",
                ha='center', va='center'
            )
        else:
            t = ax.text(
                x, y, method_name,
                fontsize=14, color="dimgray",
                ha='center', va='center'
            )

        texts.append(t)

    # 2. Adjust texts
    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color='black', lw=1, alpha=0.7),
        expand=(1.3, 1.3),
        min_arrow_len=9, zorder=3
    )

    ax.set_title(PLOT_TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)

    ax.legend()
    ax.grid(True, alpha=0.3)
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