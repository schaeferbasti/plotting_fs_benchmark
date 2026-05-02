import ast
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from utils.average import average_per_method, average_per_dataset_and_method
from utils.beautify import remove_jmi, beautify_names, add_model_name

# TODO: Adapt file and plot name
FILE_NAME = "results_per_split.csv"
PLOT_NAME = "pareto_rank_time_v1.png"

# TODO: Adapt title and labels
PLOT_TITLE = "Pareto Front: Error vs. Training Time"
X_LABEL = "Mean Training Time (s)"
Y_LABEL = "Mean Rank (Lower is Better)"


def calculate_ranks(df):
    df = df.copy()
    print(df["feature_selection_fit_time"].isna().sum())

    df = beautify_names(df)
    df = remove_jmi(df)
    df = add_model_name(df)

    # RANKING PHASE
    df["rank"] = df.groupby(["tid", "metric"])["metric_error"].rank(
        method="average",
        ascending=True,
        na_option="keep"
    )

    # Get one overall mean score and mean time per method for the Pareto front
    agg_df = df.groupby("feature_selection_method").agg(
        mean_score=("rank", "mean"),
        mean_time=("feature_selection_fit_time", "mean")
    ).reset_index()

    return agg_df


def plot(df):
    agg_df = calculate_ranks(df)

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
            score_j_better_or_eq = scores[j] <= scores[i]

            time_j_strictly_better = times[j] < times[i]
            score_j_strictly_better = scores[j] < scores[i]

            if (time_j_better_or_eq and score_j_better_or_eq) and (time_j_strictly_better or score_j_strictly_better):
                dominated = True
                break  # Point i is dominated by point j, stop checking

        if not dominated:
            pareto_idx_list.append(i)

    # -------------------------------------------------------------
    # 2. EXTRACT DATAFRAMES
    # -------------------------------------------------------------
    # We use .iloc because pareto_idx_list contains absolute integer positions (0, 1, 2...)
    pareto = agg_df.iloc[pareto_idx_list].sort_values("mean_time")

    # Non-pareto is everything else
    non_pareto = agg_df.drop(agg_df.index[pareto_idx_list])

    # Now you can print to debug!
    print(f"Total methods: {len(agg_df)}")
    print(f"Pareto methods: {len(pareto)}")
    print(f"Dominated methods: {len(non_pareto)}")

    fig, ax = plt.subplots(figsize=(12, 6))

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

        # Check if the current row index is in the Pareto index list
        if idx in pareto_idx_list:
            ax.annotate(
                method_name,
                (x, y),
                textcoords="offset points",
                xytext=(5, y_offset),
                fontsize=9,
                weight='bold',
                color="black"
            )
        else:
            ax.annotate(
                method_name,
                (x, y),
                textcoords="offset points",
                xytext=(5, y_offset),
                fontsize=8,
                color="dimgray"
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
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/performance/ranking"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    plot(df)


if __name__ == "__main__":
    main()