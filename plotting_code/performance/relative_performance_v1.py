import ast
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

FILE_NAME = "results_per_split.csv"
PLOT_NAME = "relative_performance_v1.png"
PLOT_TITLE = "Performance Relative to Random Baseline (mix-max-scaling, avg across budgets & models)"
X_LABEL = "Feature Selection Method"
Y_LABEL = "Relative Score (Percentage points better/worse than Random)"

METRIC_DIRECTIONS = {
    "log_loss": True,  # Lower error is better
    "rmse": True,  # Lower error is better
    "roc_auc": False,  # Higher performance is better
}


def calculate_relative_performance(df):
    df = df.copy()

    # --- DATA CLEANING ---
    if "feature_selection_method" in df.columns:
        # Strip "FeatureSelector" from all methods except the Random one,
        # because we need exactly "RandomFeatureSelector" for the baseline extraction later.
        mask = df["feature_selection_method"] != "RandomFeatureSelector"
        df.loc[mask, "feature_selection_method"] = df.loc[mask, "feature_selection_method"].str.replace(
            "FeatureSelector", "", regex=False)

        # Rename specific methods to their acronyms/new names
        df["feature_selection_method"] = df["feature_selection_method"].replace({
            "Accuracy": "LOCO",
            "SequentialBackwardElimination": "SBE",
            "SequentialForwardSelection": "SFS"
        })

        # Filter out JMI
        df = df[df["feature_selection_method"] != "JMI"]

    # ---------------------

    def extract_model_cls(model_details):
        if pd.isna(model_details):
            return "Unknown"
        details_dict = ast.literal_eval(str(model_details))
        return details_dict.get('model_cls', "Unknown")

    df["model_cls"] = df["model_details"].apply(extract_model_cls)

    # 1. AVERAGE PHASE
    df_collapsed = df.groupby(
        ["tid", "metric", "feature_selection_method"]
    )["metric_error"].mean().reset_index()

    # 2. Min-Max Scale the performance per task (100 = Best/Lowest Error, 0 = Worst/Highest Error)
    def min_max_scale(group):
        min_val = group.min()
        max_val = group.max()
        if max_val == min_val:
            return pd.Series(100.0, index=group.index)
        return ((max_val - group) / (max_val - min_val)) * 100.0

    df_collapsed["scaled_score"] = df_collapsed.groupby(["tid", "metric"])["metric_error"].transform(min_max_scale)

    # 3. Extract the RandomFeatureSelector baseline scores
    random_scores = df_collapsed[df_collapsed["feature_selection_method"] == "RandomFeatureSelector"][
        ["tid", "metric", "scaled_score"]
    ]
    random_scores = random_scores.rename(columns={"scaled_score": "random_score"})

    # 4. Merge the baseline back and calculate the Relative Score
    df_collapsed = df_collapsed.merge(random_scores, on=["tid", "metric"], how="left")

    # If a dataset didn't have a Random baseline run, assume baseline was 0
    df_collapsed["random_score"] = df_collapsed["random_score"].fillna(0)

    # Subtract baseline
    df_collapsed["relative_score"] = df_collapsed["scaled_score"] - df_collapsed["random_score"]

    # 5. Aggregate mean score and std per feature selection method
    agg_df = df_collapsed.groupby("feature_selection_method")["relative_score"].agg(["mean", "std"]).reset_index()
    agg_df.columns = ["feature_selection_method", "mean_score", "std_score"]
    agg_df["std_score"] = agg_df["std_score"].fillna(0)

    # 6. Drop the Random baseline from the final bar chart, since its relative score is always 0
    agg_df = agg_df[agg_df["feature_selection_method"] != "RandomFeatureSelector"]

    return agg_df


def plot_relative(df):
    agg_df = calculate_relative_performance(df)

    # Sort by score (best first: higher score = better)
    agg_df = agg_df.sort_values("mean_score", ascending=True)

    methods = agg_df["feature_selection_method"].values
    scores = agg_df["mean_score"].values
    stds = agg_df["std_score"].values

    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(methods))

    # Color code: Green if positive (better than random), Red if negative (worse)
    colors = ["#55A868" if score >= 0 else "#C44E52" for score in scores]

    bars = ax.bar(
        x,
        scores,
        yerr=stds,
        capsize=5,
        alpha=0.85,
        edgecolor="black",
        color=colors
    )

    # Draw a solid horizontal line at Y=0 (The Random Baseline)
    ax.axhline(0, color="black", linewidth=1.5, linestyle="--", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_title(PLOT_TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = OUTPUT_DIR / PLOT_NAME
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"✅ Relative Performance plot saved to {out}")


SCRIPT_DIR = Path(__file__).parent / "../../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/performance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    plot_relative(df)


if __name__ == "__main__":
    main()