import pandas as pd
import numpy as np
import random
import itertools
from pathlib import Path
from matplotlib import pyplot as plt

from utils.beautify import beautify_names, remove_jmi, add_model_name

FILE_NAME = "results_per_split.csv"
PLOT_NAME = "elo_v1.png"

PLOT_TITLE = "Global Feature Selection Performance (Bootstrapped Elo)"
X_LABEL = "Feature Selection Method"
Y_LABEL = "Elo Rating (Random Baseline = 1000)"


def calculate_bootstrapped_elo(df, k_factor=32, n_bootstraps=200, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    df = df.copy()

    df = beautify_names(df)
    df = remove_jmi(df)
    df = add_model_name(df)

    methods = df["feature_selection_method"].unique().tolist()
    # 3. Pre-compute tasks to speed up bootstrapping
    tasks = []
    for _, task_df in df.groupby(["metric"]):
        tasks.append(dict(zip(task_df["feature_selection_method"], task_df["metric_error"])))

    bootstrap_records = {m: [] for m in methods}

    # 4. Bootstrapping Loop
    print(f"🔄 Running {n_bootstraps} Elo bootstraps to compute 95% CI...")
    for b in range(n_bootstraps):
        # Sample datasets with replacement
        sampled_tasks = random.choices(tasks, k=len(tasks))
        ratings = {m: 1000 for m in methods}

        # Run matches for this bootstrap sample
        for task in sampled_tasks:
            task_methods = list(task.keys())
            random.shuffle(task_methods)  # Shuffle to remove order bias within the bootstrap

            for m1, m2 in itertools.combinations(task_methods, 2):
                p1, p2 = task[m1], task[m2]
                r1, r2 = ratings[m1], ratings[m2]

                # Expected win probability
                exp1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
                exp2 = 1 - exp1

                # Actual winner (Since we ensured Higher is Better, p1 > p2 means m1 wins)
                if p1 < p2:
                    s1, s2 = 1, 0
                elif p1 > p2:
                    s1, s2 = 0, 1
                else:
                    s1, s2 = 0.5, 0.5

                ratings[m1] += k_factor * (s1 - exp1)
                ratings[m2] += k_factor * (s2 - exp2)

        # Calibrate RandomFeatureSelector to exactly 1000 for this bootstrap run
        offset = 0
        if "RandomFeatureSelector" in ratings:
            offset = 1000 - ratings["RandomFeatureSelector"]

        for m in methods:
            bootstrap_records[m].append(ratings[m] + offset)

    # 5. Extract Mean and 95% Confidence Intervals
    results = []
    for m in methods:
        arr = np.array(bootstrap_records[m])
        results.append({
            "Method": m.replace("FeatureSelector", ""),
            "Elo_mean": np.mean(arr),
            "Elo_ci_lower": np.percentile(arr, 2.5),
            "Elo_ci_upper": np.percentile(arr, 97.5)
        })

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values("Elo_mean", ascending=False).reset_index(drop=True)
    return res_df


def plot_elo_bars(df):
    elo_df = calculate_bootstrapped_elo(df)

    elo_df = elo_df[elo_df["Method"] != "Random"].reset_index(drop=True)


    fig, ax = plt.subplots(figsize=(15, 7))

    # Highlight Random in Gray, others in Blue
    colors = ['#888888' if m == "Random" else '#4C72B0' for m in elo_df["Method"]]

    # Calculate error lengths for Matplotlib (distance from mean)
    lower_errors = elo_df["Elo_mean"] - elo_df["Elo_ci_lower"]
    upper_errors = elo_df["Elo_ci_upper"] - elo_df["Elo_mean"]
    errors = [lower_errors, upper_errors]

    # Plot bars with 95% CI error bars
    bars = ax.bar(
        elo_df["Method"],
        elo_df["Elo_mean"],
        yerr=errors,
        color=colors,
        edgecolor='black',
        alpha=0.85,
        capsize=5,  # Adds the little horizontal caps to the error bars
        error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': 'black'}
    )

    # Add the 1000 baseline line
    ax.axhline(1000, color="gray", linewidth=1.5, linestyle="--", zorder=0, label="Random Baseline (1000)")

    # Formatting
    ax.set_title(PLOT_TITLE, fontsize=14, weight='bold', pad=15)
    ax.set_xlabel(X_LABEL, fontsize=12)
    ax.set_ylabel(Y_LABEL, fontsize=12)

    ax.set_xticks(np.arange(len(elo_df)))
    ax.set_xticklabels(elo_df["Method"], rotation=35, ha="right", fontsize=11)

    # Dynamic Y-limit to ensure error bars fit, keeping 1000 in view
    min_val = elo_df["Elo_ci_lower"].min()
    max_val = elo_df["Elo_ci_upper"].max()
    ax.set_ylim(min(950, min_val - 50), max(1050, max_val + 50))

    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')

    plt.tight_layout()
    out = OUTPUT_DIR / PLOT_NAME
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Bootstrapped Elo Bar Chart saved to {out}")


# Do nothing below
SCRIPT_DIR = Path(__file__).parent / "../../../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/performance/ranking"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    plot_elo_bars(df)


if __name__ == "__main__":
    main()