from pathlib import Path
import numpy as np
import pandas as pd
import itertools
import random
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from utils.beautify import beautify_names, remove_jmi, add_model_name

# TODO: Adapt file and plot name
FILE_NAME = "results_per_split.csv"
PLOT_NAME = "elo_per_model_v1.png"

# TODO: Adapt title and labels
PLOT_TITLE = "Bootstrapped Elo Rating (Calibrated to Random Baseline)"
X_LABEL = "Feature Selection Method"
Y_LABEL = "Elo Rating (Random = 1000, >1000 is Better)"


def calculate_bootstrapped_elo(df, k_factor=32, n_bootstraps=200, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    df = df.copy()
    df = beautify_names(df)
    df = remove_jmi(df)
    df = add_model_name(df)

    models = sorted(df["model_cls"].unique())
    methods = sorted(df["feature_selection_method"].unique())

    results = []

    for model in models:
        model_df = df[df["model_cls"] == model]

        # Pre-compute tasks to speed up bootstrapping
        tasks = []
        for _, task_df in model_df.groupby("tid"):
            tasks.append(dict(zip(task_df["feature_selection_method"], task_df["metric_error"])))

        bootstrap_ratings = {m: [] for m in methods}

        # 3. Bootstrapping Loop
        print(f"🔄 Bootstrapping Elo for model: {model}...")
        for b in range(n_bootstraps):
            # Sample datasets with replacement
            sampled_tasks = random.choices(tasks, k=len(tasks))

            # Start everyone at 1000 for this bootstrap run
            ratings = {m: 1000 for m in methods}

            for task in sampled_tasks:
                task_methods = list(task.keys())
                random.shuffle(task_methods)  # Prevent order bias

                # All 1-on-1 pairs of methods in this task
                for m1, m2 in itertools.combinations(task_methods, 2):
                    p1, p2 = task[m1], task[m2]
                    r1, r2 = ratings[m1], ratings[m2]

                    # Expected win probability
                    exp1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
                    exp2 = 1 - exp1

                    # Match outcome
                    if p1 < p2:
                        s1, s2 = 1, 0
                    elif p1 > p2:
                        s1, s2 = 0, 1
                    else:
                        s1, s2 = 0.5, 0.5

                    # Update ratings
                    ratings[m1] += k_factor * (s1 - exp1)
                    ratings[m2] += k_factor * (s2 - exp2)

            # 4. Calibrate to RandomFeatureSelector = 1000
            offset = 0
            if "Random" in ratings:
                offset = 1000 - ratings["Random"]

            for m in methods:
                if m in ratings:
                    bootstrap_ratings[m].append(ratings[m] + offset)

        # 5. Extract 95% Confidence Intervals
        for m in methods:
            if bootstrap_ratings[m]:
                arr = np.array(bootstrap_ratings[m])
                results.append({
                    "model_cls": model,
                    "feature_selection_method": m,
                    "elo_mean": np.mean(arr),
                    "elo_ci_lower": np.percentile(arr, 2.5),
                    "elo_ci_upper": np.percentile(arr, 97.5)
                })

    return pd.DataFrame(results)


def plot(df):
    df_elo = calculate_bootstrapped_elo(df)

    # Sort methods by overall average Elo
    avg_elo = df_elo.groupby("feature_selection_method")["elo_mean"].mean().sort_values(ascending=False)
    methods = avg_elo.index.tolist()

    model_names = sorted(df_elo["model_cls"].unique())

    fig, ax = plt.subplots(figsize=(16, 8))

    cmap = plt.get_cmap("Set3", len(model_names))
    colors = {m: cmap(i) for i, m in enumerate(model_names)}

    x = np.arange(len(methods))
    width = 0.8 / len(model_names)

    # Draw the calibrated baseline at 1000
    ax.axhline(1000, color="black", linewidth=1.5, linestyle="--", zorder=0, label="Random Baseline (1000)")

    for j, model in enumerate(model_names):
        # Extract model data and align with 'methods' order
        model_df = df_elo[df_elo["model_cls"] == model].set_index("feature_selection_method")
        model_df = model_df.reindex(methods).fillna({"elo_mean": 1000, "elo_ci_lower": 1000, "elo_ci_upper": 1000})

        means = model_df["elo_mean"].values
        # Error bounds for Matplotlib: [lower_distance, upper_distance]
        err_lower = means - model_df["elo_ci_lower"].values
        err_upper = model_df["elo_ci_upper"].values - means

        ax.bar(
            x + j * width,
            means,
            width=width,
            yerr=[err_lower, err_upper],
            capsize=4,
            color=colors[model],
            edgecolor="black",
            linewidth=0.8,
            zorder=3
        )

    # Formatting
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=10)
    ax.set_title(PLOT_TITLE, fontsize=14, weight='bold', pad=15)
    ax.set_xlabel(X_LABEL, fontsize=12)
    ax.set_ylabel(Y_LABEL, fontsize=12)

    # Dynamically scale Y-axis but guarantee the 1000 baseline is easily visible
    # We ignore the RandomFeatureSelector stats when calculating the y-limits now
    filtered_elo = df_elo[df_elo["feature_selection_method"] != "Random"]
    min_elo = filtered_elo["elo_ci_lower"].min()
    max_elo = filtered_elo["elo_ci_upper"].max()
    ax.set_ylim(min(min_elo - 50, 950), max(max_elo + 50, 1050))

    ax.grid(True, alpha=0.4, axis="y", linestyle="--", zorder=0)

    # Legends
    legend_elements = [Patch(facecolor=colors[m], edgecolor="black", label=m) for m in model_names]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1),
              loc="upper left", title="Downstream Model", title_fontsize=11)

    plt.tight_layout()
    out = OUTPUT_DIR / PLOT_NAME
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Bootstrapped Elo plot saved to {out}")


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