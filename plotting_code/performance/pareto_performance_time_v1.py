from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt

# TODO: Adapt file and plot name
FILE_NAME = "dummy_performance_results.csv"
PLOT_NAME = "pareto_performance_time_v1.png"

# TODO: Adapt title and labels
PLOT_TITLE = "Pareto Front of Performance vs. Training Time"
X_LABEL = "Time"
Y_LABEL = "Metric Error"

# TODO: Adapt plotting function
import matplotlib.pyplot as plt
from pathlib import Path

def plot(df):
    data = df.dropna(subset=["feature_selection_method", "time_train_s", "metric_error"]).copy()

    # Aggregate one point per method
    agg = (
        data.groupby("feature_selection_method", as_index=False)
        .agg(time=("time_train_s", "min"), metric_error=("metric_error", "min"))
    )

    # Compute Pareto front: minimize time and metric_error
    pareto_idx = []
    for i, row in agg.iterrows():
        dominated = False
        for j, other in agg.iterrows():
            if i == j:
                continue
            if (
                other["time"] <= row["time"]
                and other["metric_error"] <= row["metric_error"]
                and (
                    other["time"] < row["time"]
                    or other["metric_error"] < row["metric_error"]
                )
            ):
                dominated = True
                break
        if not dominated:
            pareto_idx.append(i)

    pareto = agg.loc[pareto_idx].sort_values("time")

    fig, ax = plt.subplots(figsize=(12, 6))

    # All methods
    ax.scatter(
        agg["time"],
        agg["metric_error"],
        s=60,
        color="lightgray",
        edgecolor="none",
        alpha=0.8,
        label="Methods",
    )

    # Pareto front
    ax.scatter(
        pareto["time"],
        pareto["metric_error"],
        s=80,
        color="tab:red",
        label="Pareto front",
        zorder=3,
    )
    ax.plot(
        pareto["time"],
        pareto["metric_error"],
        color="tab:red",
        linewidth=2,
        zorder=2,
    )

    # Labels on Pareto points
    for _, row in pareto.iterrows():
        ax.annotate(
            row["feature_selection_method"],
            (row["time"], row["metric_error"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )

    ax.set_title(PLOT_TITLE)
    ax.set_xlabel("Training Time (s)")
    ax.set_ylabel("Metric Error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = OUTPUT_DIR / PLOT_NAME
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


# Do nothing below
SCRIPT_DIR = Path(__file__).parent / "../../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/performance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    plot(df)


if __name__ == "__main__":
    main()
