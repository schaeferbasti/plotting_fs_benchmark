from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import ast
import matplotlib.colors as mcolors


# TODO: Adapt file and plot name
FILE_NAME = "results_per_split.csv"
PLOT_NAME = "elo_v4.pdf"

# TODO: Adapt title and labels
PLOT_TITLE = ""
X_LABEL = "Feature Selection Time (s) per 1k Features"
Y_LABEL = "Improvement over Random (%)"

N_BOOTSTRAP = 1000
SCALE = 400 / np.log(10)   # logit units -> Elo units
ANCHOR = 1000              # rating to assign the mean method (cosmetic)
RNG = np.random.default_rng(42)
CELL_KEYS = ["tid", "downstream_model", "feature_selection_budget_index", "metric"]

NAME_MAP = {
    "SequentialBackwardEliminationFeatureSelector": "RFE",
    "SequentialForwardSelectionFeatureSelector": "SFS",
    "AccuracyFeatureSelector": "LOCO",
    "ReliefFFeatureSelector": "(R)ReliefF",
    "ANOVAFeatureSelector": "F-Test",
}


def beautify(name):
    """Apply explicit renames; fall back to stripping the 'FeatureSelector' suffix."""
    if name in NAME_MAP:
        return NAME_MAP[name]
    return name.replace("FeatureSelector", "")

def aggregate_to_cells(df):
    """Collapse splits/folds/repeats: one error value per (cell, FS method)."""
    return (
        df.dropna(subset=CELL_KEYS + ["feature_selection_method", "metric_error"])
          .groupby(CELL_KEYS + ["feature_selection_method"], as_index=False)["metric_error"]
          .mean()
          .rename(columns={"metric_error": "error"})
    )


def build_matches(cell_df, methods):
    """For each cell, generate all pairwise matches between FS methods.
    Returns arrays of (winner_idx, loser_idx, dataset_id) for cluster bootstrap."""
    method_to_idx = {m: i for i, m in enumerate(methods)}
    winners, losers, datasets = [], [], []

    for _, group in cell_df.groupby(CELL_KEYS):
        errs = group.set_index("feature_selection_method")["error"]
        ms = errs.index.tolist()
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                a, b = ms[i], ms[j]
                if errs[a] == errs[b]:
                    continue
                if errs[a] < errs[b]:
                    winners.append(method_to_idx[a]); losers.append(method_to_idx[b])
                else:
                    winners.append(method_to_idx[b]); losers.append(method_to_idx[a])
                datasets.append(group["tid"].iloc[0])

    return (
        np.array(winners, dtype=np.int64),
        np.array(losers, dtype=np.int64),
        np.array(datasets),
    )

def fit_bt(winners, losers, n_methods):
    """Bradley-Terry MLE via logistic regression with no intercept.
    Symmetrize the design: each match contributes two rows (winner=1, loser=0)
    so sklearn sees both classes."""
    M = len(winners)
    X = np.zeros((2 * M, n_methods))
    # Winner perspective: +1 winner, -1 loser, label 1
    X[np.arange(M), winners] = 1
    X[np.arange(M), losers] = -1
    # Loser perspective: -1 winner, +1 loser, label 0
    X[M + np.arange(M), winners] = -1
    X[M + np.arange(M), losers] = 1
    y = np.concatenate([np.ones(M), np.zeros(M)])

    model = LogisticRegression(fit_intercept=False, C=1e9, solver="lbfgs", max_iter=1000) # regularization to not send scores to inf if method is really good
    model.fit(X, y)
    return model.coef_.flatten()


def generate_latex_table(scores_dict, metric_name, filename, label, caption_text):
    # Sort the dictionary by score in descending order
    sorted_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)

    # Determine formatting: Elo gets 1 decimal (or 0), others get 4 decimals
    is_elo = "Elo" in metric_name

    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append(f"\\caption{{\\textbf{{Feature selection {metric_name.lower()}}}: {caption_text}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lc}")
    lines.append("\\toprule")
    lines.append(f"\\textbf{{Name}} & \\textbf{{{metric_name}}} \\\\")
    lines.append("\\midrule")

    for method, score in sorted_scores:
        # Format the score
        score_str = f"{score:.1f}" if is_elo else f"{score:.4f}"

        # Clean up method names for LaTeX if necessary (e.g., escaping underscores if you have them)
        safe_method = str(method).replace("_", "\\_")

        lines.append(f"{safe_method} & {score_str} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(filename, "w") as f:
        f.write("\n".join(lines))
    print(f"✅ Generated {filename}")

def plot(methods, point_elo, lo, hi, out_path, stability_scores=None, validity_scores=None):
    order = np.argsort(point_elo)
    methods_sorted = [beautify(m) for m in np.array(methods)[order]]
    pt, lo, hi = point_elo[order], lo[order], hi[order]

    fig, ax = plt.subplots(figsize=(11, 3))
    # We will use an array of indices for the x-axis
    x = np.arange(len(methods))

    # Define how wide the side-by-side bars should be
    total_bar_width = 0.7
    half_width = total_bar_width / 2

    base_bottom = np.floor(lo.min() / 50) * 50 - 25

    # --- COLOR MAPPING LOGIC ---
    cmap_stability = mcolors.LinearSegmentedColormap.from_list("stability_cmap", ["#ffffff", "#0072B2"])
    cmap_validity = mcolors.LinearSegmentedColormap.from_list("validity_cmap", ["#ffffff", "#D55E00"])

    stability_colors = []
    validity_colors = []

    for m in methods_sorted:
        s_score = stability_scores.get(m, 0.5) if stability_scores else 0.5
        v_score = validity_scores.get(m, 0.5) if validity_scores else 0.5

        stability_colors.append(cmap_stability(s_score))
        validity_colors.append(cmap_validity(v_score))

    # --- PLOT THE VERTICALLY SPLIT (SIDE-BY-SIDE) BARS ---

    # Left bar (Stability) -> shifted slightly to the left
    ax.bar(
        x - (half_width / 2),
        pt - base_bottom,
        bottom=base_bottom,
        width=half_width,
        color=stability_colors,
        linewidth=0,
        alpha=0.9,
        label="Stability"
    )

    # Right bar (Validity) -> shifted slightly to the right
    ax.bar(
        x + (half_width / 2),
        pt - base_bottom,
        bottom=base_bottom,
        width=half_width,
        color=validity_colors,
        linewidth=0,
        alpha=0.9,
        label="Validity"
    )

    # Error bars (plotted exactly in the middle of the two bars)
    ax.errorbar(
        x, pt, yerr=[pt - lo, hi - pt],
        fmt="none", ecolor="black", capsize=3, linewidth=1.5,
    )

    # --- FORMATTING ---
    ax.set_xticks(x)
    ax.set_xticklabels(methods_sorted, rotation=30, ha="right")
    ax.set_ylabel("Elo")
    ax.set_ylim(bottom=base_bottom)
    ax.grid(axis="y", linestyle="-", alpha=0.3)
    ax.set_axisbelow(True)
    for side in ['left', 'bottom', 'right', 'top']:
        ax.spines[side].set_color("black")
        ax.spines[side].set_alpha(0.3)

    # --- ADD DUAL COLORBARS ---
    if stability_scores is not None and validity_scores is not None:
        cax_stab = fig.add_axes([0.92, 0.0, 0.01, 0.3])
        sm_stab = plt.cm.ScalarMappable(cmap=cmap_stability, norm=plt.Normalize(vmin=0, vmax=1))
        sm_stab.set_array([])
        cbar_stab = fig.colorbar(sm_stab, cax=cax_stab)
        cbar_stab.ax.set_title("stable", pad=12, fontsize=10)
        cbar_stab.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar_stab.ax.tick_params(length=0, labelsize=10)
        cbar_stab.ax.text(
            0.5, -0.1,
            "unstable",
            transform=cbar_stab.ax.transAxes,
            ha="center",
            va="top",
            fontsize=10
        )
        cax_val = fig.add_axes([0.92, 0.55, 0.01, 0.3])
        sm_val = plt.cm.ScalarMappable(cmap=cmap_validity, norm=plt.Normalize(vmin=0, vmax=1))
        sm_val.set_array([])
        cbar_val = fig.colorbar(sm_val, cax=cax_val)
        cbar_val.ax.set_title("valid", pad=12, fontsize=10)
        cbar_val.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        cbar_val.ax.tick_params(length=0, labelsize=10)
        cbar_val.ax.text(
            0.5, -0.1,
            "invalid",
            transform=cbar_val.ax.transAxes,
            ha="center",
            va="top",
            fontsize=10
        )
        plt.subplots_adjust(right=0.88)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def extract_model_cls(s):
    try:
        return ast.literal_eval(s)["model_cls"]
    except Exception:
        return None

def main():
    COLUMNS = [
        "experiment_method_name_string", "model_details", "tid", "name",
        "fold", "repeat", "sample", "split_idx",
        "metric_error", "metric_error_val", "metric", "problem_type",
        "time_train_s", "time_infer_s",
        "feature_selection_method", "feature_selection_is_scoring_method",
        "selected_feature_names", "max_features",
        "feature_selection_budget_total", "feature_selection_budget_index",
        "feature_selection_fit_time", "feature_selection_time_limit",
    ]

    df = pd.read_csv(RESULTS_FILE, low_memory=False, header=None, names=COLUMNS)

    numeric_cols = ["metric_error", "metric_error_val", "time_train_s", "time_infer_s",
                    "max_features", "feature_selection_budget_total",
                    "feature_selection_budget_index", "feature_selection_fit_time",
                    "feature_selection_time_limit", "fold", "repeat", "sample", "split_idx"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["downstream_model"] = df["model_details"].apply(extract_model_cls)

    cell_df = aggregate_to_cells(df)
    methods = sorted(cell_df["feature_selection_method"].unique())

    winners, losers, datasets = build_matches(cell_df, methods)

    # Fit point estimate and bootstrap in raw logit space
    point_beta = fit_bt(winners, losers, len(methods))
    boot_betas = np.zeros((N_BOOTSTRAP, len(methods)))
    unique_ds = np.unique(datasets)
    for b in range(N_BOOTSTRAP):
        sampled = RNG.choice(unique_ds, size=len(unique_ds), replace=True)
        idx = np.concatenate([np.where(datasets == d)[0] for d in sampled])
        boot_betas[b] = fit_bt(winners[idx], losers[idx], len(methods))

    # Anchor consistently: pin RandomFeatureSelector to ANCHOR. now we can compare to random (one shared shift for all bootstraps)
    # uncertainty measure relative to random
    random_idx = methods.index("RandomFeatureSelector")
    shift = point_beta[random_idx]
    point_elo = (point_beta - shift) * SCALE + ANCHOR # convert to elo
    boot_elos = (boot_betas - shift) * SCALE + ANCHOR

    lo = np.percentile(boot_elos, 2.5, axis=0)
    hi = np.percentile(boot_elos, 97.5, axis=0)

    stability_scores = {
        "F-Test": 0.672325,
        "LaplacianScore": 0.654536,
        "MI": 0.644649,
        "GainRatio": 0.635743,
        "mRMR": 0.584812,
        "(R)ReliefF": 0.612715,
        "RFImportance": 0.550328,
        "ElasticNet": 0.445675,
        "CART": 0.423454,
        "MarkovBlanket": 0.416231,
        "Lasso": 0.354916,
        "SFS": 0.193160,
        "LOCO": 0.184622,
        "RFE": 0.139315,
        "Random": 0.0,
    }

    validity_scores = {
        "F-Test": 0.922208,
        "GainRatio": 0.903634,
        "MI": 0.879216,
        "LaplacianScore": 0.875109,
        "SFS": 0.854128,
        "(R)ReliefF": 0.852300,
        "ElasticNet": 0.848003,
        "Lasso": 0.837922,
        "RFImportance": 0.799050,
        "LOCO": 0.758831,
        "RFE": 0.748798,
        "mRMR": 0.738260,
        "CART": 0.708525,
        "Random": 0.675176,
        "MarkovBlanket": 0.557068,
    }

    elo_scores = dict(zip(methods, point_elo))

    generate_latex_table(
        scores_dict=elo_scores,
        metric_name="Elo rating",
        filename="table_performance_elo.tex",
        label="table-results-performance-elo",
        caption_text="We report the predictive performance of each method measured in Elo rating. Higher is better."
    )

    generate_latex_table(
        scores_dict=stability_scores,
        metric_name="Mean stability",
        filename="table_stability.tex",
        label="table-results-stability",
        caption_text="We report the average stability scores across all datasets. Higher is better."
    )

    generate_latex_table(
        scores_dict=validity_scores,
        metric_name="Mean validity",
        filename="table_validity.tex",
        label="table-results-validity",
        caption_text="We report the average validity scores across all datasets. Higher is better."
    )

    out = OUTPUT_DIR / PLOT_NAME
    plot(methods, point_elo, lo, hi, out, stability_scores=stability_scores, validity_scores=validity_scores)
    print(f"✅ Saved to {PLOT_NAME}")


# Do nothing below
SCRIPT_DIR = Path(__file__).parent / "../../../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/performance/baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
