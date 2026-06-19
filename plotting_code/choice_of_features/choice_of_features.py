from pathlib import Path
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings("ignore")

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("⚠️  umap-learn not installed — falling back to PCA for plot 3.")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
FILE_NAME = "results_per_split.csv"
SCRIPT_DIR = Path(__file__).parent / "../../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/choice_of_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_BUDGET_STAGE = 5   # only keep budget stages 1–5


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def parse_features(val) -> set:
    """Parse the selected_feature_names column into a Python set."""
    if pd.isna(val) or val == "" or val == "[]":
        return set()
    try:
        parsed = ast.literal_eval(val)
        return set(parsed) if isinstance(parsed, list) else set()
    except Exception:
        # Fallback: comma-separated string
        return set(str(val).strip("[]").replace("'", "").split(", "))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Common preprocessing shared by all plots."""
    df = df.copy()

    # Remove unwanted methods
    exclude = {"AccuracyLinear", "JMI"}
    df = df[~df["feature_selection_method"].isin(exclude)].copy()

    # Parse selected features
    df["features_set"] = df["selected_feature_names"].apply(parse_features)

    # Budget stage per dataset
    df["budget_stage"] = (
        df.groupby("tid")["max_features"]
        .rank(method="dense")
        .astype(int)
    )
    df = df[df["budget_stage"] <= MAX_BUDGET_STAGE].copy()

    # Aggregate (mean across splits/models) by taking the union of selected
    # features — so per (tid, method, budget_stage) we have ONE feature set
    # (union across splits; you can change to intersection if you prefer)
    df_agg = (
        df.groupby(["tid", "feature_selection_method", "budget_stage"])["features_set"]
        .apply(lambda sets: set.union(*sets) if len(sets) > 0 else set())
        .reset_index()
    )
    return df_agg


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Pairwise Jaccard Similarity Matrix
# ─────────────────────────────────────────────────────────────────────────────
def plot_jaccard_matrix(df_agg: pd.DataFrame):
    methods = sorted(df_agg["feature_selection_method"].unique())
    n = len(methods)
    method_idx = {m: i for i, m in enumerate(methods)}

    scores = np.zeros((n, n))
    counts = np.zeros((n, n))

    for (tid, budget), grp in df_agg.groupby(["tid", "budget_stage"]):
        grp = grp.set_index("feature_selection_method")
        present = [m for m in methods if m in grp.index]
        for ma, mb in combinations(present, 2):
            j = jaccard(grp.loc[ma, "features_set"], grp.loc[mb, "features_set"])
            i1, i2 = method_idx[ma], method_idx[mb]
            scores[i1, i2] += j
            scores[i2, i1] += j
            counts[i1, i2] += 1
            counts[i2, i1] += 1

    # Diagonal = 1
    np.fill_diagonal(scores, 0)
    np.fill_diagonal(counts, 1)
    with np.errstate(invalid="ignore"):
        mat = np.where(counts > 0, scores / counts, np.nan)
    np.fill_diagonal(mat, 1.0)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.7), max(6, n * 0.6)))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Mean Jaccard Similarity")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(methods, fontsize=9)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if val < 0.6 else "white")

    ax.set_title("Pairwise Feature Agreement (Mean Jaccard Similarity)", fontsize=12, weight="bold")
    plt.tight_layout()
    out = OUTPUT_DIR / "jaccard_matrix.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Jaccard matrix saved to {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Agreement Rate per Budget Stage
# ─────────────────────────────────────────────────────────────────────────────
def plot_agreement_rate(df_agg: pd.DataFrame):
    methods   = sorted(df_agg["feature_selection_method"].unique())
    n_methods = len(methods)
    budgets   = sorted(df_agg["budget_stage"].unique())

    # Thresholds: what fraction of methods must agree?
    thresholds = {
        "All methods":    n_methods,
        "≥75% of methods": max(2, int(np.ceil(0.75 * n_methods))),
        "Majority (≥50%)": max(2, int(np.ceil(0.5  * n_methods))),
        "≥2 methods":    2,
    }
    # Remove duplicates
    thresholds = dict(sorted(set((k, v) for k, v in thresholds.items()), key=lambda x: -x[1]))

    results = {label: [] for label in thresholds}

    for budget in budgets:
        sub = df_agg[df_agg["budget_stage"] == budget]

        tid_rates = {label: [] for label in thresholds}

        for tid, grp in sub.groupby("tid"):
            grp = grp.set_index("feature_selection_method")
            present_methods = [m for m in methods if m in grp.index]
            if len(present_methods) < 2:
                continue

            # All features that appear in ANY method's selection
            all_features = set.union(*[grp.loc[m, "features_set"] for m in present_methods])
            if not all_features:
                continue

            # Count how many methods selected each feature
            feat_counts = {
                f: sum(1 for m in present_methods if f in grp.loc[m, "features_set"])
                for f in all_features
            }

            budget_size = max(len(grp.loc[m, "features_set"]) for m in present_methods)
            if budget_size == 0:
                continue

            for label, threshold in thresholds.items():
                agreed = sum(1 for c in feat_counts.values() if c >= threshold)
                tid_rates[label].append(agreed / budget_size)

        for label in thresholds:
            vals = tid_rates[label]
            results[label].append(np.mean(vals) if vals else np.nan)

    # ── Plot ──
    colors = ["#0072B2", "#009E73", "#E69F00", "#D55E00"]
    fig, ax = plt.subplots(figsize=(8, 4))

    for (label, _), color in zip(thresholds.items(), colors):
        ax.plot(budgets, results[label], marker="o", label=label,
                color=color, linewidth=2, markersize=6)

    ax.set_xticks(budgets)
    ax.set_xticklabels([f"Budget {b}" for b in budgets])
    ax.set_xlabel("Budget Stage", fontsize=11)
    ax.set_ylabel("Avg. Fraction of Agreed Features", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Feature Agreement Rate per Budget Stage", fontsize=12, weight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")

    for side in ["left", "bottom", "right", "top"]:
        ax.spines[side].set_alpha(0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "agreement_rate_budget.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Agreement rate plot saved to {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — UMAP / PCA of Selection Vectors
# ─────────────────────────────────────────────────────────────────────────────
def plot_umap_pca(df_agg: pd.DataFrame):
    methods = sorted(df_agg["feature_selection_method"].unique())
    records = []

    for (tid, budget), grp in df_agg.groupby(["tid", "budget_stage"]):
        grp = grp.set_index("feature_selection_method")
        present = [m for m in methods if m in grp.index]
        if len(present) < 2:
            continue

        # All features in this dataset×budget slice
        all_feats = sorted(set.union(*[grp.loc[m, "features_set"] for m in present]))
        if not all_feats:
            continue

        feat_idx = {f: i for i, f in enumerate(all_feats)}
        for m in present:
            vec = np.zeros(len(all_feats), dtype=np.float32)
            for f in grp.loc[m, "features_set"]:
                if f in feat_idx:
                    vec[feat_idx[f]] = 1.0
            records.append({"method": m, "tid": tid, "budget_stage": budget, "vec": vec})

    if not records:
        print("⚠️  No records for UMAP/PCA plot — skipping.")
        return

    # PCA on the concatenated vectors (same length per dataset×budget)
    # We reduce per-dataset then stack — otherwise vectors are different lengths
    # Strategy: PCA per (tid, budget), project to 2D, collect 2D points
    points, point_methods, point_budgets = [], [], []

    for (tid, budget), grp_records in pd.DataFrame(records).groupby(["tid", "budget_stage"]):
        vecs = np.stack(grp_records["vec"].values)
        if vecs.shape[0] < 2 or vecs.shape[1] < 2:
            continue
        n_comp = min(2, vecs.shape[0] - 1, vecs.shape[1])
        pca = PCA(n_components=n_comp)
        proj = pca.fit_transform(vecs)
        if proj.shape[1] < 2:
            proj = np.hstack([proj, np.zeros((proj.shape[0], 1))])
        points.append(proj)
        point_methods.extend(grp_records["method"].tolist())
        point_budgets.extend([budget] * len(grp_records))

    if not points:
        print("⚠️  Could not compute projections — skipping UMAP/PCA plot.")
        return

    all_points = np.vstack(points)

    # Global UMAP or PCA over the per-dataset projections
    if HAS_UMAP and all_points.shape[0] > 10:
        reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(15, all_points.shape[0] - 1))
        label = "UMAP"
    else:
        reducer = PCA(n_components=2)
        label = "PCA"

    embedding = reducer.fit_transform(all_points)

    method_colors = [
        "#0072B2", "#E69F00", "#009E73", "#D55E00",
        "#CC79A7", "#56B4E9", "#F0E442", "#000000",
        "#999999", "#882255", "#44AA99", "#DDCC77",
    ]
    color_map = {m: method_colors[i % len(method_colors)] for i, m in enumerate(methods)}

    fig, ax = plt.subplots(figsize=(9, 6))
    for method in methods:
        mask = [m == method for m in point_methods]
        pts = embedding[mask]
        if pts.shape[0] == 0:
            continue
        ax.scatter(pts[:, 0], pts[:, 1], label=method,
                   color=color_map[method], alpha=0.5, s=18, linewidths=0)

    ax.set_xlabel(f"Method", fontsize=11)
    ax.set_ylabel(f"Dataset", fontsize=11)
    ax.set_title(f"{label} of Feature Selection Vectors (per dataset×budget)", fontsize=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              title="Method", framealpha=0.9, fontsize=8)
    ax.grid(True, alpha=0.2)

    for side in ["left", "bottom", "right", "top"]:
        ax.spines[side].set_alpha(0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / f"{'umap' if HAS_UMAP else 'pca'}_selection_vectors.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ {label} plot saved to {out}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Co-selection Frequency Heatmap (Rank Position × Method)
# ─────────────────────────────────────────────────────────────────────────────
def plot_coselection_heatmap(df_agg: pd.DataFrame):
    """
    For each method and each rank position (1st selected, 2nd selected, …),
    compute how often each rank position is filled — averaged across datasets.
    Because feature names differ across datasets, we use rank position (1..k)
    as the common axis.
    """
    methods = sorted(df_agg["feature_selection_method"].unique())
    budgets = sorted(df_agg["budget_stage"].unique())
    max_k = 0

    # First pass: find max budget size
    for m in methods:
        sub = df_agg[df_agg["feature_selection_method"] == m]
        for _, row in sub.iterrows():
            max_k = max(max_k, len(row["features_set"]))

    if max_k == 0:
        print("⚠️  No feature data for co-selection heatmap.")
        return

    # Matrix: methods × rank positions
    # For each (method, dataset, budget_stage), record 1 for each rank up to len(features_set)
    # Then average across datasets
    freq = {m: np.zeros(max_k) for m in methods}
    counts = {m: 0 for m in methods}

    for (tid, budget), grp in df_agg.groupby(["tid", "budget_stage"]):
        grp = grp.set_index("feature_selection_method")
        for m in methods:
            if m not in grp.index:
                continue
            feat_set = grp.loc[m, "features_set"]
            k = len(feat_set)
            if k > 0:
                freq[m][:k] += 1
                counts[m] += 1

    # Normalize: freq[m][r] = fraction of (tid×budget) instances where rank r was filled
    mat = np.zeros((len(methods), max_k))
    for i, m in enumerate(methods):
        if counts[m] > 0:
            mat[i] = freq[m] / counts[m]

    # Trim to rank positions that are non-zero for at least one method
    nonzero_cols = np.where(mat.max(axis=0) > 0)[0]
    if len(nonzero_cols) == 0:
        print("⚠️  All-zero matrix for co-selection heatmap.")
        return
    mat = mat[:, :nonzero_cols[-1] + 1]
    n_ranks = mat.shape[1]

    fig, ax = plt.subplots(figsize=(min(24, max(10, n_ranks * 0.3)), max(4, len(methods) * 0.55)))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Fraction of instances where rank is filled")

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("Feature Rank Position (by selection order)", fontsize=11)
    ax.set_title("Co-selection Frequency: Rank Position × Method (aggregated across datasets)", fontsize=12, weight="bold")

    # Only label every 5th rank to avoid clutter
    tick_positions = list(range(0, n_ranks, max(1, n_ranks // 20)))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(r + 1) for r in tick_positions], fontsize=8)

    plt.tight_layout()
    out = OUTPUT_DIR / "coselection_frequency_heatmap.pdf"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Co-selection frequency heatmap saved to {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("📂 Loading data…")
    df = pd.read_csv(RESULTS_FILE, low_memory=False)

    print("🔧 Preprocessing…")
    df_agg = prepare(df)

    print(f"   Methods: {sorted(df_agg['feature_selection_method'].unique())}")
    print(f"   Datasets: {df_agg['tid'].nunique()}")
    print(f"   Budget stages: {sorted(df_agg['budget_stage'].unique())}")

    print("\n📊 Plot 1 — Jaccard Similarity Matrix…")
    plot_jaccard_matrix(df_agg)

    print("📊 Plot 2 — Agreement Rate per Budget Stage…")
    plot_agreement_rate(df_agg)

    print("📊 Plot 3 — UMAP / PCA of Selection Vectors…")
    plot_umap_pca(df_agg)

    print("📊 Plot 4 — Co-selection Frequency Heatmap…")
    plot_coselection_heatmap(df_agg)

    print("\n✅ All plots saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()