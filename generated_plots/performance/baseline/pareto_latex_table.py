import pandas as pd

INPUT_CSV = "pareto_data.csv"
OUTPUT_TEX = "feature_selection_results_table.tex"

CAPTION = (
    "\\textbf{Feature selection performance and runtime}: "
    "We report the mean score and mean runtime for all benchmarked feature selection methods. "
    "The mean score can be interpreted as the average improvement over the random baseline. "
    "Runtime is reported in seconds."
)

LABEL = "table-feature-selection-performance"

def format_score(x):
    return f"{x:.2f}"

def format_time(x):
    if x < 1:
        return f"{x:.3f}"
    elif x < 100:
        return f"{x:.2f}"
    elif x < 10000:
        return f"{x:.1f}"
    return f"{x:.0f}"

df = pd.read_csv(INPUT_CSV)

# Optional: rename columns for prettier LaTeX headers
df = df.rename(columns={
    "feature_selection_method": "Name",
    "mean_score": "Mean score",
    "mean_time": "Mean runtime (s)"
})

# Optional: sort by score descending
df = df.sort_values("Mean score", ascending=False).reset_index(drop=True)

lines = []
lines.append("\\begin{table}[ht]")
lines.append(f"\\caption{{{CAPTION}}}")
lines.append(f"\\label{{{LABEL}}}")
lines.append("\\centering")
lines.append("\\small")
lines.append("\\begin{tabular}{lrr}")
lines.append("\\toprule")
lines.append("\\textbf{Name} & \\textbf{Mean score} & \\textbf{Mean runtime (s)} \\\\")
lines.append("\\midrule")

for _, row in df.iterrows():
    name = row["Name"]
    score = format_score(row["Mean score"])
    runtime = format_time(row["Mean runtime (s)"])
    lines.append(f"{name} & {score} & {runtime} \\\\")

lines.append("\\bottomrule")
lines.append("\\end{tabular}")
lines.append("\\end{table}")

with open(OUTPUT_TEX, "w") as f:
    f.write("\n".join(lines))

print(f"Wrote LaTeX table to {OUTPUT_TEX}")