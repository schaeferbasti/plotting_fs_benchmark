from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# TODO: Adapt file and plot name
FILE_NAME = "data_foundry.csv"
PLOT_NAME = "dataset_table_v1.txt"

# TODO: Adapt plotting function
def make_table(df):
    # Filter only usable datasets
    df_filtered = df[
        (df["Usable Task Type"].notna()) &
        (df["Problem Type"].notna()) &
        (df["Name"].notna())
    ].copy()

    # Select your columns (edit here only—no other hardcoding)
    table_cols = ["in data-foundry", "Problem Type", "# features", "samples", "# classes"]
    df_table = df_filtered[table_cols].copy()

    df_table["Problem Type"] = df_table["Problem Type"].replace({
        "Binary Classification": "binary",
        "Multiclass Classification": "multiclass",
        "Regression": "regression",
        "Ordinal Classification": "ordinal"
    })

    # Sort by Year then Name, limit to first 20
    df_table = df_table.sort_values(["in data-foundry"])

    # Dynamic table body from CSV (matches your format exactly)
    csv_str = df_table.to_csv(index=False, sep='|', escapechar='\\', header=False)
    csv_str = csv_str.replace('_', r'\_')
    latex_body = csv_str.replace('|', ' & ').replace('\n', ' \\\\\n')

    latex = r"""\begin{table}[ht]
\centering
\scriptsize
\begin{tabular}{p{5cm}cccc}
\toprule
Name & Problem Type & \# features & \# samples & \# classes \\
\midrule
""" + latex_body + r"""\bottomrule
\end{tabular}
\caption{Characteristics of datasets included in SelectArena}
\label{tab:datasets}
\end{table}
"""
    txt_path = OUTPUT_DIR / PLOT_NAME
    with open(txt_path, "w") as f:
        f.write(latex)


# Do nothing below
SCRIPT_DIR = Path(__file__).parent / "../../"
RESULTS_FILE = SCRIPT_DIR / "result_files/curation" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/datasets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    make_table(df)


if __name__ == "__main__":
    main()
