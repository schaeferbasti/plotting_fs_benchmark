from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

FILE_NAME = "method_curation.csv"
PLOT_NAME = "method_table_v1.txt"


def latex_escape(text):
    text = str(text)
    text = text.replace("\\", r"\textbackslash ")
    text = text.replace("_", r"\_")
    text = text.replace("^", r"\^{}")
    text = text.replace("&", r"\&")
    text = text.replace("%", r"\%")
    text = text.replace("#", r"\#")
    return text


def make_table(df):
    df_filtered = df[
        (df["Name"].notna()) &
        (df["Source (Paper)"].notna()) &
        (df["Year (Paper)"].notna()) &
        (df["Number of appearances"].notna()) &
        (df["Final Decision "].notna())
    ].copy()

    table_cols = ["Name", "Year (Paper)", "Number of appearances", "Final Decision ", "Source (Paper)"]
    df_table = df_filtered[table_cols].copy()

    df_table = df_table.sort_values(["Name"])

    latex_rows = []
    for _, row in df_table.iterrows():
        name = latex_escape(row["Name"])
        year = str(row["Year (Paper)"]).replace(".0", "")
        appearances = str(row["Number of appearances"]).replace(".0", "")
        decision = latex_escape(row["Final Decision "])
        source = str(row["Source (Paper)"]).strip()

        name_cell = rf"\href{{{source}}}{{{name}}}"

        latex_row = f"{name_cell} & {year} & {appearances} & {decision} \\\\"
        latex_row = latex_row.replace("Variation of another method", r"Variation")
        latex_row = latex_row.replace("Too few citations", r"Citations")
        latex_row = latex_row.replace("Wrong Data Domain", r"Wrong Domain")
        latex_rows.append(latex_row)

    latex_body = "\n".join(latex_rows)

    latex = r"""\begin{longtable}{p{6.5cm}ccc}
\caption[\textbf{Feature selection methods overview.}]{%
  \textbf{Feature selection methods overview.} We list the method name with a link to the source of the method, provide year and the number of appearances across the considered literature. In the "Decision" column we list the reason for excluding the method ("Citations" if there are below three citations, "Variation" if the method is a variation of another method, and "Wrong Domain" if the method is not suited for tabular data) or we put a yes if the method is included.
  \label{appendix-table-methods-overview}
}\\
\toprule
Feature Selection Method Name & Year & \# appearances & Decision \\
\midrule
\endfirsthead

\caption[]{\textbf{Feature selection methods overview.} (continued)}\\
\toprule
Feature Selection Method Name & Year & \# appearances & Decision \\
\midrule
\endhead
\bottomrule
\endlastfoot

""" + latex_body + r"""
\end{longtable}
"""
    txt_path = OUTPUT_DIR / PLOT_NAME
    with open(txt_path, "w") as f:
        f.write(latex)


SCRIPT_DIR = Path(__file__).parent / "../../"
RESULTS_FILE = SCRIPT_DIR / "result_files/curation" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/methods"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    make_table(df)


if __name__ == "__main__":
    main()