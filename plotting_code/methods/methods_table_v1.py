from pathlib import Path
import pandas as pd

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
    df_filtered = df.dropna(thresh=2)

    """[
        (df["Name"].notna()) &
        (df["Source (Paper)"].notna()) &
        (df["Year (Paper)"].notna()) &
        (df["Number of appearances"].notna()) &
        (df["Final Decision "].notna())
    ].copy()"""

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
        latex_row = latex_row.replace("Variation of another method", r"\textcolor{red}{Variation}")
        latex_row = latex_row.replace("Too few citations", r"\textcolor{red}{Citations}")
        latex_row = latex_row.replace("Wrong data domain", r"\textcolor{red}{Wrong Domain}")
        latex_row = latex_row.replace("No", r"\textcolor{red}{Not task-agnostic}")
        latex_row = latex_row.replace("Yes", r"\textcolor{green}{Yes}")
        latex_rows.append(latex_row)

    latex_body = "\n".join(latex_rows)

    latex = r"""\begin{longtable}{p{6.5cm}ccc}
\caption[\textbf{Overview of all investigated feature selection methods.}]{%
    \textbf{Overview of all investigated feature selection methods.} We list all methods with their source links, provide the year, and the number of appearances across the relevant literature (Section \ref{subsection-benchmarking-setup}). In the \textbf{Decision} column, we use \textcolor{green}{Yes} if the method is included in \benchmarkName{} or list the reason for exclusion (\textcolor{red}{Citations} if there are fewer than three citations, \textcolor{red}{Variation} if the method is a variation of another method, \textcolor{red}{Wrong Domain} if the method is not suitable for tabular data, and \textcolor{red}{Not task-agnostic} if the method is not suitable for all tasks). We additionally mark some methods with \textcolor{orange}{Yes, future work} to indicate that their implementations are ongoing. In all cases, these implementations are missing differential entropy logic.
\label{appendix-table-methods-overview}
}\\
\toprule
Feature Selection Method Name & Year & \# appearances & Decision \\
\midrule
\endfirsthead

\caption[]{\textbf{Overview of all investigated feature selection methods} (continued).}\\
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