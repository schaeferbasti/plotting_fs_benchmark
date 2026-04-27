from pathlib import Path
import numpy as np
import pandas as pd
import re

FILE_NAME = "data_foundry.csv"
PLOT_NAME = "dataset_table_v3.txt"
CITATION_FILE_NAME = "citations.txt"


def latex_escape(text):
    """Escape LaTeX special characters"""
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
        (df["Usable Task Type"].notna()) &
        (df["Problem Type"].notna()) &
        (df["Name"].notna())
    ].copy()

    # ADDED "License" column
    table_cols = ["in data-foundry", "# features", "samples", "# classes", "Problem Type", "License"]
    df_table = df_filtered[table_cols].copy()

    df_table["Problem Type"] = df_table["Problem Type"].replace({
        "Binary Classification": "binary",
        "Multiclass Classification": "multiclass",
        "Regression": "regression",
        "Ordinal Classification": "ordinal"
    })

    df_table = df_table.sort_values(["in data-foundry"])

    # Citations lookup (your existing code)
    cite_keys = {}
    doi_links = {}
    with open(CITATION_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    sections = re.split(r'===\s*(.*?)\s*===', content)
    for i in range(1, len(sections), 2):
        folder_name = sections[i].strip()
        section_body = sections[i + 1] if i + 1 < len(sections) else ""

        bib_match = re.search(r'@\w+\s*\{([^},\s{]+)', section_body)
        if bib_match:
            cite_keys[folder_name] = bib_match.group(1).strip()

        link_match = re.search(r'original_dataset_source_download_link\s*=\s*["\']([^"\']+)["\']', section_body)
        if link_match:
            doi_links[folder_name] = link_match.group(1).strip()

    # Build LaTeX rows with License column
    latex_rows = []
    for _, row in df_table.iterrows():
        folder_name = row["in data-foundry"]
        cite_key = cite_keys.get(folder_name)
        doi = doi_links.get(folder_name)
        license = latex_escape(row["License"])
        license = license.replace("nan", "/")

        escaped_name = folder_name.replace("_", "\\_")

        if cite_key and doi:
            name_cell = rf'\href{{{doi}}}{{{escaped_name}}}\cite{{{cite_key}}}'
        elif cite_key:
            name_cell = f"{escaped_name}\\cite{{{cite_key}}}"
        elif doi:
            name_cell = rf'\href{{{doi}}}{{{escaped_name}}}'
        else:
            name_cell = escaped_name

        latex_row = f"{name_cell} & {int(row['# features'])} & {int(row['samples'])} & {row['# classes']} & {row['Problem Type']} & {license}\\\\"
        latex_row = latex_row.replace("nan", "/")
        latex_row = latex_row.replace(".0", "")
        latex_row = latex_row.replace("\_", "-")
        latex_rows.append(latex_row)

    latex_body = "\n".join(latex_rows)

    latex = rf"""\begin{{longtable}}{{p{{4cm}}ccccp{{2cm}}}}
    \caption[\textbf{{Dataset overview.}}]{{%
    \textbf{{Dataset overview.}} Characteristics of datasets included in SelectArena.
    \label{{appendix-table-datasets-overview-selected}}
    }}\\
    \toprule
    Name & \# features & \# samples & \# classes & Problem & License \\
    & & & & Type & \\
    \midrule
    \endfirsthead
    \caption[]{{\textbf{{Dataset overview.}} (continued)}}\\
    \toprule
    Name & \# features & \# samples & \# classes & Problem & License \\
    & & & & Type & \\    \midrule
    \endhead
    \bottomrule
    \endlastfoot
{latex_body}
\end{{longtable}}
"""
    txt_path = OUTPUT_DIR / PLOT_NAME
    with open(txt_path, "w") as f:
        f.write(latex)

    print(f"✅ Table written to {txt_path}")


SCRIPT_DIR = Path(__file__).parent / "../../"
RESULTS_FILE = SCRIPT_DIR / "result_files/curation" / FILE_NAME
CITATION_FILE = SCRIPT_DIR / "result_files/curation" / CITATION_FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/datasets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(RESULTS_FILE, low_memory=False)
    make_table(df)


if __name__ == "__main__":
    main()