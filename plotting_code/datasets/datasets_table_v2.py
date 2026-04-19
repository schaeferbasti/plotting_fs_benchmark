from pathlib import Path
import numpy as np
import pandas as pd
import re

# TODO: Adapt file and plot name
FILE_NAME = "data_foundry.csv"
PLOT_NAME = "dataset_table_v2.txt"
CITATION_FILE_NAME = "citations.txt"


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

    # Read citations and create lookup dicts - PARSE ONCE
    cite_keys = {}
    doi_links = {}

    with open(CITATION_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # Split file into sections by "=== folder_name ==="
    sections = re.split(r'===\s*(.*?)\s*===', content)
    # sections = ["", "folder1", "...content1...", "folder2", "...content2...", ...]

    for i in range(1, len(sections), 2):
        folder_name = sections[i].strip()
        section_body = sections[i + 1] if i + 1 < len(sections) else ""

        # Extract bibtex key: @article{KEY, or @misc{KEY,
        bib_match = re.search(r'@\w+\s*\{([^},\s{]+)', section_body)
        if bib_match:
            cite_keys[folder_name] = bib_match.group(1).strip()

        # Extract DOI: original_dataset_source_download_link="..."
        link_match = re.search(r'original_dataset_source_download_link\s*=\s*["\']([^"\']+)["\']', section_body)
        if link_match:
            doi_links[folder_name] = link_match.group(1).strip()

    print(f"DEBUG: Parsed {len(cite_keys)} cite keys, {len(doi_links)} DOIs")

    # Build LaTeX body with clickable citations
    latex_rows = []
    for _, row in df_table.iterrows():
        folder_name = row["in data-foundry"]
        cite_key = cite_keys.get(folder_name)
        doi = doi_links.get(folder_name)

        # Dataset name with citation and link - FIXED ORDER
        if cite_key and doi:
            escaped_name = folder_name.replace("_", "\\_")
            name_cell = rf'\href{{{doi}}}{{{escaped_name}}}\cite{{{cite_key}}}'
        elif cite_key:
            escaped_name = folder_name.replace("_", "\\_")
            name_cell = f"{escaped_name}\\cite{{{cite_key}}}"
        elif doi:
            escaped_name = folder_name.replace("_", "\\_")
            name_cell = rf'\href{{{doi}}}{{{escaped_name}}}'
        else:
            name_cell = folder_name.replace('_', r'\_')

        latex_row = f"{name_cell} & {row['Problem Type']} & {int(row['# features'])} & {int(row['samples'])} & {row['# classes']} \\\\"
        latex_rows.append(latex_row)

    latex_body = "\n".join(latex_rows)

    latex = rf"""\begin{{table}}[ht]
\centering
\scriptsize
\begin{{tabular}}{{p{{5cm}}cccc}}
\toprule
Name & Problem Type & \# features & \# samples & \# classes \\
\midrule
{latex_body}
\bottomrule
\end{{tabular}}
\caption{{Characteristics of datasets included in SelectArena}}
\label{{datasets-table}}
\end{{table}}
"""
    txt_path = OUTPUT_DIR / PLOT_NAME
    with open(txt_path, "w") as f:
        f.write(latex)

    print(f"✅ Table written to {txt_path}")


# Do nothing below
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