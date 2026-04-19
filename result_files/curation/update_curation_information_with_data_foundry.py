import re
from pathlib import Path

import nbformat
import pandas as pd

BASE_DIR = Path(".")  # current folder
CSV_PATH = BASE_DIR / "data_foundry.csv"
MATCH_COL = "in data-foundry"
ROWS_COL = "samples"
COLS_COL = "# features"

# row_pattern = re.compile(r"Rows:\s*(\d{1,3}(?:,\d{3})*|\d+)", re.IGNORECASE)
# col_pattern = re.compile(r"Columns:\s*(\d{1,3}(?:,\d{3})*|\d+)", re.IGNORECASE)
row_pattern = re.compile(r"Rows:\s*([\d,]+)", re.IGNORECASE)
col_pattern = re.compile(r"Columns:\s*([\d,]+)", re.IGNORECASE)


def extract_rows_cols_from_notebook(ipynb_path):
    with open(ipynb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    rows = None
    cols = None

    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue

        for output in cell.get("outputs", []):
            output_type = output.get("output_type")

            texts = []

            if output_type == "stream":
                text = output.get("text", "")
                if isinstance(text, list):
                    text = "".join(text)
                texts.append(text)

            elif output_type in {"execute_result", "display_data"}:
                data = output.get("data", {})
                text_plain = data.get("text/plain", "")
                if isinstance(text_plain, list):
                    text_plain = "".join(text_plain)
                texts.append(text_plain)

            for text in texts:
                if rows is None:
                    m = row_pattern.search(text)
                    if m:
                        rows = int(m.group(1).replace(",", ""))
                if cols is None:
                    m = col_pattern.search(text)
                    if m:
                        cols = int(m.group(1).replace(",", ""))

                if rows is not None and cols is not None:
                    return rows, cols

    return rows, cols


def main():
    df = pd.read_csv(CSV_PATH)

    extracted = {}

    for subdir in BASE_DIR.iterdir():
        if not subdir.is_dir():
            continue

        ipynb_files = list(subdir.glob("*.ipynb"))
        if len(ipynb_files) != 1:
            print(f"Skipping {subdir.name}: expected 1 ipynb, found {len(ipynb_files)}")
            continue

        ipynb_path = ipynb_files[0]
        rows, cols = extract_rows_cols_from_notebook(ipynb_path)

        if rows is None and cols is None:
            print(f"No Rows/Columns info found in {ipynb_path}")
            continue

        extracted[subdir.name] = {
            ROWS_COL: rows,
            COLS_COL: cols,
        }
        print(f"{subdir.name}: rows={rows}, cols={cols}")

    for folder_name, values in extracted.items():
        mask = df[MATCH_COL] == folder_name
        if mask.any():
            if values[ROWS_COL] is not None:
                df.loc[mask, ROWS_COL] = values[ROWS_COL]
            if values[COLS_COL] is not None:
                df.loc[mask, COLS_COL] = values[COLS_COL]
        else:
            print(f"No match in CSV for folder: {folder_name}")

    output_path = BASE_DIR / "data_foundry.csv"
    df.to_csv(output_path, index=False)
    print(f"\nUpdated CSV written to: {output_path}")


if __name__ == "__main__":
    main()