from pathlib import Path

import pandas as pd
import re

def make_table(content):

    # 2. Split the content by the file path headers
    # This regex looks for lines that start with a path (e.g., /Users/... or generated_plots/...)
    blocks = re.split(r'(?m)^(/[A-Za-z0-9_/.-]+|generated_plots/[A-Za-z0-9_/.-]+)\n', content.strip())

    # The first element might be empty or preamble, so we start from index 1
    # blocks[1] is the path, blocks[2] is the data, blocks[3] is path, blocks[4] is data, etc.
    data_dict = {}

    for i in range(1, len(blocks), 2):
        path_line = blocks[i].strip()
        data_text = blocks[i + 1].strip()

        if not path_line or not data_text:
            continue

        # Determine the column name based on the path
        if "stability" in path_line.lower():
            col_name = "Mean Stability"
        elif "noise_1_0" in path_line:
            col_name = "Mean Validity (100%)"
        elif "noise_0_75" in path_line:
            col_name = "Mean Validity (75%)"
        elif "noise_0_5" in path_line:
            col_name = "Mean Validity (50%)"
        else:
            col_name = "Unknown"

        # Parse the data text
        lines = data_text.split('\n')

        # Skip the "selector" header line if it exists
        if lines[0].strip() == "selector":
            lines = lines[1:]

        # Skip the "dtype: float64" line at the end if it exists
        if lines[-1].startswith("dtype:"):
            lines = lines[:-1]

        # Extract method names and values
        series_data = {}
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                method = parts[0]
                val = float(parts[1])
                series_data[method] = val

        # Add to our dictionary
        data_dict[col_name] = series_data

    # 3. Combine into a single pandas DataFrame
    df = pd.DataFrame(data_dict)

    # 4. Set the index name to Method, then reset the index so it becomes a regular column
    df.index.name = "Method"
    df = df.reset_index()

    # 5. Order the columns exactly as requested
    # Method -> Stability -> Validity 50% -> Validity 75% -> Validity 100%
    cols = ["Method", "Stability", "Validity (50\%)", "Validity (75\%)", "Validity (100\%)"]
    # Only keep columns that actually exist (in case the text file is missing one)
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    # 6. Sort by Mean Stability (highest to lowest)
    if "Mean Stability" in df.columns:
        df = df.sort_values(by="Stability", ascending=False).reset_index(drop=True)

    # 7. Generate the LaTeX table
    latex_str = df.to_latex(
        index=False,
        float_format="{:.6f}".format,
        caption=r"\textbf{Mean validity and stability.} We report the average scores across all datasets. Higher is better.",
        label="table-results-stability-validity",
        column_format="l" + "c" * (len(cols) - 1),
        position="ht",
        escape=False
    )

    # Bold the headers
    for col in cols:
        latex_str = latex_str.replace(col, rf"\textbf{{{col}}}")

    print(latex_str)


FILE_NAME = "stability_validity_results.txt"
SCRIPT_DIR = Path(__file__).parent / "../../"
RESULTS_FILE = SCRIPT_DIR / "result_files" / FILE_NAME
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/stability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    with open(RESULTS_FILE, "r") as f:
        content = f.read()
    make_table(content)


if __name__ == "__main__":
    main()
