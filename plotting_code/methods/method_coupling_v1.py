from pathlib import Path
import matplotlib.pyplot as plt

# Plot settings
PLOT_NAME = "method_coupling_v1.svg"
PLOT_TITLE = "Method Coupling"


def plot_coupling():
    # Data: 3 with, 11 without
    labels = ['Yes', 'No']
    counts = [3, 11]

    fig, ax = plt.subplots(figsize=(4, 4))

    # Bar plot
    ax.bar(labels, counts, color=['#4C72B0', '#4C72B0'], alpha=0.8, edgecolor="black")

    # Labels and Formatting
    ax.set_title(PLOT_TITLE)
    ax.set_ylabel("Number of Methods")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save to your output directory
    out = OUTPUT_DIR / PLOT_NAME
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


# Directory setup
SCRIPT_DIR = Path(__file__).parent / "../../"
OUTPUT_DIR = SCRIPT_DIR / "generated_plots/methods"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    plot_coupling()