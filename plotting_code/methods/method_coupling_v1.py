from pathlib import Path
import matplotlib.pyplot as plt

# Plot settings
PLOT_NAME = "method_coupling_v1.pdf"
PLOT_TITLE = "Model Coupling"


def plot_coupling():
    labels = ['Yes', 'No']
    counts = [3, 11]

    fig, ax = plt.subplots(figsize=(2.9, 2.9))

    # Bar plot
    ax.bar(labels, counts, color=['#0072B2', '#0072B2'], alpha=0.8)

    # Labels and Formatting
    ax.set_title(PLOT_TITLE)
    ax.set_ylabel("Number of Methods", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    for side in ['left', 'bottom', 'right', 'top']:
        ax.spines[side].set_color("black")
        ax.spines[side].set_alpha(0.3)

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