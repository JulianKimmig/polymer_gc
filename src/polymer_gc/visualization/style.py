import matplotlib.pyplot as plt
import seaborn as sns

FIGSIZE = (10, 8)
CBAR_FONT_SIZE = 16
CBAR_TICKS_FONT_SIZE = 14
AXIS_LABEL_FONT_SIZE = 21
TITLE_FONT_SIZE = 22
LEGEND_TITLE_FONT_SIZE = 18
LEGEND_LABEL_FONT_SIZE = 16
TICK_LABEL_FONT_SIZE = 20
ANNOTATION_FONT_SIZE = 20
DPI = 600
CONTINUOUS_PALETTE = "viridis"
DISCRETE_PALETTE = "tab10"


def setup_plotting_style():
    """Configure matplotlib and seaborn for consistent, publication-quality plots"""

    # Set the style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Configure matplotlib
    plt.rcParams.update(
        {
            # Figure settings
            "figure.figsize": FIGSIZE,
            "figure.dpi": 300,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            # Font settings - Larger fonts for publication multi-plot figures
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 16,  # Increased from 12
            "axes.titlesize": TITLE_FONT_SIZE,  # Increased from 16
            "axes.labelsize": AXIS_LABEL_FONT_SIZE,  # Increased from 14
            "axes.labelweight": "bold",  # Make axis labels bold
            "xtick.labelsize": TICK_LABEL_FONT_SIZE,  # Increased from 12
            "ytick.labelsize": TICK_LABEL_FONT_SIZE,  # Increased from 12
            "legend.fontsize": LEGEND_LABEL_FONT_SIZE,  # Increased from 12
            # Line settings
            "lines.linewidth": 2,
            "lines.markersize": 8,
            # Grid settings
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            # Color settings
            "axes.prop_cycle": plt.cycler(
                "color", sns.color_palette(DISCRETE_PALETTE, 10)
            ),
            # Layout
            "figure.autolayout": True,
            "figure.constrained_layout.use": True,
        }
    )

    # Configure seaborn
    sns.set_context("paper", font_scale=1.2)
    sns.set_palette(DISCRETE_PALETTE)


# Apply plotting style
setup_plotting_style()
