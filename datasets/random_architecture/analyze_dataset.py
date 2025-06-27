import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings

# --- Suppress minor warnings for cleaner output ---
warnings.filterwarnings("ignore", category=UserWarning)

# --- Import from your project structure ---
from polymer_gc.data.database import SessionManager
from polymer_gc.data.dataset import (
    Dataset,
    PgDatasetConfig,
    DatasetItem,
    GraphDatasetItemLink,
)

# =============================================================================
# 0. CONFIGURATION & LABEL MAPPING
# =============================================================================

# --- Define the mapping from integer codes to human-readable labels ---
ARCHITECTURE_MAP = {0: "Linear", 1: "Star", 2: "Cross-Linked", 3: "Branching"}
STRUCTURE_MAP = {0: "Homopolymer", 1: "Random Copolymer", 2: "Gradient", 3: "Block"}

# =============================================================================
# 1. SETUP & DATA LOADING
# =============================================================================


def setup_plot_style():
    """Sets a professional, publication-ready plotting style."""
    style_options = {
        "figure.figsize": (10, 6),
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "font.family": "serif",
        "lines.linewidth": 2,
        "lines.markersize": 8,
    }
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(style_options)
    print("Seaborn style 'seaborn-v0_8-whitegrid' set for plotting.")


def load_and_prepare_data(db_path: str, ds_name: str) -> pd.DataFrame:
    """
    Loads the dataset from the database and prepares a Pandas DataFrame for analysis,
    mapping integer targets to human-readable labels.
    """
    print(f"Connecting to database at: {db_path}")
    with SessionManager(db_path) as session:
        dataset = Dataset.get(name=ds_name)
        if not dataset or not dataset.items:
            raise ValueError(
                f"Dataset '{ds_name}' not found or is empty. Please ensure the database is populated."
            )

        entries = dataset.items
        print(f"Found {len(entries)} entries in the dataset.")

        data_list = []
        for entry in tqdm(entries, desc="Processing dataset entries"):
            num_graphs = len(GraphDatasetItemLink.get_graphs(entry))

            arch_int = entry.targets.get("hot_encoded_architecture")
            struct_int = entry.targets.get("hot_encoded_structure")

            data_list.append(
                {
                    "id": entry.id,
                    "mn": entry.mn,
                    "mw": entry.mw,
                    "architecture": ARCHITECTURE_MAP.get(
                        arch_int, f"Unknown ({arch_int})"
                    ),
                    "structure": STRUCTURE_MAP.get(
                        struct_int, f"Unknown ({struct_int})"
                    ),
                    "num_graphs": num_graphs,
                }
            )

    df = pd.DataFrame(data_list)

    # --- Feature Engineering & Type Conversion ---
    df["pdi"] = df["mw"] / df["mn"]

    # Convert string columns to ordered Categorical types. This is crucial for
    # ensuring plots maintain a logical, non-alphabetical order.
    architecture_order = list(ARCHITECTURE_MAP.values())
    structure_order = list(STRUCTURE_MAP.values())

    df["architecture"] = pd.Categorical(
        df["architecture"], categories=architecture_order, ordered=True
    )
    df["structure"] = pd.Categorical(
        df["structure"], categories=structure_order, ordered=True
    )

    print("\nData loaded and prepared successfully. DataFrame info:")
    df.info()
    print("\nSample of the data with mapped labels:")
    print(df.head())

    return df, dataset


# =============================================================================
# 2. PLOTTING FUNCTIONS
# =============================================================================


def create_figure_1_target_heatmap(df: pd.DataFrame, output_dir: Path):
    """Figure 1: Joint distribution of architecture and structure targets."""
    print("\nGenerating Figure 1: Target Distribution Heatmap...")

    counts_df = pd.crosstab(df["architecture"], df["structure"])

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        counts_df,
        annot=True,
        fmt="d",
        cmap="viridis",
        linewidths=0.5,
        annot_kws={"size": 12},
    )

    ax.set_title("Figure 1: Joint Distribution of Polymer Targets", pad=20)
    ax.set_xlabel("Polymer Structure")
    ax.set_ylabel("Polymer Architecture")

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    filepath = output_dir / "figure_1_target_heatmap.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")


def create_figure_2_marginal_distributions(df: pd.DataFrame, output_dir: Path):
    """Figure 2: Marginal distributions for each target variable."""
    print("Generating Figure 2: Marginal Target Distributions...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Figure 2: Marginal Distribution of Polymer Targets", fontsize=18, y=1.02
    )

    sns.countplot(
        x="architecture",
        data=df,
        ax=axes[0],
        palette="mako",
        hue="architecture",
        legend=False,
    )
    axes[0].set_title("A) Distribution of Architectures")
    axes[0].set_xlabel("Architecture")
    axes[0].set_ylabel("Count")

    sns.countplot(
        x="structure",
        data=df,
        ax=axes[1],
        palette="flare",
        hue="structure",
        legend=False,
    )
    axes[1].set_title("B) Distribution of Structures")
    axes[1].set_xlabel("Structure")
    axes[1].set_ylabel("Count")

    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        for p in ax.patches:
            ax.annotate(
                f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                fontsize=10,
                color="black",
                xytext=(0, 5),
                textcoords="offset points",
            )
        ax.margins(y=0.1)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filepath = output_dir / "figure_2_marginal_distributions.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")


def create_figure_3_molecular_weight_analysis(df: pd.DataFrame, output_dir: Path):
    """Figure 3: Joint plot of Mw vs Mn, colored by architecture."""
    print("Generating Figure 3: Molecular Weight (Mw vs. Mn) Analysis...")

    g = sns.jointplot(
        data=df,
        x="mn",
        y="mw",
        hue="architecture",
        palette="viridis",
        height=8,
        s=50,
        alpha=0.7,
    )
    g.fig.suptitle(
        "Figure 3: Weight-Average vs. Number-Average Molecular Weight",
        y=1.02,
        fontsize=16,
    )
    g.set_axis_labels(
        "Number-Average Molecular Weight (Mn) [Da]",
        "Weight-Average Molecular Weight (Mw) [Da]",
        fontsize=14,
    )

    ax = g.ax_joint
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, "--", color="gray", label="PDI = 1 (Mw=Mn)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend()

    # ax.set_xscale("log")
    # ax.set_yscale("log")

    filepath = output_dir / "figure_3_mw_vs_mn.png"
    g.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")


def create_figure_4_pdi_analysis(df: pd.DataFrame, output_dir: Path):
    """Figure 4: Violin plots showing PDI distribution across target categories."""
    print("Generating Figure 4: Polydispersity Index (PDI) Analysis...")

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    fig.suptitle(
        "Figure 4: Analysis of Polydispersity Index (PDI)", fontsize=18, y=1.02
    )

    sns.violinplot(
        x="architecture",
        y="pdi",
        data=df,
        ax=axes[0],
        palette="mako",
        hue="architecture",
        legend=False,
        inner="quartile",
    )
    axes[0].set_title("A) PDI Distribution by Architecture")
    axes[0].set_xlabel("Architecture")
    axes[0].set_ylabel("Polydispersity Index (PDI = Mw/Mn)")

    sns.violinplot(
        x="structure",
        y="pdi",
        data=df,
        ax=axes[1],
        palette="flare",
        hue="structure",
        legend=False,
        inner="quartile",
    )
    axes[1].set_title("B) PDI Distribution by Structure")
    axes[1].set_xlabel("Structure")
    axes[1].set_ylabel("")

    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        ax.axhline(y=1.0, color="r", linestyle="--", label="PDI = 1")
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filepath = output_dir / "figure_4_pdi_analysis.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")


def create_figure_5_graph_representation_analysis(
    df: pd.DataFrame, expected_n_graphs: int, output_dir: Path
):
    """Figure 5: Histogram of the number of graph representations per data point."""
    print("Generating Figure 5: Graph Representation Analysis...")

    plt.figure(figsize=(10, 6))
    ax = sns.countplot(
        x="num_graphs",
        data=df,
        palette="crest",
        hue="num_graphs",
        legend=False,
        native_scale=True,
    )

    ax.set_title("Figure 5: Distribution of Graph Representations per Data Point")
    ax.set_xlabel("Number of Associated Graphs")
    ax.set_ylabel("Count of Data Points")

    ax.axvline(
        x=expected_n_graphs,
        color="crimson",
        linestyle="--",
        linewidth=2.5,
        label=f"Expected n_graphs = {expected_n_graphs}",
    )

    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            fontsize=11,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
        )

    ax.legend()
    ax.margins(y=0.1)

    filepath = output_dir / "figure_5_graph_analysis.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filepath}")


# =============================================================================
# 3. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    DB_PATH = "database.db"
    OUTPUT_DIR = Path("./analysis_figures")
    OUTPUT_DIR.mkdir(exist_ok=True)

    DS_NAME = "RandomArchitecture"
    setup_plot_style()

    try:
        analytics_df, dataset = load_and_prepare_data(DB_PATH, DS_NAME)

        create_figure_1_target_heatmap(analytics_df, OUTPUT_DIR)
        create_figure_2_marginal_distributions(analytics_df, OUTPUT_DIR)
        create_figure_3_molecular_weight_analysis(analytics_df, OUTPUT_DIR)
        create_figure_4_pdi_analysis(analytics_df, OUTPUT_DIR)
        create_figure_5_graph_representation_analysis(
            analytics_df, dataset.config.n_graphs, OUTPUT_DIR
        )

        print(f"\n✅ Analysis complete. All figures saved to: {OUTPUT_DIR.resolve()}")

    except (ValueError, RuntimeError) as e:
        print(f"\n❌ An error occurred: {e}")
        print(
            "Please check your database path and ensure the dataset is populated correctly."
        )
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
