import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# --- Custom Library Imports ---
# Assuming these are in your environment and the script is run from the same location
# as the original training script.
from polymer_gc.data.database import SessionManager
from polymer_gc.data.dataset import Dataset

# --- Configuration ---
SEED = 42
DPI = 300  # High resolution for publication-quality plots

# Set a professional plotting style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
PALETTE = "viridis"  # A color-blind friendly and perceptually uniform colormap

np.random.seed(SEED)
torch.manual_seed(SEED)

# --- Directory Setup ---
# Assumes this script is in the same directory as the main `train_...` folder
db_path = "database.db"
main_dir = Path(__file__).parent / "train_tg_prediction"
data_dir = main_dir / "data"
# Create a new directory for these analysis plots
output_dir = Path(__file__).parent / "dataset_analysis_plots"
output_dir.mkdir(parents=True, exist_ok=True)

# --- Data Loading (reusing logic from the training script) ---
graph_data_file = data_dir / "tg_graph_data.pt"


def load_full_dataset():
    """
    Loads the full dataset from the cache or processes it from the database.
    This function correctly handles the one-embedding-per-graph structure.
    Returns a pandas DataFrame ready for analysis.
    """
    print("Loading full dataset...")

    # This list will store dictionaries, one for each graph/measurement
    data_records = []

    # The training script creates a cache of PyG Data objects. This is the easiest way to load.
    if graph_data_file.exists():
        print(f"Loading cached data from {graph_data_file}...")
        all_graph_data = torch.load(graph_data_file, weights_only=False)

        # This part is a bit of a workaround to get the original structure_id
        # We assume the embedding itself is a unique key for a polymer structure.
        # A more robust solution would be to save the structure_id in the Data object itself.
        embedding_to_id = {}
        next_id = 0

        for g in tqdm(all_graph_data, desc="Processing cached graphs"):
            # The Tg value is stored in 'y' for the graph
            tg_value = g.y.item()
            # All nodes in the graph have the same embedding. We take the first one.
            embedding = g.x[0].numpy()

            # Use the embedding tuple as a hashable key to assign a unique ID
            embedding_tuple = tuple(embedding)
            if embedding_tuple not in embedding_to_id:
                embedding_to_id[embedding_tuple] = next_id
                next_id += 1

            structure_id = embedding_to_id[embedding_tuple]

            record = {"Tg": tg_value, "structure_id": structure_id}
            # Add embedding features to the record
            for i, val in enumerate(embedding):
                record[f"embed_{i}"] = val
            data_records.append(record)

    else:
        # Fallback to loading directly from the database if cache doesn't exist
        print("Processing data from database (cache not found)...")
        with SessionManager(db_path) as session:
            dataset = Dataset.get(name="Tg_Prediction_from_CSV")
            data = dataset.load_entries_data()

        # Create a mapping from structure ID to its embedding vector
        strucid_to_embedding = {
            sid: emb for sid, emb in zip(data["structure_ids"], data["all_embeddings"])
        }

        # Iterate through the graph definitions
        for g_info in tqdm(data["graphs"], desc="Processing graphs from DB"):
            # The entry_pos links the graph to its target value
            entry_pos = g_info["entry_pos"]
            tg_value = data["targets"]["Tg"][entry_pos]

            # All nodes have the same structure ID in this dataset design
            structure_id = g_info["nodes"][0]
            embedding = strucid_to_embedding[structure_id]

            record = {"Tg": tg_value, "structure_id": structure_id}
            # Add embedding features to the record
            for i, val in enumerate(embedding):
                record[f"embed_{i}"] = val
            data_records.append(record)

    df = pd.DataFrame(data_records)
    print(f"Successfully created DataFrame with {len(df)} entries (measurements).")
    print(f"Found {df['structure_id'].nunique()} unique polymer structures.")
    return df


def plot_tg_distribution(df: pd.DataFrame, save_path: Path):
    """Plots the distribution of the target variable, Tg."""
    plt.figure(figsize=(8, 6))

    mean_tg = df["Tg"].mean()
    median_tg = df["Tg"].median()

    sns.histplot(df["Tg"], kde=True, bins=40, color="teal", alpha=0.7)

    plt.axvline(
        mean_tg, color="r", linestyle="--", linewidth=2, label=f"Mean: {mean_tg:.1f} K"
    )
    plt.axvline(
        median_tg,
        color="purple",
        linestyle=":",
        linewidth=2,
        label=f"Median: {median_tg:.1f} K",
    )

    plt.title("Distribution of Glass Transition Temperature (Tg)", fontsize=18, pad=20)
    plt.xlabel("Tg (K)", fontsize=14)
    plt.ylabel("Count of Measurements", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved Tg distribution plot to {save_path}")


def plot_tg_distribution_log(df: pd.DataFrame, save_path: Path):
    """Plots the distribution of Tg in log space to better visualize the spread across orders of magnitude."""
    plt.figure(figsize=(10, 6))

    # Filter out any zero or negative values for log scale
    tg_values = df["Tg"][df["Tg"] > 0]

    if len(tg_values) == 0:
        print("Warning: No positive Tg values found for log plot")
        return

    # Calculate statistics in log space
    log_tg = np.log10(tg_values)
    mean_log_tg = log_tg.mean()
    median_log_tg = log_tg.median()

    # Create histogram in log space
    sns.histplot(log_tg, kde=True, bins=40, color="coral", alpha=0.7)

    # Add vertical lines for mean and median in log space
    plt.axvline(
        mean_log_tg,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Mean (log): {10**mean_log_tg:.1f} K",
    )
    plt.axvline(
        median_log_tg,
        color="purple",
        linestyle=":",
        linewidth=2,
        label=f"Median (log): {10**median_log_tg:.1f} K",
    )

    # Add grid for better readability
    plt.grid(True, alpha=0.3)

    plt.title(
        "Distribution of Glass Transition Temperature (Tg) - Log Scale",
        fontsize=18,
        pad=20,
    )
    plt.xlabel("log₁₀(Tg) [log₁₀(K)]", fontsize=14)
    plt.ylabel("Count of Measurements", fontsize=14)
    plt.legend()

    # Add secondary x-axis showing actual Tg values
    ax1 = plt.gca()
    ax2 = ax1.twiny()

    # Set the secondary x-axis limits to match the primary axis
    ax2.set_xlim(ax1.get_xlim())

    # Create tick positions and labels for the secondary axis
    log_ticks = np.arange(np.floor(log_tg.min()), np.ceil(log_tg.max()) + 1)
    actual_ticks = 10**log_ticks

    ax2.set_xticks(log_ticks)
    ax2.set_xticklabels([f"{val:.0f}" for val in actual_ticks], fontsize=10)
    ax2.set_xlabel("Tg (K)", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved Tg distribution (log scale) plot to {save_path}")


def plot_input_embedding_tsne(df: pd.DataFrame, save_path: Path):
    """Generates and plots a t-SNE visualization of the input embeddings."""
    print("Generating t-SNE plot of unique embeddings (this may take a moment)...")

    # We only need to plot each unique polymer once on the t-SNE plot
    unique_polymers_df = df.drop_duplicates(subset=["structure_id"])

    embedding_cols = [col for col in df.columns if col.startswith("embed_")]
    embeddings = unique_polymers_df[embedding_cols].values

    # For coloring, we can use the mean Tg for each polymer
    mean_tg_per_polymer = df.groupby("structure_id")["Tg"].mean()
    targets = unique_polymers_df["structure_id"].map(mean_tg_per_polymer).values

    perplexity = min(30, len(embeddings) - 1)

    tsne = TSNE(
        n_components=2,
        verbose=0,
        perplexity=perplexity,
        max_iter=1000,
        random_state=SEED,
    )
    tsne_results = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        tsne_results[:, 0], tsne_results[:, 1], c=targets, cmap=PALETTE, alpha=0.8, s=20
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Mean Tg (K) of Polymer", fontsize=14)

    plt.title("t-SNE Projection of Unique Polymer Embeddings", fontsize=18, pad=20)
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved t-SNE plot to {save_path}")


def plot_tg_per_polymer_distribution(
    df: pd.DataFrame, save_path: Path, top_n: int = 20
):
    """
    Plots the distribution of Tg values for the most frequently measured polymers.
    This highlights the experimental variability for the same material.
    """
    counts = df["structure_id"].value_counts()
    top_polymers = counts[counts > 1].nlargest(top_n).index

    if len(top_polymers) == 0:
        print(
            "Skipping polymer-specific Tg plot: No polymers with multiple measurements."
        )
        return

    df_top = df[df["structure_id"].isin(top_polymers)]

    # Order the polymers by their median Tg for a cleaner plot
    ordered_polymers = df_top.groupby("structure_id")["Tg"].median().sort_values().index

    plt.figure(figsize=(12, 8))
    sns.boxplot(
        x="Tg",
        y="structure_id",
        data=df_top,
        order=ordered_polymers,
        orient="h",
        hue="structure_id",
        legend=False,
        palette="coolwarm",
    )

    plt.title(
        f"Tg Distribution for Top {len(top_polymers)} Most Measured Polymers",
        fontsize=18,
        pad=20,
    )
    plt.xlabel("Glass Transition Temperature (K)", fontsize=14)
    plt.ylabel("Polymer Structure ID", fontsize=14)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved polymer-specific Tg distribution plot to {save_path}")


def plot_tg_uncertainty_by_temperature(
    df: pd.DataFrame, save_path: Path, bin_width: float = 10.0
):
    """
    Analyzes Tg uncertainty by binning structures by their mean Tg and calculating
    local min/max ranges within each temperature bin. Treats this as a true vs predicted
    scenario to calculate global error metrics.

    Args:
        df: DataFrame with Tg and structure_id columns
        save_path: Path to save the plot
        bin_width: Width of temperature bins in Kelvin (default: 10K)
    """
    # Calculate statistics for each polymer structure
    polymer_stats = (
        df.groupby("structure_id")["Tg"]
        .agg(["mean", "min", "max", "count"])
        .reset_index()
    )

    # Filter to only include structures with multiple measurements
    polymer_stats = polymer_stats[polymer_stats["count"] > 1]

    if len(polymer_stats) == 0:
        print("Skipping Tg uncertainty plot: No polymers with multiple measurements.")
        return

    # Create temperature bins
    min_tg = polymer_stats["mean"].min()
    max_tg = polymer_stats["mean"].max()

    # Create bins from min to max with specified width
    bin_edges = np.arange(min_tg, max_tg + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Assign each polymer to a bin
    polymer_stats["bin"] = pd.cut(polymer_stats["mean"], bins=bin_edges, labels=False)

    # Calculate local min/max for each bin
    bin_stats = []
    for bin_idx in range(len(bin_centers)):
        bin_data = polymer_stats[polymer_stats["bin"] == bin_idx]

        if len(bin_data) > 0:
            # Calculate the overall min and max across all polymers in this bin
            local_min = bin_data["min"].min()
            local_max = bin_data["max"].max()
            n_polymers = len(bin_data)
            n_measurements = bin_data["count"].sum()

            bin_stats.append(
                {
                    "bin_center": bin_centers[bin_idx],
                    "local_min": local_min,
                    "local_max": local_max,
                    "uncertainty_range": local_max - local_min,
                    "n_polymers": n_polymers,
                    "n_measurements": n_measurements,
                }
            )

    if not bin_stats:
        print("Skipping Tg uncertainty plot: No valid bins found.")
        return

    bin_df = pd.DataFrame(bin_stats)

    # Calculate error metrics treating bin_center as "predicted" and actual measurements as "true"
    # We'll use the mean of min/max as a representative "true" value for each bin
    bin_df["true_representative"] = (bin_df["local_min"] + bin_df["local_max"]) / 2
    bin_df["predicted"] = bin_df["bin_center"]
    
    # Calculate errors
    errors = bin_df["true_representative"] - bin_df["predicted"]
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors**2).mean())
    
    # Calculate weighted errors (weighted by number of measurements in each bin)
    weights = bin_df["n_measurements"]
    weighted_mae = np.average(np.abs(errors), weights=weights)
    weighted_rmse = np.sqrt(np.average(errors**2, weights=weights))
    
    # Additional analysis: Calculate errors using actual polymer means vs bin centers
    # This gives us a more direct comparison of how well binning represents the data
    polymer_errors = []
    polymer_weights = []
    
    for bin_idx in range(len(bin_centers)):
        bin_data = polymer_stats[polymer_stats["bin"] == bin_idx]
        if len(bin_data) > 0:
            bin_center = bin_centers[bin_idx]
            # Calculate error for each polymer in this bin
            for _, polymer in bin_data.iterrows():
                polymer_error = polymer["mean"] - bin_center
                polymer_errors.append(polymer_error)
                polymer_weights.append(polymer["count"])
    
    polymer_errors = np.array(polymer_errors)
    polymer_weights = np.array(polymer_weights)
    
    # Calculate polymer-level error metrics
    polymer_mae = np.abs(polymer_errors).mean()
    polymer_rmse = np.sqrt((polymer_errors**2).mean())
    weighted_polymer_mae = np.average(np.abs(polymer_errors), weights=polymer_weights)
    weighted_polymer_rmse = np.sqrt(np.average(polymer_errors**2, weights=polymer_weights))

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1], sharex=True)

    # Plot 1: Uncertainty ranges as vertical bars
    bin_df["minmaxdiff"]=bin_df["local_max"]- bin_df["local_min"]
    bin_df["minmaxdiff_norm"]=bin_df["minmaxdiff"]-bin_df["minmaxdiff"].min()
    bin_df["minmaxdiff_norm"]=bin_df["minmaxdiff_norm"]/bin_df["minmaxdiff_norm"].max()
    
    colorrange=bin_df["minmaxdiff_norm"]*0.4+0.1
    colors = plt.cm.brg_r(
       colorrange
        )
    colors = colors[:, :3]  # Take only the first three channels (RGB)

    for i, (_, row) in enumerate(bin_df.iterrows()):
        ax1.plot(
            [row["bin_center"], row["bin_center"]],
            [row["local_min"], row["local_max"]],
            color=colors[i],
            linewidth=3,
            alpha=0.8,
            zorder=2,
        )
        ax1.scatter(
            row["bin_center"],
            row["local_min"],
            color=colors[i],
            s=50,
            alpha=0.8,
            zorder=3,
        )
        ax1.scatter(
            row["bin_center"],
            row["local_max"],
            color=colors[i],
            s=50,
            alpha=0.8,
            zorder=3,
        )

    # Add colorbar for uncertainty range
    import matplotlib as mpl
    # Create a custom colormap that maps uncertainty range to the desired color range (0.1 to 0.5)
    uncertainty_min = bin_df["minmaxdiff"].min()
    uncertainty_max = bin_df["minmaxdiff"].max()
    
    # Create a custom normalization that maps uncertainty values to the color range 0.1-0.5
    norm = mpl.colors.Normalize(vmin=uncertainty_min, vmax=uncertainty_max)
    
    # Create a custom colormap that only uses the portion of brg_r from 0.1 to 0.5
    colors_used = plt.cm.brg_r(np.linspace(0.1, 0.5, 256))
    custom_cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_brg', colors_used)
    
    sm = mpl.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, pad=0.02)
    cbar.set_label('Uncertainty Range (max-min, K)', fontsize=12)

    # Add diagonal line representing perfect prediction (y=x)
    plot_min = min(bin_df["bin_center"].min(), bin_df["local_min"].min())
    plot_max = max(bin_df["bin_center"].max(), bin_df["local_max"].max())
    ax1.plot([plot_min, plot_max], [plot_min, plot_max], 'k--', linewidth=2, 
            #  label='Perfect Prediction (y=x)', 
             alpha=0.7, zorder=1)

    # Add mean Tg line for reference
    overall_mean = df["Tg"].mean()
    ax1.axhline(
        overall_mean,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Overall Mean: {overall_mean:.1f} K",
        zorder=1
    )

    # Add error metrics as text box
    error_text = (f'Bin-level Analysis:\n'
                #  f'MAE: {mae:.1f}K\n'
                #  f'RMSE: {rmse:.1f}K\n'
                 f'Sample weighted MAE: {weighted_mae:.1f}K\n'
                 f'Sample weighted RMSE: {weighted_rmse:.1f}K\n'
                #  '\n'
                #  f'Polymer-level Analysis:\n'
                #  f'MAE: {polymer_mae:.1f}K\n'
                #  f'RMSE: {polymer_rmse:.1f}K\n'
                #  f'Weighted MAE: {weighted_polymer_mae:.1f}K\n'
                #  f'Weighted RMSE: {weighted_polymer_rmse:.1f}K'
                  )
    ax1.text(0.02, 0.98, error_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), zorder=10)

    ax1.set_ylabel("Tg Range (K)", fontsize=14)
    ax1.set_title(
        f"Tg Uncertainty Ranges by Temperature Bin (Bin Width: {bin_width}K)\n"
        f"Error Analysis: Bin Centers vs Representative True Values",
        fontsize=16,
        pad=20,
    )
    ax1.legend()
    ax1.grid(True, alpha=0.7, zorder=0)

    # Plot 2: Number of polymers and measurements per bin
    bin_centers = bin_df["bin_center"].values
    width = bin_width*0.4  # or another value, but should be less than bin_width

    bars1 = ax2.bar(
        bin_centers - width / 2,
        bin_df["n_polymers"],
        width,
        label="Number of Polymers",
        alpha=0.7,
        color="skyblue",
    )
    bars2 = ax2.bar(
        bin_centers + width / 2,
        bin_df["n_measurements"],
        width,
        label="Number of Measurements",
        alpha=0.7,
        color="lightcoral",
    )

    ax2.set_xlabel("Mean Tg (K)", fontsize=14)
    ax2.set_ylabel("Count", fontsize=14)
    ax2.set_title("Sample Size per Temperature Bin", fontsize=16, pad=20)
    # ax2.set_xticks(bin_centers)
    # ax2.set_xticklabels([f"{center:.0f}K" for center in bin_centers], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    # Print summary statistics
    print(f"Tg Uncertainty Analysis Summary:")
    print(f"  - Analyzed {len(polymer_stats)} polymers with multiple measurements")
    print(f"  - Created {len(bin_df)} temperature bins of {bin_width}K width")
    print(
        f"  - Overall Tg range: {polymer_stats['min'].min():.1f}K - {polymer_stats['max'].max():.1f}K"
    )
    print(
        f"  - Average uncertainty range per bin: {bin_df['uncertainty_range'].mean():.1f}K"
    )
    print(f"  - Maximum uncertainty range: {bin_df['uncertainty_range'].max():.1f}K")
    print(f"  - MAE (bin centers vs representative true): {mae:.1f}K")
    print(f"  - RMSE (bin centers vs representative true): {rmse:.1f}K")
    print(f"  - Weighted MAE: {weighted_mae:.1f}K")
    print(f"  - Weighted RMSE: {weighted_rmse:.1f}K")
    print(f"  - Polymer-level MAE (polymer means vs bin centers): {polymer_mae:.1f}K")
    print(f"  - Polymer-level RMSE (polymer means vs bin centers): {polymer_rmse:.1f}K")
    print(f"  - Weighted Polymer-level MAE: {weighted_polymer_mae:.1f}K")
    print(f"  - Weighted Polymer-level RMSE: {weighted_polymer_rmse:.1f}K")
    print(f"Saved Tg uncertainty analysis plot to {save_path}")


# def main():
#     """Main function to run the data analysis and plotting."""
#     print("--- Starting Dataset Analysis ---")

#     # 1. Load data into a correctly structured DataFrame
#     df = load_full_dataset()

#     # 2. Generate and save plots
#     plot_tg_distribution(df, output_dir / "1_tg_distribution.png")
#     plot_tg_distribution_log(df, output_dir / "1_tg_distribution_log.png")
#     plot_input_embedding_tsne(df, output_dir / "2_input_embedding_tsne.png")

#     # The following plots analyze the embeddings themselves, which are per-polymer.
#     # We can use the unique polymers for this.
#     unique_polymers_df = df.drop_duplicates(subset=["structure_id"])

#     # 3. Generate the new plot for per-polymer Tg variability
#     plot_tg_per_polymer_distribution(
#         df, output_dir / "5_tg_per_polymer_distribution.png"
#     )

#     # 4. Generate and save Tg uncertainty plot
#     plot_tg_uncertainty_by_temperature(
#         df, output_dir / "6_tg_uncertainty_by_temperature.png", bin_width=10
#     )

#     print("\n--- Analysis Complete ---")
#     print(f"All plots have been saved to the '{output_dir.name}' directory.")


# if __name__ == "__main__":
#     main()

# import torch
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.manifold import TSNE

# # --- Custom Library Imports ---
# # Assuming these are in your environment and the script is run from the same location
# # as the original training script.
# from polymer_gc.data.database import SessionManager
# from polymer_gc.data.dataset import Dataset

# # --- Configuration ---
# SEED = 42
# DPI = 300  # High resolution for publication-quality plots

# # Set a professional plotting style
# sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
# PALETTE = "viridis"  # A color-blind friendly and perceptually uniform colormap

# np.random.seed(SEED)
# torch.manual_seed(SEED)

# # --- Directory Setup ---
# db_path = "database.db"
# main_dir = Path(__file__).parent / "train_tg_prediction"
# data_dir = main_dir / "data"
# output_dir = Path(__file__).parent / "dataset_analysis_plots"
# output_dir.mkdir(parents=True, exist_ok=True)

# # --- Data Loading (reusing logic from the training script) ---
# graph_data_file = data_dir / "tg_graph_data.pt"


# def load_full_dataset():
#     """
#     Loads the full dataset from the cache or processes it from the database.
#     This function correctly handles the one-embedding-per-graph structure.
#     Returns a pandas DataFrame ready for analysis.
#     """
#     print("Loading full dataset...")
#     data_records = []
#     if graph_data_file.exists():
#         print(f"Loading cached data from {graph_data_file}...")
#         all_graph_data = torch.load(graph_data_file, weights_only=False)
#         embedding_to_id = {}
#         next_id = 0
#         for g in tqdm(all_graph_data, desc="Processing cached graphs"):
#             tg_value = g.y.item()
#             embedding = g.x[0].numpy()
#             embedding_tuple = tuple(embedding)
#             if embedding_tuple not in embedding_to_id:
#                 embedding_to_id[embedding_tuple] = next_id
#                 next_id += 1
#             structure_id = embedding_to_id[embedding_tuple]
#             record = {"Tg": tg_value, "structure_id": structure_id}
#             for i, val in enumerate(embedding):
#                 record[f"embed_{i}"] = val
#             data_records.append(record)
#     else:
#         print("Processing data from database (cache not found)...")
#         with SessionManager(db_path) as session:
#             dataset = Dataset.get(name="Tg_Prediction_from_CSV")
#             data = dataset.load_entries_data()
#         strucid_to_embedding = {sid: emb for sid, emb in zip(data["structure_ids"], data["all_embeddings"])}
#         for g_info in tqdm(data["graphs"], desc="Processing graphs from DB"):
#             entry_pos = g_info["entry_pos"]
#             tg_value = data["targets"]["Tg"][entry_pos]
#             structure_id = g_info["nodes"][0]
#             embedding = strucid_to_embedding[structure_id]
#             record = {"Tg": tg_value, "structure_id": structure_id}
#             for i, val in enumerate(embedding):
#                 record[f"embed_{i}"] = val
#             data_records.append(record)

#     df = pd.DataFrame(data_records)
#     print(f"Successfully created DataFrame with {len(df)} entries (measurements).")
#     print(f"Found {df['structure_id'].nunique()} unique polymer structures.")
#     return df

# # --- Text Summary Generation Functions ---

def generate_tg_distribution_summary(df: pd.DataFrame) -> str:
    """Generates a text summary for the Tg distribution."""
    stats = df['Tg'].describe()
    summary = f"""
======================================================================
ANALYSIS: Overall Tg Distribution (Plot: 1_tg_distribution.png)
======================================================================
This plot shows the distribution of all glass transition temperature (Tg)
measurements in the dataset.

Dataset Overview:
- Total number of measurements: {int(stats['count'])}
- Number of unique polymers:    {df['structure_id'].nunique()}

Descriptive Statistics for Tg (K):
- Mean:        {stats['mean']:.2f} K
- Std. Dev.:   {stats['std']:.2f} K
- Min:         {stats['min']:.2f} K
- 25% (Q1):    {stats['25%']:.2f} K
- Median (Q2): {stats['50%']:.2f} K
- 75% (Q3):    {stats['75%']:.2f} K
- Max:         {stats['max']:.2f} K
======================================================================
"""
    return summary.strip()

def generate_tsne_summary(df: pd.DataFrame) -> str:
    """Generates a text summary for the t-SNE plot."""
    num_polymers = df['structure_id'].nunique()
    embedding_dim = len([col for col in df.columns if col.startswith('embed_')])
    summary = f"""
======================================================================
ANALYSIS: t-SNE Projection of Embeddings (Plot: 2_input_embedding_tsne.png)
======================================================================
This plot visualizes the high-dimensional feature space of the unique
polymers by projecting it into two dimensions using the t-SNE algorithm.
Each point represents a unique polymer, colored by its average Tg value.
This helps to qualitatively assess whether polymers with similar properties
cluster together in the feature space.

- Number of unique polymers plotted: {num_polymers}
- Original embedding dimension:      {embedding_dim}
- Projection dimension:              2
======================================================================
"""
    return summary.strip()

def generate_per_polymer_summary(df: pd.DataFrame, top_n: int) -> str:
    """Generates a summary for the per-polymer Tg distribution plot."""
    counts = df['structure_id'].value_counts()
    multi_measurement_polymers = counts[counts > 1]

    if len(multi_measurement_polymers) == 0:
        return "No polymers with multiple measurements were found."

    polymer_ranges = df.groupby('structure_id')['Tg'].agg(lambda x: x.max() - x.min())
    max_range_polymer_id = polymer_ranges.idxmax()
    max_range_val = polymer_ranges.max()
    
    max_count_polymer_id = counts.idxmax()
    max_count_val = counts.max()

    summary = f"""
======================================================================
ANALYSIS: Per-Polymer Tg Variability (Plot: 5_tg_per_polymer_distribution.png)
======================================================================
This box plot highlights the experimental uncertainty by showing the
distribution of reported Tg values for individual polymers that have
multiple measurements in the dataset. It displays the top {top_n} most
frequently measured polymers.

Key Observations:
- Total polymers with >1 measurement: {len(multi_measurement_polymers)}
- Polymer with most measurements:     ID {max_count_polymer_id} ({max_count_val} times)
- Polymer with largest Tg range:      ID {max_range_polymer_id} ({max_range_val:.2f} K spread)
======================================================================
"""
    return summary.strip()

def generate_uncertainty_summary(df: pd.DataFrame, bin_width: float) -> str:
    """Generates a text summary and table for the Tg uncertainty plot."""
    polymer_stats = df.groupby("structure_id")["Tg"].agg(["mean", "min", "max", "count"]).reset_index()
    polymer_stats = polymer_stats[polymer_stats["count"] > 1]
    
    if len(polymer_stats) == 0:
        return "No polymers with multiple measurements found for uncertainty analysis."

    bin_edges = np.arange(polymer_stats["mean"].min(), polymer_stats["mean"].max() + bin_width, bin_width)
    polymer_stats["bin"] = pd.cut(polymer_stats["mean"], bins=bin_edges, labels=False)
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_stats_list = []
    for bin_idx, center in enumerate(bin_centers):
        bin_data = polymer_stats[polymer_stats["bin"] == bin_idx]
        if not bin_data.empty:
            bin_stats_list.append({
                "bin_center": center, "local_min": bin_data["min"].min(), 
                "local_max": bin_data["max"].max(), "n_measurements": bin_data["count"].sum()
            })
    bin_stats = pd.DataFrame(bin_stats_list)
    bin_stats["true_representative"] = (bin_stats["local_min"] + bin_stats["local_max"]) / 2
    
    errors = bin_stats["true_representative"] - bin_stats["bin_center"]
    weights = bin_stats["n_measurements"]
    weighted_mae = np.average(np.abs(errors), weights=weights)
    weighted_rmse = np.sqrt(np.average(errors**2, weights=weights))

    summary = f"""
======================================================================
ANALYSIS: Intrinsic Dataset Uncertainty (Plot: 6_tg_uncertainty_by_temperature.png)
======================================================================
This plot quantifies the inherent uncertainty in the dataset by binning
polymers based on their mean Tg and observing the range of experimental
values (min to max) within each bin. This provides an estimate of the
irreducible error a model might face.

Scope of Analysis:
- Polymers with >1 measurement: {len(polymer_stats)}
- Temperature bin width:        {bin_width} K
- Number of bins created:       {len(bin_stats)}

Error metrics are calculated by comparing the center of each temperature
bin ('prediction') against a representative true value for that bin
(the mean of the bin's min/max range). This simulates a simple
'predict the average' model to gauge baseline error.

Error Metrics (Weighted by Measurement Count per Bin):
------------------------------------------------------
| Metric       | Value      |
|--------------|------------|
| MAE          | {weighted_mae:>7.2f} K |
| RMSE         | {weighted_rmse:>7.2f} K |
------------------------------------------------------
======================================================================
"""
    return summary.strip()


def main():
    """Main function to run the data analysis and plotting."""
    print("--- Starting Dataset Analysis ---")

    # 1. Load data into a correctly structured DataFrame
    df = load_full_dataset()
    

    # 2. Generate summary and plot for overall Tg distribution
    print(generate_tg_distribution_summary(df))
    plot_tg_distribution(df, output_dir / "1_tg_distribution.png")
    
    # 3. Generate summary and plot for log-scale Tg distribution
    print("\nINFO: Generating log-scale version of the Tg distribution plot (1_tg_distribution_log.png)")
    plot_tg_distribution_log(df, output_dir / "1_tg_distribution_log.png")

    # 4. Generate summary and plot for t-SNE
    print(generate_tsne_summary(df))
    plot_input_embedding_tsne(df, output_dir / "2_input_embedding_tsne.png")

    # 5. Generate summary and plot for per-polymer Tg variability
    print(generate_per_polymer_summary(df, top_n=20))
    plot_tg_per_polymer_distribution(df, output_dir / "5_tg_per_polymer_distribution.png")

    # 6. Generate summary and plot for Tg uncertainty
    print(generate_uncertainty_summary(df, bin_width=10))
    plot_tg_uncertainty_by_temperature(df, output_dir / "6_tg_uncertainty_by_temperature.png", bin_width=10)

    print("\n--- Analysis Complete ---")
    print(f"All plots and summaries generated. Files saved to the '{output_dir.name}' directory.")


if __name__ == "__main__":
    main()





