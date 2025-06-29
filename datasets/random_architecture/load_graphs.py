"""
Graph Classification Analysis Pipeline for Polymer Architecture/Structure
========================================================================
This script loads polymer graphs, trains a classification model, and generates
comprehensive visualizations with consistent styling.
"""

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
from tqdm import tqdm
from itertools import cycle
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from matplotlib.gridspec import GridSpec
from torch import optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx

from polymer_gc.data.database import SessionManager
from polymer_gc.data.dataset import Dataset
from polymer_gc.model.base import PolyGCBaseModel

# =============================================================================
# CONFIGURATION
# =============================================================================

# Global settings
SEED = 42
EPOCHS = 0  # Set to 0 to skip training and use existing model
DPI = 600
BATCH_SIZE = 128
LEARNING_RATE = 0.001

# Data splits
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.9  # 0.8-0.9 = 10% validation

# Paths
DB_PATH = "database.db"
MAIN_DIR = Path(__file__).parent / Path(__file__).stem
DATA_DIR = MAIN_DIR / "data"
RESULT_DIR = MAIN_DIR / "results"

# Set random seeds
np.random.seed(SEED)
torch.manual_seed(SEED)

# =============================================================================
# PLOTTING STYLE CONFIGURATION
# =============================================================================

FIGSIZE = (10, 8)
CBAR_FONT_SIZE = 16
CBAR_TICKS_FONT_SIZE = 14
AXIS_LABEL_FONT_SIZE = 21
TITLE_FONT_SIZE = 22
LEGEND_TITLE_FONT_SIZE = 18
LEGEND_LABEL_FONT_SIZE = 16
TICK_LABEL_FONT_SIZE = 20
ANNOTATION_FONT_SIZE = 14


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
            "axes.titlesize": 32,  # Increased from 16
            "axes.labelsize": 24,  # Increased from 14
            "axes.labelweight": "bold",  # Make axis labels bold
            "xtick.labelsize": TICK_LABEL_FONT_SIZE,  # Increased from 12
            "ytick.labelsize": TICK_LABEL_FONT_SIZE,  # Increased from 12
            "legend.fontsize": 24,  # Increased from 12
            # Line settings
            "lines.linewidth": 2,
            "lines.markersize": 8,
            # Grid settings
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            # Color settings
            "axes.prop_cycle": plt.cycler("color", sns.color_palette("husl", 10)),
            # Layout
            "figure.autolayout": True,
            "figure.constrained_layout.use": True,
        }
    )

    # Configure seaborn
    sns.set_context("paper", font_scale=1.2)
    sns.set_palette("husl")


# Apply plotting style
setup_plotting_style()

# Color palettes for different plot types
COLOR_PALETTES = {
    "confusion_matrix": "Blues",
    "probability_matrix": "viridis",
    "tsne": "tab10",
    "roc": cycle(
        [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
    ),
    "bar_chart": "tab20",  # Changed from 'husl' to valid matplotlib colormap
    "graph_nodes": "viridis",
}

# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================


def make_bidirectional(edge_index):
    """Convert unidirectional edges to bidirectional by adding reverse edges."""
    reverse_edges = edge_index.flip(0)
    bidirectional_edges = torch.cat([edge_index, reverse_edges], dim=1)

    # Remove duplicates
    edge_tuples = [
        (int(edge_index[0, i]), int(edge_index[1, i]))
        for i in range(edge_index.shape[1])
    ]
    reverse_tuples = [
        (int(reverse_edges[0, i]), int(reverse_edges[1, i]))
        for i in range(reverse_edges.shape[1])
    ]

    all_edges = list(set(edge_tuples + reverse_tuples))

    if all_edges:
        bidirectional_edges = torch.tensor(all_edges, dtype=torch.long).T
    else:
        bidirectional_edges = torch.empty((2, 0), dtype=torch.long)

    return bidirectional_edges


def load_or_create_graph_data(db_path, data_dir):
    """Load graph data from cache or create from database."""
    graph_data_file = data_dir / "graph_data.pt"

    if graph_data_file.exists():
        print("Loading cached graph data...")
        return torch.load(graph_data_file, weights_only=False)

    print("Creating graph data from database...")
    with SessionManager(db_path) as session:
        dataset = Dataset.get(name="RandomArchitecture")
        data = dataset.load_entries_data()

    strucid_to_idx = {val: idx for idx, val in enumerate(data["structure_ids"])}
    vec_strucid_to_idx = np.vectorize(strucid_to_idx.get)
    target_names = list(data["targets"].keys())
    targets_array = np.stack([data["targets"][n] for n in target_names]).T

    all_graph_data = []
    for g in tqdm(data["graphs"], desc="Loading graphs"):
        structure_idx = vec_strucid_to_idx(g["nodes"])
        embeddings = data["all_embeddings"][structure_idx]

        edges = torch.tensor(g["edges"], dtype=torch.long).T
        bidirectional_edges = make_bidirectional(edges)

        graph_data = Data(
            x=torch.tensor(embeddings, dtype=torch.float32),
            edge_index=bidirectional_edges,
            y=torch.tensor(
                np.atleast_2d(targets_array[g["entry_pos"]]), dtype=torch.float32
            ),
            entry_pos=g["entry_pos"],
        )
        all_graph_data.append(graph_data)

    print(f"Total graphs loaded: {len(all_graph_data)}")
    torch.save(all_graph_data, graph_data_file)

    return all_graph_data


def create_data_splits(all_graph_data, train_split, val_split, seed=SEED):
    """Split data into train, validation, and test sets based on entries."""
    entry_indices = np.unique([g.entry_pos for g in all_graph_data])
    num_entries = len(entry_indices)
    np.random.RandomState(seed).shuffle(entry_indices)

    train_idx_limit = int(num_entries * train_split)
    val_idx_limit = int(num_entries * val_split)

    train_entry_indices = set(entry_indices[:train_idx_limit])
    val_entry_indices = set(entry_indices[train_idx_limit:val_idx_limit])
    test_entry_indices = set(entry_indices[val_idx_limit:])

    train_graphs = [g for g in all_graph_data if g.entry_pos in train_entry_indices]
    val_graphs = [g for g in all_graph_data if g.entry_pos in val_entry_indices]
    test_graphs = [g for g in all_graph_data if g.entry_pos in test_entry_indices]

    print(f"\nData split summary:")
    print(
        f"  Train: {len(train_graphs)} graphs from {len(train_entry_indices)} entries"
    )
    print(f"  Val: {len(val_graphs)} graphs from {len(val_entry_indices)} entries")
    print(f"  Test: {len(test_graphs)} graphs from {len(test_entry_indices)} entries")

    return train_graphs, val_graphs, test_graphs


# =============================================================================
# MODEL TRAINING
# =============================================================================


def create_model(dataset_config, device):
    """Create and configure the PolyGC model."""
    target_names = dataset_config.targets
    num_classes_per_task = [
        len(dataset_config.target_classes[target]) for target in target_names
    ]

    model_config = PolyGCBaseModel.ModelConfig(
        task_type="classification",
        monomer_features=64,
        num_classes_per_task=num_classes_per_task,
        num_gnn_layers=3,
        mlp_layer=2,
        dropout_rate=0.2,
        mass_distribution_buckets=0,
        pooling_layers=[{"type": "mean"}, {"type": "max"}],
    )

    return PolyGCBaseModel(config=model_config).to(device)


def train_model(model, train_loader, val_loader, device, epochs, lr):
    """Train the model with early stopping."""
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        factor=0.5,
        patience=15,
    )

    print("\n--- Starting Training ---")
    for epoch in range(epochs):
        try:
            # Training
            model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Training epoch {epoch + 1}"):
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = model.batch_loss(batch, "train")
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs

            avg_train_loss = total_loss / len(train_loader.dataset)

            # Validation
            model.eval()
            total_val_loss = 0
            correct_preds = [0] * len(model.config.num_classes_per_task)

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Validating epoch {epoch + 1}"):
                    batch = batch.to(device)
                    total_val_loss += (
                        model.batch_loss(batch, "val").item() * batch.num_graphs
                    )
                    preds = model.predict(batch)

                    for i in range(len(model.config.num_classes_per_task)):
                        correct_preds[i] += (
                            (preds[:, i] == batch.y[:, i].long()).sum().item()
                        )

            avg_val_loss = total_val_loss / len(val_loader.dataset)
            accuracies = [c / len(val_loader.dataset) for c in correct_preds]
            scheduler.step(avg_val_loss)

            print(f"\nEpoch {epoch + 1:02d}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            for i, acc in enumerate(accuracies):
                print(f"  Val Acc (Task {i + 1}): {acc:.2%}")

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            break

    return model


# =============================================================================
# EVALUATION UTILITIES
# =============================================================================


def evaluate_model(model, test_loader, device):
    """Evaluate model and collect predictions, probabilities, and embeddings."""
    model.eval()
    all_true_labels = []
    all_preds = []
    all_probas = []
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on test set"):
            batch = batch.to(device)

            all_true_labels.append(batch.y.cpu())
            all_preds.append(model.predict(batch).cpu())

            probas = model.predict_proba(batch)
            all_probas.append([p.cpu() for p in probas])

            embeddings = model.predict_embedding(batch)
            all_embeddings.append(embeddings.cpu())

    # Concatenate results
    y_true = torch.cat(all_true_labels, dim=0).numpy()
    y_pred = torch.cat(all_preds, dim=0).numpy()

    # Handle probabilities properly
    y_probas = []
    for i in range(len(all_probas[0])):  # For each task
        task_probas = torch.cat([batch[i] for batch in all_probas], dim=0).numpy()
        y_probas.append(task_probas)

    embeddings = torch.cat(all_embeddings, dim=0).numpy()

    return y_true, y_pred, y_probas, embeddings


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Plot confusion matrix with consistent styling."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(
        # figsize=(8, 6)
    )
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=COLOR_PALETTES["confusion_matrix"],
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
        square=True,
        annot_kws={"size": ANNOTATION_FONT_SIZE},  # Annotation font size
    )

    # Set colorbar label font size
    cbar = ax.collections[0].colorbar
    cbar.set_label("Count", fontsize=CBAR_FONT_SIZE, weight="bold")

    # Set tick label font sizes
    plt.xticks(
        fontsize=TICK_LABEL_FONT_SIZE,
        rotation=45,
    )
    plt.yticks(
        fontsize=TICK_LABEL_FONT_SIZE,
        rotation=45,
    )

    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.ylabel("True Label", fontsize=AXIS_LABEL_FONT_SIZE, weight="bold")
    plt.xlabel("Predicted Label", fontsize=AXIS_LABEL_FONT_SIZE, weight="bold")
    plt.savefig(save_path)
    plt.close()


def plot_probability_matrix(y_true, y_probas, class_names, title, save_path):
    """Plot average probability confusion matrix."""
    num_classes = len(class_names)
    prob_cm = np.zeros((num_classes, num_classes))

    for true_class_idx in range(num_classes):
        mask = y_true == true_class_idx
        if np.sum(mask) > 0:
            prob_cm[true_class_idx, :] = np.mean(y_probas[mask], axis=0)

    plt.figure(
        # figsize=(10, 8)
    )
    ax = sns.heatmap(
        prob_cm,
        annot=True,
        fmt=".2f",
        cmap=COLOR_PALETTES["probability_matrix"],
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Average Probability"},
        square=True,
        annot_kws={"size": ANNOTATION_FONT_SIZE},  # Annotation font size
    )

    # Set colorbar label font size
    cbar = ax.collections[0].colorbar
    cbar.set_label("Average Probability", fontsize=CBAR_FONT_SIZE, weight="bold")

    # Set tick label font sizes
    plt.xticks(
        fontsize=TICK_LABEL_FONT_SIZE,
        rotation=45,
    )
    plt.yticks(
        fontsize=TICK_LABEL_FONT_SIZE,
        rotation=45,
    )

    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.ylabel("True Label", fontsize=AXIS_LABEL_FONT_SIZE, weight="bold")
    plt.xlabel("Predicted Label", fontsize=AXIS_LABEL_FONT_SIZE, weight="bold")
    plt.savefig(save_path)
    plt.close()


def plot_roc_curves(y_true, y_probas, class_names, title, save_path):
    """Plot multi-class ROC curves."""
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probas[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(
        # figsize=(10, 8)
    )
    colors = COLOR_PALETTES["roc"]

    for i, color in zip(range(len(class_names)), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2, label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=AXIS_LABEL_FONT_SIZE, weight="bold")
    plt.ylabel("True Positive Rate", fontsize=AXIS_LABEL_FONT_SIZE, weight="bold")
    plt.xticks(fontsize=TICK_LABEL_FONT_SIZE)
    plt.yticks(fontsize=TICK_LABEL_FONT_SIZE)
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.legend(
        loc="lower right",
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=LEGEND_LABEL_FONT_SIZE,
    )
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()


def plot_tsne_embeddings(embeddings, labels, class_names, title, save_path):
    """Plot t-SNE visualization of embeddings."""
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=300, random_state=SEED)
    tsne_results = tsne.fit_transform(embeddings)

    plt.figure(
        # figsize=(10, 8)
    )

    # Create scatter plot
    scatter = plt.scatter(
        tsne_results[:, 0],
        tsne_results[:, 1],
        c=labels,
        cmap=COLOR_PALETTES["tsne"],
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )

    # Add colorbar with class labels
    cbar = plt.colorbar(scatter, ticks=range(len(class_names)))
    cbar.set_label("Class", fontsize=CBAR_FONT_SIZE, weight="bold")
    cbar.ax.set_yticklabels(class_names, fontsize=CBAR_TICKS_FONT_SIZE)

    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.xlabel("t-SNE Dimension 1", fontsize=AXIS_LABEL_FONT_SIZE, weight="bold")
    plt.ylabel("t-SNE Dimension 2", fontsize=AXIS_LABEL_FONT_SIZE, weight="bold")
    plt.xticks(fontsize=TICK_LABEL_FONT_SIZE)
    plt.yticks(fontsize=TICK_LABEL_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()


def plot_probability_bars(y_true, y_probas, class_names, title, save_path):
    """Plot average probability distribution as grouped bars."""
    num_classes = len(class_names)
    avg_probs = np.zeros((num_classes, num_classes))

    for true_class_idx in range(num_classes):
        mask = y_true == true_class_idx
        if np.sum(mask) > 0:
            avg_probs[true_class_idx] = np.mean(y_probas[mask], axis=0)

    # Create DataFrame for easier plotting
    data = []
    for i, true_class in enumerate(class_names):
        for j, pred_class in enumerate(class_names):
            data.append(
                {
                    "True Class": true_class,
                    "Predicted Class": pred_class,
                    "Average Probability": avg_probs[i, j],
                }
            )

    df = pd.DataFrame(data)

    # Create grouped bar plot
    plt.figure(
        # figsize=(14, 8)
    )

    # Pivot for grouped bars
    df_pivot = df.pivot(
        index="True Class", columns="Predicted Class", values="Average Probability"
    )

    ax = df_pivot.plot(
        kind="bar",
        colormap=COLOR_PALETTES["bar_chart"],
        width=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.xlabel("True Class", fontsize=AXIS_LABEL_FONT_SIZE, weight="bold")
    plt.ylabel("Average Probability", fontsize=AXIS_LABEL_FONT_SIZE, weight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=TICK_LABEL_FONT_SIZE)
    plt.yticks(fontsize=TICK_LABEL_FONT_SIZE)
    plt.legend(
        title="Predicted Class",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=LEGEND_LABEL_FONT_SIZE,
        title_fontsize=LEGEND_TITLE_FONT_SIZE,
    )
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_combined_roc_curves(
    y_true, y_probas, class_names_per_task, target_names, save_path
):
    """Plot combined ROC curves for both tasks on the same figure."""
    plt.figure()

    # Define different line styles for each task
    line_styles = ["-", "--"]
    task_colors = ["blue", "red"]

    all_auc_scores = []

    for task_idx, task_name in enumerate(target_names):
        y_true_task = y_true[:, task_idx]
        y_probas_task = y_probas[task_idx]
        class_names = class_names_per_task[task_idx]

        # Binarize labels for multiclass ROC
        y_true_bin = label_binarize(y_true_task, classes=range(len(class_names)))

        # Calculate ROC curve for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(len(class_names)):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probas_task[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            all_auc_scores.append(roc_auc[i])

        # Plot ROC curves for this task
        for i in range(len(class_names)):
            plt.plot(
                fpr[i],
                tpr[i],
                # color=task_colors[task_idx],
                linestyle=line_styles[task_idx],
                # alpha=0.7,
                linewidth=2,
                label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})",  # {task_name.replace("_", " ").title()}:
            )

    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], "k--", lw=2, alpha=0.5, label="Random (AUC = 0.50)")

    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=AXIS_LABEL_FONT_SIZE, weight="bold")
    plt.ylabel("True Positive Rate", fontsize=AXIS_LABEL_FONT_SIZE, weight="bold")
    plt.xticks(fontsize=TICK_LABEL_FONT_SIZE)
    plt.yticks(fontsize=TICK_LABEL_FONT_SIZE)

    # Calculate overall statistics
    mean_auc = np.mean(all_auc_scores)
    std_auc = np.std(all_auc_scores)

    plt.title(
        f"Combined ROC Curves: {target_names[0].replace('_', ' ').title()} vs {target_names[1].replace('_', ' ').title()}\n"
        f"Mean AUC: {mean_auc:.3f} ± {std_auc:.3f}",
        fontsize=TITLE_FONT_SIZE,
    )

    # Create custom legend with task grouping
    handles, labels = plt.gca().get_legend_handles_labels()

    # Sort legend entries by task
    task1_entries = [
        (h, l)
        for h, l in zip(handles, labels)
        if target_names[0].replace("_", " ").title() in l
    ]
    task2_entries = [
        (h, l)
        for h, l in zip(handles, labels)
        if target_names[1].replace("_", " ").title() in l
    ]
    random_entry = [(h, l) for h, l in zip(handles, labels) if "Random" in l]

    # Reorganize legend
    ordered_handles = []
    ordered_labels = []

    # Add task 1 entries
    if task1_entries:
        for h, l in task1_entries:
            ordered_handles.append(h)
            ordered_labels.append(l)

    # Add task 2 entries
    if task2_entries:
        for h, l in task2_entries:
            ordered_handles.append(h)
            ordered_labels.append(l)

    # Add random line
    if random_entry:
        ordered_handles.extend([h for h, l in random_entry])
        ordered_labels.extend([l for h, l in random_entry])

    plt.legend(  # ordered_handles, ordered_labels,
        loc="lower right",
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=LEGEND_LABEL_FONT_SIZE,  # Slightly smaller for this dense legend
        ncol=1,
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_combined_tsne(
    embeddings, y_true, class_names_per_task, target_names, save_path
):
    """Plot combined t-SNE with colors for architecture and symbols for structure."""
    print("Running t-SNE for combined visualization...")
    tsne = TSNE(n_components=2, perplexity=40, max_iter=500, random_state=SEED)
    tsne_results = tsne.fit_transform(embeddings)

    # Get unique class names
    arch_classes = class_names_per_task[0]
    struct_classes = class_names_per_task[1]

    # Create color map for architecture (first task)
    arch_colors = getattr(plt.cm, COLOR_PALETTES["tsne"])(
        np.linspace(0, 1, len(arch_classes))
    )
    arch_color_map = {name: color for name, color in zip(arch_classes, arch_colors)}

    # Create marker map for structure (second task)
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "+", "x"]
    struct_marker_map = {
        name: markers[i % len(markers)] for i, name in enumerate(struct_classes)
    }

    # Prepare data
    arch_labels = [arch_classes[int(i)] for i in y_true[:, 0]]
    struct_labels = [struct_classes[int(i)] for i in y_true[:, 1]]

    # Create plot
    plt.figure(
        # figsize=(16, 12)
    )  # Larger figure for better readability

    # Plot each combination of architecture and structure
    plotted_combinations = set()

    for arch_name in arch_classes:
        for struct_name in struct_classes:
            # Find points that match this combination
            mask = [
                (arch == arch_name and struct == struct_name)
                for arch, struct in zip(arch_labels, struct_labels)
            ]

            if any(mask):  # Only plot if we have data points
                mask_indices = np.where(mask)[0]

                plt.scatter(
                    tsne_results[mask_indices, 0],
                    tsne_results[mask_indices, 1],
                    c=[arch_color_map[arch_name]],
                    marker=struct_marker_map[struct_name],
                    s=120,  # Slightly larger markers
                    alpha=0.8,
                    edgecolors="black",
                    linewidth=0.8,
                    label=f"{arch_name} ({struct_name})"
                    if (arch_name, struct_name) not in plotted_combinations
                    else "",
                )
                plotted_combinations.add((arch_name, struct_name))

    # Create custom legends
    # Architecture legend (colors)
    arch_legend_elements = []
    for arch_name, color in arch_color_map.items():
        arch_legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=12,
                markeredgecolor="black",
                label=arch_name,
            )
        )

    # Structure legend (markers)
    struct_legend_elements = []
    for struct_name, marker in struct_marker_map.items():
        struct_legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor="gray",
                markersize=12,
                markeredgecolor="black",
                label=struct_name,
            )
        )

    # Add legends
    arch_legend = plt.legend(
        handles=arch_legend_elements,
        title=f"{target_names[0].replace('_', ' ').title()} (Color)",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=LEGEND_LABEL_FONT_SIZE,
        title_fontsize=LEGEND_TITLE_FONT_SIZE,
    )
    plt.gca().add_artist(arch_legend)  # Keep first legend when adding second

    struct_legend = plt.legend(
        handles=struct_legend_elements,
        title=f"{target_names[1].replace('_', ' ').title()} (Symbol)",
        loc="upper left",
        bbox_to_anchor=(1.02, 0.6),
        fontsize=LEGEND_LABEL_FONT_SIZE,
        title_fontsize=LEGEND_TITLE_FONT_SIZE,
    )

    plt.title(
        f"t-SNE Embedding Space\nColor: {target_names[0].replace('_', ' ').title()} | Symbol: {target_names[1].replace('_', ' ').title()}",
        fontsize=TITLE_FONT_SIZE,
    )
    plt.xlabel("t-SNE Dimension 1", fontsize=AXIS_LABEL_FONT_SIZE, weight="bold")
    plt.ylabel("t-SNE Dimension 2", fontsize=AXIS_LABEL_FONT_SIZE, weight="bold")
    plt.xticks(fontsize=TICK_LABEL_FONT_SIZE)
    plt.yticks(fontsize=TICK_LABEL_FONT_SIZE)
    plt.grid(True, alpha=0.3)

    # Adjust layout to accommodate legends
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Make room for legends
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_worst_predictions(
    test_graphs,
    y_true,
    y_pred,
    y_probas,
    class_names_per_task,
    target_names,
    save_path,
    n_top=10,
):
    """Visualize the most confident misclassifications."""
    print(f"\nIdentifying top {n_top} confident misclassifications...")

    errors = []
    for i in range(len(test_graphs)):
        for task_idx in range(2):
            true_label = int(y_true[i, task_idx])
            pred_label = int(y_pred[i, task_idx])

            if true_label != pred_label:
                error_prob = y_probas[task_idx][i, pred_label]
                errors.append(
                    {
                        "test_idx": i,
                        "task_idx": task_idx,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "error_prob": error_prob,
                        "true_prob": y_probas[task_idx][i, true_label],
                    }
                )

    # Sort by confidence in wrong prediction
    top_errors = sorted(errors, key=lambda x: x["error_prob"], reverse=True)[:n_top]

    if not top_errors:
        print("No misclassifications found.")
        return

    # Create visualization
    fig = plt.figure(figsize=(20, 4 * ((n_top + 4) // 5)))
    gs = GridSpec(((n_top + 4) // 5), 5, figure=fig, hspace=0.4, wspace=0.3)

    for idx, error in enumerate(top_errors):
        ax = fig.add_subplot(gs[idx // 5, idx % 5])

        # Get graph
        graph_data = test_graphs[error["test_idx"]]
        G = to_networkx(graph_data, to_undirected=True)

        # Node colors based on features
        node_features = graph_data.x
        node_colors = torch.mean(node_features, dim=1).numpy()

        # Plot graph
        pos = nx.kamada_kawai_layout(G)
        nx.draw(
            G,
            pos,
            ax=ax,
            node_color=node_colors,
            cmap=COLOR_PALETTES["graph_nodes"],
            node_size=50,
            with_labels=False,
            edge_color="gray",
            width=1,
        )

        # Title
        task_name = target_names[error["task_idx"]]
        true_name = class_names_per_task[error["task_idx"]][error["true_label"]]
        pred_name = class_names_per_task[error["task_idx"]][error["pred_label"]]

        ax.set_title(
            f"#{idx + 1}: {task_name}\n"
            f"Pred: {pred_name} ({error['error_prob']:.1%})\n"
            f"True: {true_name} ({error['true_prob']:.1%})",
            fontsize=AXIS_LABEL_FONT_SIZE,  # Increased from 10 for better readability
        )
        ax.axis("off")

    plt.suptitle(
        f"Top {len(top_errors)} Most Confident Misclassifications", fontsize=TITLE_FONT_SIZE
    )
    plt.savefig(save_path)
    plt.close()


def generate_analysis_report(
    y_true, y_pred, class_names_per_task, target_names, save_path
):
    """Generate comprehensive analysis report."""
    with open(save_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("POLYMER GRAPH CLASSIFICATION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        for i, task_name in enumerate(target_names):
            f.write(f"\n{'-' * 40}\n")
            f.write(f"TASK: {task_name}\n")
            f.write(f"{'-' * 40}\n\n")

            y_true_task = y_true[:, i]
            y_pred_task = y_pred[:, i]
            class_names = class_names_per_task[i]

            # Classification report
            report = classification_report(
                y_true_task, y_pred_task, target_names=class_names, digits=3
            )
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n\n")

            # Confusion matrix
            cm = confusion_matrix(y_true_task, y_pred_task)
            f.write("Confusion Matrix:\n")
            f.write(
                pd.DataFrame(cm, index=class_names, columns=class_names).to_string()
            )
            f.write("\n\n")


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def main():
    """Main analysis pipeline."""

    # Create directories
    for directory in [MAIN_DIR, DATA_DIR, RESULT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # Load dataset configuration
    with SessionManager(DB_PATH) as session:
        dataset = Dataset.get(name="RandomArchitecture")
        dataset_config = dataset.config
        target_names = dataset_config.targets

    # Get class names
    class_names_per_task = []
    for target in target_names:
        if (
            hasattr(dataset_config, "target_classes")
            and target in dataset_config.target_classes
        ):
            class_names_per_task.append(dataset_config.target_classes[target])
        else:
            num_classes = len(dataset_config.target_classes[target])
            class_names_per_task.append([f"Class {j}" for j in range(num_classes)])

    # Load data
    all_graph_data = load_or_create_graph_data(DB_PATH, DATA_DIR)
    train_graphs, val_graphs, test_graphs = create_data_splits(
        all_graph_data, TRAIN_SPLIT, VAL_SPLIT
    )

    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    model = create_model(dataset_config, device)
    model_file = MAIN_DIR / "model.pt"

    # Train or load model
    if EPOCHS > 0:
        model = train_model(
            model, train_loader, val_loader, device, EPOCHS, LEARNING_RATE
        )
        torch.save(model.state_dict(), model_file)
    elif model_file.exists():
        print("\nLoading existing model...")
        model.load_state_dict(torch.load(model_file, map_location=device))
    else:
        print("\nWarning: No training requested and no model file found!")

    # Evaluate model
    print("\n" + "=" * 60)
    print("EVALUATING MODEL ON TEST SET")
    print("=" * 60)

    y_true, y_pred, y_probas, embeddings = evaluate_model(model, test_loader, device)

    # Generate analysis report
    report_path = RESULT_DIR / "analysis_report.txt"
    generate_analysis_report(
        y_true, y_pred, class_names_per_task, target_names, report_path
    )
    print(f"\nAnalysis report saved to: {report_path}")

    # Generate visualizations for each task
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    for i, task_name in enumerate(target_names):
        print(f"\nProcessing visualizations for: {task_name}")

        y_true_task = y_true[:, i]
        y_pred_task = y_pred[:, i]
        y_probas_task = y_probas[i]
        class_names = class_names_per_task[i]

        # Generate plots
        plot_confusion_matrix(
            y_true_task,
            y_pred_task,
            class_names,
            f"Confusion Matrix: {task_name}",
            RESULT_DIR / f"confusion_matrix_{task_name}.png",
        )

        plot_probability_matrix(
            y_true_task,
            y_probas_task,
            class_names,
            f"Probability Matrix: {task_name}",
            RESULT_DIR / f"probability_matrix_{task_name}.png",
        )

        plot_roc_curves(
            y_true_task,
            y_probas_task,
            class_names,
            f"ROC Curves: {task_name}",
            RESULT_DIR / f"roc_curves_{task_name}.png",
        )

        plot_tsne_embeddings(
            embeddings,
            y_true_task,
            class_names,
            f"t-SNE Embeddings: {task_name}",
            RESULT_DIR / f"tsne_{task_name}.png",
        )

        plot_probability_bars(
            y_true_task,
            y_probas_task,
            class_names,
            f"Probability Distribution: {task_name}",
            RESULT_DIR / f"probability_bars_{task_name}.png",
        )

    # Generate combined visualizations
    print("\nGenerating combined visualizations...")

    plot_combined_roc_curves(
        y_true,
        y_probas,
        class_names_per_task,
        target_names,
        RESULT_DIR / "combined_roc_curves.png",
    )

    plot_combined_tsne(
        embeddings,
        y_true,
        class_names_per_task,
        target_names,
        RESULT_DIR / "combined_tsne.png",
    )

    plot_worst_predictions(
        test_graphs,
        y_true,
        y_pred,
        y_probas,
        class_names_per_task,
        target_names,
        RESULT_DIR / "worst_predictions.png",
    )

    print(f"\nAll visualizations saved to: {RESULT_DIR}")
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
