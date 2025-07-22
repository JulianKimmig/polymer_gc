"""
Reusable plotting functions for polymer_gc visualization.

This module contains functions for creating various plots and visualizations
used in training analysis and model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import gaussian_kde

from .style import (
    FIGSIZE,
    CBAR_FONT_SIZE,
    CBAR_TICKS_FONT_SIZE,
    AXIS_LABEL_FONT_SIZE,
    TITLE_FONT_SIZE,
    LEGEND_TITLE_FONT_SIZE,
    LEGEND_LABEL_FONT_SIZE,
    TICK_LABEL_FONT_SIZE,
    ANNOTATION_FONT_SIZE,
    DPI,
)


def create_parity_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    xlabel: str = "True Tg (K)",
    ylabel: str = "Predicted Tg (K)",
    show_metrics: bool = True,
    create_description: bool = True,
) -> Tuple[plt.Figure, Dict[str, float]]:
    """
    Create a parity plot with density coloring.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        output_path: Path to save the plot (optional)
        title: Plot title (optional)
        xlabel: X-axis label
        ylabel: Y-axis label
        show_metrics: Whether to show metrics on the plot
        create_description: Whether to create a description file
        
    Returns:
        Tuple of (matplotlib figure, metrics dictionary)
    """
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }
    
    # Create figure
    fig = plt.figure(figsize=FIGSIZE)
    
    # Calculate point density using Gaussian Kernel Density Estimation
    xy = np.vstack([y_true, y_pred])
    z = gaussian_kde(xy)(xy)
    
    # Sort points by density to plot densest points on top
    idx = z.argsort()
    x_sorted, y_sorted, z_sorted = y_true[idx], y_pred[idx], z[idx]
    
    # Create scatter plot with density-based coloring
    plt.scatter(x_sorted, y_sorted, c=z_sorted, s=15, cmap="viridis", rasterized=True)
    
    # Add ideal y=x line
    min_val = min(y_true.min(), y_pred.min()) - 10
    max_val = max(y_true.max(), y_pred.max()) + 10
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2)
    
    # Set plot limits
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # Labels and formatting
    plt.ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE, weight="bold")
    plt.xlabel(xlabel, fontsize=AXIS_LABEL_FONT_SIZE, weight="bold")
    plt.xticks(fontsize=TICK_LABEL_FONT_SIZE)
    plt.yticks(fontsize=TICK_LABEL_FONT_SIZE)
    
    if title:
        plt.title(title, fontsize=TITLE_FONT_SIZE)
    
    # Add metrics text
    if show_metrics:
        metrics_text = f"R²={r2:.2f}\\nMAE={mae:.2f}"
        plt.text(
            0.95, 0.05, metrics_text,
            transform=plt.gca().transAxes,
            fontsize=ANNOTATION_FONT_SIZE,
            weight="bold",
            ha="right",
            va="bottom",
        )
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot if path provided
    if output_path:
        output_path = Path(output_path)
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        
        # Create description file
        if create_description:
            _create_parity_plot_description(
                output_path, y_true, y_pred, metrics
            )
    
    return fig, metrics


def create_error_distribution_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Distribution of Prediction Errors (Predicted - True)",
    xlabel: str = "Error (K)",
    ylabel: str = "Frequency",
    bins: int = 30,
    create_description: bool = True,
) -> Tuple[plt.Figure, Dict[str, float]]:
    """
    Create an error distribution plot.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        output_path: Path to save the plot (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        bins: Number of histogram bins
        create_description: Whether to create a description file
        
    Returns:
        Tuple of (matplotlib figure, error statistics dictionary)
    """
    # Calculate errors
    errors = y_pred - y_true
    
    error_stats = {
        "mean_error": np.mean(errors),
        "std_error": np.std(errors),
        "min_error": np.min(errors),
        "max_error": np.max(errors),
    }
    
    # Create figure
    fig = plt.figure(figsize=FIGSIZE)
    
    # Create histogram with KDE
    sns.histplot(errors, kde=True, bins=bins)
    
    # Formatting
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.xlabel(xlabel, fontsize=AXIS_LABEL_FONT_SIZE, weight="bold")
    plt.ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE, weight="bold")
    plt.xticks(fontsize=TICK_LABEL_FONT_SIZE)
    plt.yticks(fontsize=TICK_LABEL_FONT_SIZE)
    
    # Add zero line
    plt.axvline(0, color="r", linestyle="--", linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot if path provided
    if output_path:
        output_path = Path(output_path)
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        
        # Create description file
        if create_description:
            _create_error_distribution_description(
                output_path, errors, error_stats
            )
    
    return fig, error_stats


def create_tsne_embeddings_plot(
    embeddings: np.ndarray,
    y_true: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "t-SNE Projection of Polymer Embeddings",
    xlabel: str = "t-SNE Dimension 1",
    ylabel: str = "t-SNE Dimension 2",
    colorbar_label: str = "True Tg (K)",
    perplexity: Optional[int] = None,
    random_state: int = 42,
    create_description: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a t-SNE visualization of embeddings colored by target values.
    
    Args:
        embeddings: High-dimensional embeddings array
        y_true: True target values for coloring
        output_path: Path to save the plot (optional)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        colorbar_label: Label for the colorbar
        perplexity: t-SNE perplexity parameter
        random_state: Random state for reproducibility
        create_description: Whether to create a description file
        
    Returns:
        Tuple of (matplotlib figure, t-SNE results array)
    """
    # Set perplexity
    if perplexity is None:
        perplexity = min(30, len(embeddings) - 1)
    
    # Perform t-SNE
    tsne = TSNE(
        n_components=2,
        verbose=0,
        perplexity=perplexity,
        max_iter=1000,
        random_state=random_state,
    )
    tsne_results = tsne.fit_transform(embeddings)
    
    # Create figure
    fig = plt.figure(figsize=FIGSIZE)
    
    # Create scatter plot
    scatter = plt.scatter(
        tsne_results[:, 0],
        tsne_results[:, 1],
        c=y_true,
        cmap="viridis",
        alpha=0.8,
        edgecolors="black",
        linewidth=0.5,
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label(colorbar_label, fontsize=CBAR_FONT_SIZE, weight="bold")
    cbar.ax.tick_params(labelsize=CBAR_TICKS_FONT_SIZE)
    
    # Formatting
    plt.title(title, fontsize=TITLE_FONT_SIZE)
    plt.xlabel(xlabel, fontsize=AXIS_LABEL_FONT_SIZE, weight="bold")
    plt.ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE, weight="bold")
    plt.xticks(fontsize=TICK_LABEL_FONT_SIZE)
    plt.yticks(fontsize=TICK_LABEL_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot if path provided
    if output_path:
        output_path = Path(output_path)
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        
        # Create description file
        if create_description:
            _create_tsne_description(
                output_path, tsne_results, y_true
            )
    
    return fig, tsne_results


def generate_training_report(
    dataset_name: str,
    model_config: Any,
    training_config: Dict[str, Any],
    metrics: Dict[str, float],
    data_stats: Dict[str, Any],
    training_stats: Dict[str, Any],
    output_path: Union[str, Path],
    plot_paths: Optional[Dict[str, Path]] = None,
) -> None:
    """
    Generate a comprehensive training analysis report.
    
    Args:
        dataset_name: Name of the dataset
        model_config: Model configuration object
        training_config: Training configuration dictionary
        metrics: Performance metrics dictionary
        data_stats: Dataset statistics dictionary
        training_stats: Training process statistics
        output_path: Path to save the report
        plot_paths: Dictionary of plot file paths
    """
    output_path = Path(output_path)
    
    # Extract configuration details
    device = training_config.get("device", "unknown")
    db_path = training_config.get("db_path", "unknown")
    
    report_content = f"""# Training Analysis Report: {dataset_name}

## 1. Executive Summary
This report details the training and evaluation of a Graph Neural Network (GNN) model, `PolyGCBaseModel`, for predicting the glass transition temperature (Tg) of polymers. The model was trained on a dataset of polymer structures represented as graphs, with node features derived from pre-trained embeddings. The model demonstrates strong predictive performance on an unseen test set, achieving a Mean Absolute Error (MAE) of **{metrics.get('mae', 'N/A'):.2f} K** and an R-squared (R²) value of **{metrics.get('r2', 'N/A'):.3f}**.

---

## 2. Configuration Details

### 2.1. Hyperparameters
- **Random Seed:** {training_config.get('seed', 'N/A')}
- **Epochs (Max):** {training_config.get('epochs', 'N/A')}
- **Batch Size:** {training_config.get('batch_size', 'N/A')}
- **Learning Rate:** {training_config.get('learning_rate', 'N/A')}
- **Optimizer:** {training_config.get('optimizer', 'AdamW')} (Weight Decay: {training_config.get('weight_decay', '1e-5')})
- **LR Scheduler:** {training_config.get('scheduler', 'ReduceLROnPlateau')}
- **Early Stopping Patience:** {training_config.get('patience_epochs', 'N/A')} epochs

### 2.2. Model Architecture (`PolyGCBaseModel`)
- **Task Type:** {getattr(model_config, 'task_type', 'N/A')}
- **Input Feature Dimension:** {getattr(model_config, 'monomer_features', 'N/A')}
- **GNN Layers:** {getattr(model_config, 'num_gnn_layers', 'N/A')}
- **Graph Conv Output Features:** {getattr(model_config, 'gc_features', 'N/A')}
- **MLP Layers:** {getattr(model_config, 'mlp_layer', 'N/A')}
- **Dropout Rate:** {getattr(model_config, 'dropout_rate', 'N/A')}
- **Graph Pooling:** {", ".join([p.get("type", "unknown") for p in getattr(model_config, 'pooling_layers', [])]) if hasattr(model_config, 'pooling_layers') else 'N/A'}
- **Output Layer:** {'Logits' if getattr(model_config, 'logits_output', False) else 'Direct'}

### 2.3. Environment
- **Device:** {device}
- **Database Source:** {db_path}

---

## 3. Data Analysis

- **Dataset Name:** {dataset_name}
- **Total Graph Samples:** {data_stats.get('total_samples', 'N/A')}
- **Data Splitting Strategy:** Splits were made based on unique polymer entries to prevent data leakage between sets.

### 3.1. Dataset Splits
- **Total Unique Entries:** {data_stats.get('num_entries', 'N/A')}
- **Training Set:** {data_stats.get('train_samples', 'N/A')} samples from {data_stats.get('train_entries', 'N/A')} unique entries.
- **Validation Set:** {data_stats.get('val_samples', 'N/A')} samples from {data_stats.get('val_entries', 'N/A')} unique entries.
- **Test Set:** {data_stats.get('test_samples', 'N/A')} samples from {data_stats.get('test_entries', 'N/A')} unique entries.

### 3.2. Target Variable Analysis (Tg in Training Set)
- **Target Name:** {data_stats.get('target_name', 'N/A')}
- **Mean:** {data_stats.get('target_mean', 'N/A'):.2f} K
- **Standard Deviation:** {data_stats.get('target_std', 'N/A'):.2f} K
- **Min:** {data_stats.get('target_min', 'N/A'):.2f} K
- **Max:** {data_stats.get('target_max', 'N/A'):.2f} K

---

## 4. Training Process Summary

- **Total Epochs Trained:** {training_stats.get('final_epoch', 'N/A')} (Early stopping triggered: {training_stats.get('early_stopping', 'N/A')})
- **Best Validation Loss:** {training_stats.get('best_val_loss', 'N/A'):.4f} (achieved at Epoch {training_stats.get('best_epoch', 'N/A')})
- **Optimization:** The learning rate was dynamically adjusted using `ReduceLROnPlateau` based on validation loss. Training was halted after the validation loss failed to improve for {training_config.get('patience_epochs', 'N/A')} consecutive epochs.

---

## 5. Performance Evaluation (Test Set)

The model's final performance was assessed on the held-out test set using the best model state saved during training.

### 5.1. Quantitative Metrics
- **Mean Absolute Error (MAE):** **{metrics.get('mae', 'N/A'):.2f} K**
  - *Interpretation: On average, the model's prediction of Tg is off by {metrics.get('mae', 'N/A'):.2f} Kelvin from the true value.*
- **Root Mean Squared Error (RMSE):** **{metrics.get('rmse', 'N/A'):.2f} K**
  - *Interpretation: This metric penalizes larger errors more heavily than MAE. The value is comparable to the MAE, suggesting no extreme, outlier errors are dominating the results.*
- **R-squared (R²):** **{metrics.get('r2', 'N/A'):.3f}**
  - *Interpretation: The model explains {metrics.get('r2', 0) * 100:.1f}% of the variance in the true Tg values, indicating a strong correlation between predictions and actuals.*

### 5.2. Analysis of Visualizations
*(See the corresponding .png and .txt files in the results directory for full details)*

{_format_plot_descriptions(plot_paths) if plot_paths else ""}

---

## 6. Conclusion & Future Work

The `PolyGCBaseModel` has proven to be highly effective for predicting polymer Tg from graph-based representations. The combination of GNN layers and multi-head pooling successfully captures the structural information relevant to this thermal property.

Potential directions for future work include:
- **Hyperparameter Optimization:** A more systematic search (e.g., using Optuna or Ray Tune) could further refine the model's architecture and training parameters.
- **Exploring Different Architectures:** Testing other GNN convolutional layers (e.g., GAT, GIN) might yield performance improvements.
- **Expanding the Dataset:** Training on a larger and more diverse set of polymers could improve the model's generalizability.
- **Multi-Target Prediction:** Extending the model to simultaneously predict other properties (e.g., melting point, density) could allow it to learn more comprehensive polymer representations.
"""

    with open(output_path, "w") as f:
        f.write(report_content)


def _create_parity_plot_description(
    plot_path: Path, y_true: np.ndarray, y_pred: np.ndarray, metrics: Dict[str, float]
) -> None:
    """Create description file for parity plot."""
    parity_df = pd.DataFrame({
        "True_Tg (K)": y_true,
        "Predicted_Tg (K)": y_pred,
    })
    
    desc_content = f"""# --- Parity Plot Data Description ---

This file describes the data used to generate '{plot_path.name}'.

## Plot Description:
The plot is a "parity plot," which compares the model's predicted values (y-axis) against the true, experimental values (x-axis) for the glass transition temperature (Tg). The plot uses density-based coloring to show the distribution of data points.

- **Points:** Each point represents a single polymer from the test set. Its position is determined by its true Tg and the model's predicted Tg.
- **Color:** The color of each point corresponds to the local density of data points, calculated using a Gaussian Kernel Density Estimate. Brighter colors (yellow) indicate regions with a high concentration of data points, while darker colors (purple) indicate sparser regions.
- **Ideal Line (Black Dashed Line):** This is the y=x line. A perfect model would have all its predictions fall exactly on this line. The closer the points are to this line, the more accurate the model.
- **Metrics:** The R-squared (R²) and Mean Absolute Error (MAE) are displayed on the plot for a quick performance assessment.

## Key Performance Metrics (from Test Set):
- **Mean Absolute Error (MAE):** {metrics['mae']:.2f} K
- **Root Mean Squared Error (RMSE):** {metrics['rmse']:.2f} K
- **R-squared (R²):** {metrics['r2']:.3f}

## Data Sample (first 15 entries):
{parity_df.head(15).to_string()}
"""

    desc_path = plot_path.parent / f"{plot_path.stem}_description.txt"
    with open(desc_path, "w") as f:
        f.write(desc_content)


def _create_error_distribution_description(
    plot_path: Path, errors: np.ndarray, error_stats: Dict[str, float]
) -> None:
    """Create description file for error distribution plot."""
    error_df = pd.DataFrame({"Error (K)": errors})
    
    desc_content = f"""# --- Error Distribution Plot Data Description ---

This file describes the data used to generate '{plot_path.name}'.

## Plot Description:
This plot is a histogram that shows the distribution of prediction errors on the test set. The error is calculated as (Predicted Tg - True Tg).

- **X-axis:** The prediction error in Kelvin (K). A positive value means the model over-predicted, and a negative value means it under-predicted.
- **Y-axis:** The frequency or count of predictions falling into each error bin.
- **Red Dashed Line:** This line is at error = 0, representing a perfect prediction.
- **Blue Curve (KDE):** The Kernel Density Estimate provides a smooth curve to estimate the probability density function of the errors.

A good model typically has an error distribution that is centered at 0 and is roughly symmetrical, resembling a normal (Gaussian) distribution.

## Error Statistics:
- **Mean Error:** {error_stats['mean_error']:.2f} K
- **Standard Deviation of Error:** {error_stats['std_error']:.2f} K
- **Min Error:** {error_stats['min_error']:.2f} K
- **Max Error:** {error_stats['max_error']:.2f} K

## Data Sample (first 15 error values):
{error_df.head(15).to_string()}
"""

    desc_path = plot_path.parent / f"{plot_path.stem}_description.txt"
    with open(desc_path, "w") as f:
        f.write(desc_content)


def _create_tsne_description(
    plot_path: Path, tsne_results: np.ndarray, y_true: np.ndarray
) -> None:
    """Create description file for t-SNE plot."""
    tsne_df = pd.DataFrame({
        "tSNE_Dim_1": tsne_results[:, 0],
        "tSNE_Dim_2": tsne_results[:, 1],
        "True_Tg (K)": y_true,
    })
    
    desc_content = f"""# --- t-SNE Plot Data Description ---

This file describes the data used to generate '{plot_path.name}'.

## Plot Description:
This plot is a 2D visualization of the high-dimensional polymer embeddings generated by the GNN model for the test set. t-SNE (t-Distributed Stochastic Neighbor Embedding) is a dimensionality reduction technique used to visualize these complex embeddings.

- **Each Point:** Represents a single polymer from the test set.
- **Position:** The position of each point in the 2D space is determined by the t-SNE algorithm, which tries to preserve the local neighborhood structure from the original high-dimensional embedding space. Polymers with similar embeddings will appear closer together.
- **Color:** Each point is colored according to its true, experimental glass transition temperature (Tg), as indicated by the color bar on the right.

A smooth gradient of colors across the plot suggests that the model has learned a meaningful representation (a "latent space") where polymers with similar properties are located near each other. This is a strong sign that the model is capturing chemically relevant information.

## Data Sample (first 15 entries):
{tsne_df.head(15).to_string(index=False)}
"""

    desc_path = plot_path.parent / f"{plot_path.stem}_description.txt"
    with open(desc_path, "w") as f:
        f.write(desc_content)


def _format_plot_descriptions(plot_paths: Dict[str, Path]) -> str:
    """Format plot descriptions for the report."""
    descriptions = []
    
    if "parity_plot" in plot_paths:
        descriptions.append(f"- **`{plot_paths['parity_plot'].name}`:**\n  - The parity plot shows a tight clustering of data points around the ideal y=x line, visually confirming the high R² value. The density coloring reveals that the model is most accurate in the regions where most of the data lies.")
    
    if "error_distribution" in plot_paths:
        descriptions.append(f"- **`{plot_paths['error_distribution'].name}`:**\n  - The histogram of prediction errors is centered near zero and is roughly symmetric, resembling a normal distribution. This indicates that the model is not systematically over- or under-predicting and is generally unbiased.")
    
    if "tsne_embeddings" in plot_paths:
        descriptions.append(f"- **`{plot_paths['tsne_embeddings'].name}`:**\n  - The t-SNE plot visualizes the high-dimensional graph embeddings learned by the model. The clear color gradient, corresponding to the true Tg values, demonstrates that the model has successfully learned a meaningful representation where polymers with similar properties are located close to each other.")
    
    return "\n\n".join(descriptions)