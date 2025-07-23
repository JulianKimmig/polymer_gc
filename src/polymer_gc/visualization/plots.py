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
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    property_name: str = "Values",
    unit: str = "",
    show_metrics: bool = True,
) -> Tuple[plt.Figure, Dict[str, float]]:
    """
    Create a parity plot with density coloring.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        output_path: Path to save the plot (optional)
        title: Plot title (optional)
        xlabel: X-axis label (if None, auto-generated from property_name and unit)
        ylabel: Y-axis label (if None, auto-generated from property_name and unit)
        property_name: Name of the property being predicted (e.g., "Tg", "Density")
        unit: Unit of measurement (e.g., "K", "g/cm³")
        show_metrics: Whether to show metrics on the plot
        
    Returns:
        Tuple of (matplotlib figure, metrics dictionary)
    """
    # Auto-generate labels if not provided
    unit_suffix = f" [{unit}]" if unit else ""
    if xlabel is None:
        xlabel = f"True {property_name}{unit_suffix}"
    if ylabel is None:
        ylabel = f"Predicted {property_name}{unit_suffix}"
    
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
        metrics_text = f"R²={r2:.2f}\nMAE={mae:.2f}"
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
        
    
    return fig, metrics


def create_error_distribution_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "Frequency",
    property_name: str = "Values",
    unit: str = "",
    bins: int = 30,
) -> Tuple[plt.Figure, Dict[str, float]]:
    """
    Create an error distribution plot.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        output_path: Path to save the plot (optional)
        title: Plot title (if None, auto-generated from property_name)
        xlabel: X-axis label (if None, auto-generated from unit)
        ylabel: Y-axis label
        property_name: Name of the property being predicted (e.g., "Tg", "Density")
        unit: Unit of measurement (e.g., "K", "g/cm³")
        bins: Number of histogram bins
        
    Returns:
        Tuple of (matplotlib figure, error statistics dictionary)
    """
    # Auto-generate labels if not provided
    unit_suffix = f" [{unit}]" if unit else ""
    if title is None:
        title = f"Distribution of {property_name} Prediction Errors (Predicted - True)"
    if xlabel is None:
        xlabel = f"Error{unit_suffix}"
    
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
        
    
    return fig, error_stats


def create_tsne_embeddings_plot(
    embeddings: np.ndarray,
    y_true: np.ndarray,
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    xlabel: str = "t-SNE Dimension 1",
    ylabel: str = "t-SNE Dimension 2",
    colorbar_label: Optional[str] = None,
    property_name: str = "Values",
    unit: str = "",
    perplexity: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a t-SNE visualization of embeddings colored by target values.
    
    Args:
        embeddings: High-dimensional embeddings array
        y_true: True target values for coloring
        output_path: Path to save the plot (optional)
        title: Plot title (if None, auto-generated from property_name)
        xlabel: X-axis label
        ylabel: Y-axis label
        colorbar_label: Label for the colorbar (if None, auto-generated from property_name and unit)
        property_name: Name of the property being predicted (e.g., "Tg", "Density")
        unit: Unit of measurement (e.g., "K", "g/cm³")
        perplexity: t-SNE perplexity parameter
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (matplotlib figure, t-SNE results array)
    """
    # Auto-generate labels if not provided
    unit_suffix = f" [{unit}]" if unit else ""
    if title is None:
        title = f"t-SNE Projection of {property_name} Embeddings"
    if colorbar_label is None:
        colorbar_label = f"True {property_name}{unit_suffix}"
    
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
        
    
    return fig, tsne_results