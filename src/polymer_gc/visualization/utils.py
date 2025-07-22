"""
Utility functions for visualization workflows.

This module provides helper functions for common visualization workflows
and data preparation tasks.
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .plots import (
    create_parity_plot,
    create_error_distribution_plot,
    create_tsne_embeddings_plot,
    generate_training_report,
)


def create_full_evaluation_suite(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    embeddings: np.ndarray,
    output_dir: Path,
    dataset_name: str = "dataset",
    model_config: Any = None,
    training_config: Dict[str, Any] = None,
    data_stats: Dict[str, Any] = None,
    training_stats: Dict[str, Any] = None,
    create_tsne: bool = True,
) -> Dict[str, Any]:
    """
    Create a complete evaluation suite with all visualization plots and report.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        embeddings: Model embeddings for t-SNE visualization
        output_dir: Directory to save all outputs
        dataset_name: Name of the dataset
        model_config: Model configuration object
        training_config: Training configuration dictionary
        data_stats: Dataset statistics dictionary
        training_stats: Training process statistics
        create_tsne: Whether to create t-SNE plot (can be slow for large datasets)
        
    Returns:
        Dictionary containing all results and file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    plot_paths = {}
    
    # Create parity plot
    parity_path = output_dir / "parity_plot_with_density.png"
    parity_fig, parity_metrics = create_parity_plot(
        y_true=y_true,
        y_pred=y_pred,
        output_path=parity_path
    )
    parity_fig.close()
    
    results["parity_metrics"] = parity_metrics
    plot_paths["parity_plot"] = parity_path
    
    # Create error distribution plot
    error_path = output_dir / "error_distribution.png"
    error_fig, error_stats = create_error_distribution_plot(
        y_true=y_true,
        y_pred=y_pred,
        output_path=error_path
    )
    error_fig.close()
    
    results["error_stats"] = error_stats
    plot_paths["error_distribution"] = error_path
    
    # Create t-SNE plot if requested
    if create_tsne and embeddings is not None:
        tsne_path = output_dir / "tsne_embeddings_by_tg.png"
        tsne_fig, tsne_results = create_tsne_embeddings_plot(
            embeddings=embeddings,
            y_true=y_true,
            output_path=tsne_path
        )
        tsne_fig.close()
        
        results["tsne_results"] = tsne_results
        plot_paths["tsne_embeddings"] = tsne_path
    
    # Generate comprehensive report
    report_path = output_dir / "training_analysis_report.md"
    
    # Use provided configs or create defaults
    training_config = training_config or {}
    data_stats = data_stats or {}
    training_stats = training_stats or {}
    
    generate_training_report(
        dataset_name=dataset_name,
        model_config=model_config,
        training_config=training_config,
        metrics=parity_metrics,
        data_stats=data_stats,
        training_stats=training_stats,
        output_path=report_path,
        plot_paths=plot_paths,
    )
    
    results["plot_paths"] = plot_paths
    results["report_path"] = report_path
    
    return results


def calculate_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of evaluation metrics
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mean_error": np.mean(y_pred - y_true),
        "std_error": np.std(y_pred - y_true),
        "min_error": np.min(y_pred - y_true),
        "max_error": np.max(y_pred - y_true),
    }


def prepare_data_stats(
    train_graphs: List[Any],
    val_graphs: List[Any],
    test_graphs: List[Any],
    target_name: str = "Tg",
) -> Dict[str, Any]:
    """
    Prepare data statistics dictionary for reporting.
    
    Args:
        train_graphs: Training graph data objects
        val_graphs: Validation graph data objects
        test_graphs: Test graph data objects
        target_name: Name of the target variable
        
    Returns:
        Dictionary of data statistics
    """
    # Extract target values from training set
    train_targets = np.concatenate([g.y.numpy().flatten() for g in train_graphs])
    
    # Count unique entries
    train_entries = len(set(g.entry_pos for g in train_graphs))
    val_entries = len(set(g.entry_pos for g in val_graphs))
    test_entries = len(set(g.entry_pos for g in test_graphs))
    
    return {
        "total_samples": len(train_graphs) + len(val_graphs) + len(test_graphs),
        "num_entries": train_entries + val_entries + test_entries,
        "train_samples": len(train_graphs),
        "val_samples": len(val_graphs),
        "test_samples": len(test_graphs),
        "train_entries": train_entries,
        "val_entries": val_entries,
        "test_entries": test_entries,
        "target_name": target_name,
        "target_mean": np.mean(train_targets),
        "target_std": np.std(train_targets),
        "target_min": np.min(train_targets),
        "target_max": np.max(train_targets),
    }


def prepare_training_stats(
    final_epoch: int,
    best_val_loss: float,
    best_epoch: int,
    early_stopping: bool = False,
) -> Dict[str, Any]:
    """
    Prepare training statistics dictionary for reporting.
    
    Args:
        final_epoch: Final training epoch reached
        best_val_loss: Best validation loss achieved
        best_epoch: Epoch where best validation loss was achieved
        early_stopping: Whether early stopping was triggered
        
    Returns:
        Dictionary of training statistics
    """
    return {
        "final_epoch": final_epoch,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "early_stopping": early_stopping,
    }