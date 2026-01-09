"""
Example usage of the polymer_gc visualization module.

This script demonstrates how to use the visualization functions with
typical model evaluation data.
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Any

from polymer_gc.visualization import (
    create_full_evaluation_suite,
    create_parity_plot,
    create_error_distribution_plot,
    create_embeddings_plot,
    calculate_comprehensive_metrics,
    prepare_data_stats,
    prepare_training_stats,
)


def example_with_synthetic_data():
    """Example using synthetic data to demonstrate the visualization functions."""
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Synthetic true values (Tg temperatures in Kelvin)
    y_true = np.random.normal(300, 50, n_samples)
    
    # Synthetic predictions with some noise
    y_pred = y_true + np.random.normal(0, 10, n_samples)
    
    # Synthetic high-dimensional embeddings
    embeddings = np.random.normal(0, 1, (n_samples, 128))
    
    # Create output directory
    output_dir = Path("./visualization_example_output")
    output_dir.mkdir(exist_ok=True)
    
    # Example 1: Create individual plots
    print("Creating individual plots...")
    
    # Parity plot
    parity_fig, parity_metrics = create_parity_plot(
        y_true=y_true,
        y_pred=y_pred,
        output_path=output_dir / "example_parity_plot.png",
        xlabel="True Tg (K)",
        ylabel="Predicted Tg (K)"
    )
    parity_fig.close()
    print(f"Parity plot metrics: {parity_metrics}")
    
    # Error distribution plot
    error_fig, error_stats = create_error_distribution_plot(
        y_true=y_true,
        y_pred=y_pred,
        output_path=output_dir / "example_error_distribution.png"
    )
    error_fig.close()
    print(f"Error statistics: {error_stats}")
    
    # t-SNE plot (optional, can be slow)
    emb_fig, emb_results = create_embeddings_plot(
        embeddings=embeddings,
        y_true=y_true,
        output_path=output_dir / "example_embeddings_plot.png",
        colorbar_label="True Tg (K)"
    )
    emb_fig.close()
    print(f"t-SNE results shape: {emb_results["tsne"].shape}")
    print(f"UMAP results shape: {emb_results["umap"].shape}")
    
    # Example 2: Create full evaluation suite
    print("\\nCreating full evaluation suite...")
    
    # Prepare configuration data
    training_config = {
        "seed": 42,
        "epochs": 100,
        "batch_size": 64,
        "learning_rate": 0.001,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "patience_epochs": 20,
        "device": "cpu",
        "db_path": "example_database.db"
    }
    
    data_stats = {
        "total_samples": 1000,
        "num_entries": 200,
        "train_samples": 600,
        "val_samples": 200,
        "test_samples": 200,
        "train_entries": 120,
        "val_entries": 40,
        "test_entries": 40,
        "target_name": "Tg",
        "target_mean": 300.0,
        "target_std": 50.0,
        "target_min": 200.0,
        "target_max": 400.0,
    }
    
    training_stats = prepare_training_stats(
        final_epoch=85,
        best_val_loss=0.15,
        best_epoch=75,
        early_stopping=True
    )
    
    # Create full suite
    results = create_full_evaluation_suite(
        y_true=y_true,
        y_pred=y_pred,
        embeddings=embeddings,
        output_dir=output_dir / "full_suite",
        dataset_name="Synthetic_Tg_Dataset",
        training_config=training_config,
        data_stats=data_stats,
        training_stats=training_stats,
        create_embeddings=True
    )
    
    print(f"Full evaluation suite created at: {results['report_path']}")
    print(f"Plot files: {list(results['plot_paths'].values())}")
    
    # Example 3: Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(y_true, y_pred)
    print(f"\\nComprehensive metrics: {metrics}")
    
    return results


def example_with_model_data(
    train_graphs: List[Any],
    val_graphs: List[Any], 
    test_graphs: List[Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    embeddings: np.ndarray,
    model_config: Any,
    output_dir: Path,
):
    """
    Example using real model training data.
    
    This function demonstrates how to use the visualization functions
    with actual training results from runs/tg_bj_polybert_optim.py
    
    Args:
        train_graphs: Training graph data objects
        val_graphs: Validation graph data objects
        test_graphs: Test graph data objects
        y_true: True target values from test set
        y_pred: Model predictions on test set
        embeddings: Model embeddings from test set
        model_config: Model configuration object
        output_dir: Directory to save visualization outputs
    """
    
    # Prepare data statistics
    data_stats = prepare_data_stats(
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        test_graphs=test_graphs,
        target_name="Tg"
    )
    
    # Prepare training configuration
    training_config = {
        "seed": 42,
        "epochs": 150,
        "batch_size": 32,
        "learning_rate": 0.0005,
        "optimizer": "AdamW",
        "weight_decay": 1e-5,
        "scheduler": "ReduceLROnPlateau",
        "patience_epochs": 40,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "db_path": "database.db"
    }
    
    # Example training statistics (you would get these from actual training)
    training_stats = prepare_training_stats(
        final_epoch=120,
        best_val_loss=0.25,
        best_epoch=110,
        early_stopping=True
    )
    
    # Create full evaluation suite
    results = create_full_evaluation_suite(
        y_true=y_true,
        y_pred=y_pred,
        embeddings=embeddings,
        output_dir=output_dir,
        dataset_name="tg_bayreuth_jena",
        model_config=model_config,
        training_config=training_config,
        data_stats=data_stats,
        training_stats=training_stats,
        create_embeddings=True
    )
    
    print(f"Visualization suite created at: {results['report_path']}")
    return results


if __name__ == "__main__":
    # Run the synthetic data example
    example_with_synthetic_data()
    print("\\nExample completed! Check the 'visualization_example_output' directory for results.")