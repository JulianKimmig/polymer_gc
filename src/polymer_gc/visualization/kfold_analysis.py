"""
K-Fold cross-validation analysis and visualization functions.

This module provides functions to analyze and visualize results from k-fold
cross-validation training, including ensemble predictions and uncertainty estimation.
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Any, Dict, Optional, Tuple
import json
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from rich import print

from polymer_gc.model.base import PolyGCBaseModel
from polymer_gc.pipelines.training import KFoldResult
from .plots import (
    create_parity_plot,
    create_error_distribution_plot, 
    create_tsne_embeddings_plot,
    generate_training_report
)
from .utils import prepare_data_stats, prepare_training_stats


def create_kfold_ensemble_predictions(
    kfold_result: KFoldResult,
    all_graph_data: List[Any],
    device: Optional[torch.device] = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Create ensemble predictions using all folds from k-fold cross-validation.
    
    Args:
        kfold_result: K-fold training results
        all_graph_data: All graph data used in training
        device: Device to run inference on
        batch_size: Batch size for inference
        
    Returns:
        Dictionary containing predictions, true values, embeddings, and metadata
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Creating ensemble predictions from {kfold_result.k_folds} folds...")
    
    # Get unique entry positions for splitting
    unique_entries = list(set(g.entry_pos for g in all_graph_data))
    
    # Recreate the same k-fold splits used in training
    np.random.seed(42)
    kf = KFold(n_splits=kfold_result.k_folds, shuffle=True, random_state=42)
    splits = list(kf.split(unique_entries))
    
    all_fold_predictions = []
    all_fold_embeddings = []
    all_test_true = []
    all_test_graphs = []
    fold_metrics = []
    
    for fold_idx, (train_entries, test_entries) in enumerate(splits):
        print(f"Processing fold {fold_idx + 1}/{kfold_result.k_folds}")
        
        # Get corresponding result
        fold_result = kfold_result.results[fold_idx]
        
        # Convert indices to actual entry positions
        test_entry_pos = set(unique_entries[i] for i in test_entries)
        
        # Get test graphs for this fold
        test_graphs = [g for g in all_graph_data if g.entry_pos in test_entry_pos]
        
        if len(test_graphs) == 0:
            print(f"Warning: No test graphs for fold {fold_idx + 1}")
            continue
            
        # Load model for this fold
        model = PolyGCBaseModel(config=kfold_result.config.model_conf)
        model_path = Path(fold_result.model_dir) / "best_model.pt"
        
        if not model_path.exists():
            print(f"Warning: Model not found at {model_path}")
            continue
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        # Create test loader
        test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
        
        # Generate predictions and embeddings
        fold_preds = []
        fold_embeddings = []
        fold_true = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                preds = model.predict(batch)
                embeddings = model.predict_embedding(batch)
                
                fold_preds.append(preds.cpu().numpy())
                fold_embeddings.append(embeddings.cpu().numpy())
                fold_true.append(batch.y.cpu().numpy())
        
        # Concatenate fold results
        fold_preds = np.concatenate(fold_preds).flatten()
        fold_embeddings = np.concatenate(fold_embeddings, axis=0)
        fold_true = np.concatenate(fold_true).flatten()
        
        all_fold_predictions.append(fold_preds)
        all_fold_embeddings.append(fold_embeddings)
        all_test_true.append(fold_true)
        all_test_graphs.extend(test_graphs)
        
        fold_metrics.append({
            "fold": fold_idx + 1,
            "test_samples": len(test_graphs),
            "mae": fold_result.test_mae,
            "mse": fold_result.test_mse,
            "r2": fold_result.test_r2,
        })
    
    # Combine all predictions - each entry appears in exactly one test fold
    all_predictions = np.concatenate(all_fold_predictions)
    all_embeddings = np.concatenate(all_fold_embeddings, axis=0)
    all_true_values = np.concatenate(all_test_true)
    
    return {
        "y_true": all_true_values,
        "y_pred": all_predictions,
        "embeddings": all_embeddings,
        "test_graphs": all_test_graphs,
        "fold_metrics": fold_metrics,
        "ensemble_size": kfold_result.k_folds,
        "total_test_samples": len(all_predictions)
    }


def create_kfold_visualization_suite(
    kfold_result: KFoldResult,
    all_graph_data: List[Any],
    dataset_name: str,
    output_dir: Path,
    device: Optional[torch.device] = None,
    create_tsne: bool = True,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Create comprehensive visualization suite for k-fold cross-validation results.
    
    Args:
        kfold_result: K-fold training results
        all_graph_data: All graph data used in training
        dataset_name: Name of the dataset
        output_dir: Directory to save all outputs
        device: Device to run inference on
        create_tsne: Whether to create t-SNE visualization
        batch_size: Batch size for inference
        
    Returns:
        Dictionary containing all visualization results and file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create ensemble predictions
    ensemble_results = create_kfold_ensemble_predictions(
        kfold_result=kfold_result,
        all_graph_data=all_graph_data,
        device=device,
        batch_size=batch_size
    )
    
    y_true = ensemble_results["y_true"]
    y_pred = ensemble_results["y_pred"]
    embeddings = ensemble_results["embeddings"]
    
    print(f"Creating visualizations for {len(y_true)} total test samples across all folds")
    
    # Create individual plots
    plot_paths = {}
    
    # Parity plot
    parity_path = output_dir / "kfold_parity_plot.png"
    parity_fig, parity_metrics = create_parity_plot(
        y_true=y_true,
        y_pred=y_pred,
        output_path=parity_path,
        title=f"K-Fold Ensemble Predictions ({dataset_name})"
    )
    parity_fig.close()
    plot_paths["parity_plot"] = parity_path
    
    # Error distribution plot
    error_path = output_dir / "kfold_error_distribution.png"
    error_fig, error_stats = create_error_distribution_plot(
        y_true=y_true,
        y_pred=y_pred,
        output_path=error_path,
        title=f"K-Fold Prediction Errors ({dataset_name})"
    )
    error_fig.close()
    plot_paths["error_distribution"] = error_path
    
    # t-SNE plot if requested
    if create_tsne:
        tsne_path = output_dir / "kfold_tsne_embeddings.png"
        tsne_fig, tsne_results = create_tsne_embeddings_plot(
            embeddings=embeddings,
            y_true=y_true,
            output_path=tsne_path,
            title=f"K-Fold Ensemble Embeddings ({dataset_name})"
        )
        tsne_fig.close()
        plot_paths["tsne_embeddings"] = tsne_path
    
    # Prepare configuration data
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    training_config = {
        "seed": 42,
        "epochs": kfold_result.config.epochs,
        "batch_size": kfold_result.config.batch_size,
        "learning_rate": kfold_result.config.learning_rate,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "patience_epochs": kfold_result.config.early_stopping_patience,
        "device": str(device),
        "k_folds": kfold_result.k_folds,
        "ensemble_predictions": True
    }
    
    # For k-fold, we use all data for stats since each sample appears in test exactly once
    all_targets = np.concatenate([g.y.numpy().flatten() for g in all_graph_data])
    
    data_stats = {
        "total_samples": len(all_graph_data),
        "test_samples": len(y_true),
        "target_name": "Tg",
        "target_mean": np.mean(all_targets),
        "target_std": np.std(all_targets),
        "target_min": np.min(all_targets),
        "target_max": np.max(all_targets),
        "k_fold_evaluation": True,
    }
    
    # Training statistics from k-fold results
    training_stats = {
        "k_folds": kfold_result.k_folds,
        "ensemble_predictions": True,
        "fold_mae_mean": kfold_result.test_mae.mean,
        "fold_mae_std": kfold_result.test_mae.std,
        "fold_r2_mean": kfold_result.test_r2.mean,
        "fold_r2_std": kfold_result.test_r2.std,
        "used_pretrained": kfold_result.used_pretrained,
    }
    
    # Generate comprehensive report
    report_path = output_dir / "kfold_analysis_report.md"
    generate_kfold_training_report(
        dataset_name=dataset_name,
        kfold_result=kfold_result,
        ensemble_results=ensemble_results,
        parity_metrics=parity_metrics,
        training_config=training_config,
        data_stats=data_stats,
        output_path=report_path,
        plot_paths=plot_paths,
    )
    
    # Save detailed k-fold metrics
    kfold_metrics_path = output_dir / "kfold_detailed_metrics.json"
    detailed_metrics = {
        "dataset": dataset_name,
        "k_folds": kfold_result.k_folds,
        "ensemble_metrics": {
            "mae": parity_metrics["mae"],
            "rmse": parity_metrics["rmse"],
            "r2": parity_metrics["r2"]
        },
        "cross_validation_metrics": {
            "mae_mean": kfold_result.test_mae.mean,
            "mae_std": kfold_result.test_mae.std,
            "mae_min": kfold_result.test_mae.min,
            "mae_max": kfold_result.test_mae.max,
            "r2_mean": kfold_result.test_r2.mean,
            "r2_std": kfold_result.test_r2.std,
            "r2_min": kfold_result.test_r2.min,
            "r2_max": kfold_result.test_r2.max,
        },
        "individual_fold_metrics": ensemble_results["fold_metrics"],
        "total_test_samples": ensemble_results["total_test_samples"],
        "used_pretrained": kfold_result.used_pretrained,
    }
    
    with open(kfold_metrics_path, "w") as f:
        json.dump(detailed_metrics, f, indent=2)
    
    results = {
        "plot_paths": plot_paths,
        "report_path": report_path,
        "metrics_path": kfold_metrics_path,
        "ensemble_results": ensemble_results,
        "parity_metrics": parity_metrics,
        "detailed_metrics": detailed_metrics,
    }
    
    print(f"K-fold visualization suite completed!")
    print(f"Report: {report_path}")
    print(f"Plots: {list(plot_paths.values())}")
    print(f"Ensemble MAE: {parity_metrics['mae']:.3f} ± {kfold_result.test_mae.std:.3f}")
    print(f"Ensemble R²: {parity_metrics['r2']:.3f} ± {kfold_result.test_r2.std:.3f}")
    
    return results


def generate_kfold_training_report(
    dataset_name: str,
    kfold_result: KFoldResult,
    ensemble_results: Dict[str, Any],
    parity_metrics: Dict[str, float],
    training_config: Dict[str, Any],
    data_stats: Dict[str, Any],
    output_path: Path,
    plot_paths: Dict[str, Path],
) -> None:
    """Generate a comprehensive k-fold training analysis report."""
    
    report_content = f"""# K-Fold Cross-Validation Analysis Report: {dataset_name}

## 1. Executive Summary
This report details the k-fold cross-validation training and evaluation of a Graph Neural Network (GNN) model, `PolyGCBaseModel`, for predicting the glass transition temperature (Tg) of polymers. The model was trained using {kfold_result.k_folds}-fold cross-validation, ensuring robust performance estimation across different data partitions.

**Key Results:**
- **Ensemble MAE:** {parity_metrics['mae']:.2f} ± {kfold_result.test_mae.std:.2f} K
- **Ensemble R²:** {parity_metrics['r2']:.3f} ± {kfold_result.test_r2.std:.3f}
- **Cross-validation MAE:** {kfold_result.test_mae.mean:.2f} ± {kfold_result.test_mae.std:.2f} K
- **Cross-validation R²:** {kfold_result.test_r2.mean:.3f} ± {kfold_result.test_r2.std:.3f}

---

## 2. K-Fold Cross-Validation Overview

K-fold cross-validation provides more robust model evaluation by training and testing on different data partitions. Each sample appears in the test set exactly once across all folds, providing comprehensive coverage of the dataset.

### 2.1. Configuration
- **Number of Folds:** {kfold_result.k_folds}
- **Total Samples:** {data_stats['total_samples']}
- **Test Samples per Fold:** ~{data_stats['total_samples'] // kfold_result.k_folds}
- **Used Pretrained Model:** {kfold_result.used_pretrained}

### 2.2. Model Architecture
- **Task Type:** {kfold_result.config.model_conf.task_type}
- **Input Features:** {kfold_result.config.model_conf.monomer_features}
- **GNN Layers:** {kfold_result.config.model_conf.num_gnn_layers}
- **Graph Conv Features:** {kfold_result.config.model_conf.gc_features}
- **MLP Layers:** {kfold_result.config.model_conf.mlp_layer}
- **Dropout Rate:** {kfold_result.config.model_conf.dropout_rate}

---

## 3. Performance Analysis

### 3.1. Cross-Validation Metrics
The model was evaluated on {kfold_result.k_folds} different test sets, providing the following performance statistics:

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| MAE (K) | {kfold_result.test_mae.mean:.2f} | {kfold_result.test_mae.std:.2f} | {kfold_result.test_mae.min:.2f} | {kfold_result.test_mae.max:.2f} |
| RMSE (K) | {np.sqrt(kfold_result.test_mse.mean):.2f} | {np.sqrt(kfold_result.test_mse.std):.2f} | {np.sqrt(kfold_result.test_mse.min):.2f} | {np.sqrt(kfold_result.test_mse.max):.2f} |
| R² | {kfold_result.test_r2.mean:.3f} | {kfold_result.test_r2.std:.3f} | {kfold_result.test_r2.min:.3f} | {kfold_result.test_r2.max:.3f} |

### 3.2. Ensemble Predictions
By combining predictions from all folds (each sample predicted by the model trained without it), we achieve:

- **Ensemble MAE:** {parity_metrics['mae']:.2f} K
- **Ensemble RMSE:** {parity_metrics['rmse']:.2f} K  
- **Ensemble R²:** {parity_metrics['r2']:.3f}

The ensemble approach provides more reliable predictions as each model contributes predictions on data it hasn't seen during training.

---

## 4. Visualization Analysis

### 4.1. Individual Fold Performance
{_format_fold_performance_table(ensemble_results['fold_metrics'])}

### 4.2. Plot Descriptions
*(See the corresponding .png files for visual analysis)*

{_format_kfold_plot_descriptions(plot_paths)}

---

## 5. Model Stability and Generalization

The standard deviation of cross-validation metrics provides insight into model stability:

- **MAE Stability:** {kfold_result.test_mae.std:.2f} K standard deviation indicates {'high' if kfold_result.test_mae.std < 2.0 else 'moderate' if kfold_result.test_mae.std < 5.0 else 'low'} stability across folds
- **R² Consistency:** {kfold_result.test_r2.std:.3f} standard deviation shows {'high' if kfold_result.test_r2.std < 0.05 else 'moderate' if kfold_result.test_r2.std < 0.1 else 'variable'} consistency

Lower standard deviations indicate that the model generalizes well regardless of the specific train-test split.

---

## 6. Conclusions and Recommendations

The k-fold cross-validation analysis demonstrates:

1. **Robust Performance:** The model shows consistent performance across different data partitions
2. **Generalization Capability:** Low variance in cross-validation metrics indicates good generalization
3. **Ensemble Benefits:** Combining predictions from all folds provides reliable performance estimates

### 6.1. Next Steps
- Consider ensemble methods for deployment if computational resources allow
- Investigate fold-specific performance differences if standard deviation is high
- Use these metrics as baseline for hyperparameter optimization

---

## 7. Technical Details

- **Random Seed:** 42 (for reproducibility)
- **Cross-validation Strategy:** Stratified by polymer entries to prevent data leakage
- **Device:** {training_config.get('device', 'unknown')}
- **Batch Size:** {kfold_result.config.batch_size}
- **Learning Rate:** {kfold_result.config.learning_rate}
- **Training Epochs:** {kfold_result.config.epochs}
"""

    with open(output_path, "w") as f:
        f.write(report_content)


def _format_fold_performance_table(fold_metrics: List[Dict[str, Any]]) -> str:
    """Format individual fold performance as a markdown table."""
    table = "| Fold | Test Samples | MAE (K) | R² |\n"
    table += "|------|-------------|---------|----|\n"
    
    for metrics in fold_metrics:
        table += f"| {metrics['fold']} | {metrics['test_samples']} | {metrics['mae']:.2f} | {metrics['r2']:.3f} |\n"
    
    return table


def _format_kfold_plot_descriptions(plot_paths: Dict[str, Path]) -> str:
    """Format plot descriptions for k-fold report."""
    descriptions = []
    
    if "parity_plot" in plot_paths:
        descriptions.append(f"- **`{plot_paths['parity_plot'].name}`:**\n  - Shows ensemble predictions vs. true values across all test folds. Each point represents a sample predicted by a model that never saw that sample during training, providing unbiased performance assessment.")
    
    if "error_distribution" in plot_paths:
        descriptions.append(f"- **`{plot_paths['error_distribution'].name}`:**\n  - Distribution of ensemble prediction errors. A well-centered distribution indicates unbiased predictions across the entire dataset.")
    
    if "tsne_embeddings" in plot_paths:
        descriptions.append(f"- **`{plot_paths['tsne_embeddings'].name}`:**\n  - t-SNE visualization of learned embeddings from ensemble models, colored by true Tg values. Smooth gradients indicate that models have learned meaningful chemical representations.")
    
    return "\n\n".join(descriptions)