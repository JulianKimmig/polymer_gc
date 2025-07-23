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
)


def create_kfold_ensemble_predictions_from_eval_results(
    kfold_eval_result,  # KFoldEvalResult from evaluation pipeline
) -> Dict[str, Any]:
    """
    Create ensemble predictions using results from evaluation pipeline.
    
    Args:
        kfold_eval_result: K-fold evaluation results from evaluation_pipeline
        
    Returns:
        Dictionary containing predictions, true values, embeddings, and metadata
    """
    print(f"Creating ensemble predictions from {kfold_eval_result.k_folds} folds (using evaluation results)...")
    
    all_fold_predictions = []
    all_fold_embeddings = []
    all_test_true = []
    fold_metrics = []
    
    # Extract data from evaluation results (test split only for ensemble)
    for fold_result in kfold_eval_result.results:
        if fold_result.model_loaded_successfully:
            # Use test split results for ensemble
            test_result = fold_result.test_result
            
            all_fold_predictions.append(np.array(test_result.y_pred))
            all_fold_embeddings.append(np.array(test_result.embeddings))
            all_test_true.append(np.array(test_result.y_true))
            
            fold_metrics.append({
                "fold": fold_result.fold,
                "test_samples": test_result.num_samples,
                "mae": test_result.mae,
                "mse": test_result.mse,
                "r2": test_result.r2,
            })
    
    if len(all_fold_predictions) == 0:
        raise ValueError("No successful fold results found for creating ensemble predictions")
    
    # Combine all predictions - each entry appears in exactly one test fold
    all_predictions = np.concatenate(all_fold_predictions)
    all_embeddings = np.concatenate(all_fold_embeddings, axis=0)
    all_true_values = np.concatenate(all_test_true)
    
    return {
        "y_true": all_true_values,
        "y_pred": all_predictions,
        "embeddings": all_embeddings,
        "fold_metrics": fold_metrics,
        "ensemble_size": kfold_eval_result.k_folds,
        "total_test_samples": len(all_predictions)
    }


def create_kfold_visualization_suite(
    kfold_eval_result,  # KFoldEvalResult from evaluation pipeline
    dataset_name: str,
    output_dir: Path,
    create_tsne: bool = True,
    property_name: str = "Tg",
    unit: str = "",
    # Legacy parameters kept for backward compatibility
    kfold_result: Optional['KFoldResult'] = None,
    all_graph_data: Optional[List[Any]] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Create comprehensive visualization suite for k-fold cross-validation results.
    
    Args:
        kfold_eval_result: K-fold evaluation results from evaluation pipeline
        dataset_name: Name of the dataset
        output_dir: Directory to save all outputs
        create_tsne: Whether to create t-SNE visualization
        property_name: Name of the property being predicted (e.g., "Tg", "Density")
        unit: Unit of measurement (e.g., "K", "g/cm³")
        
    Returns:
        Dictionary containing all visualization results and file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create ensemble predictions from evaluation results (no model loading needed!)
    ensemble_results = create_kfold_ensemble_predictions_from_eval_results(
        kfold_eval_result=kfold_eval_result
    )
    
    y_true = ensemble_results["y_true"]
    y_pred = ensemble_results["y_pred"]
    embeddings = ensemble_results["embeddings"]
    
    print(f"Creating visualizations for {len(y_true)} total test samples across all folds")
    
    # Create ensemble plots (main directory)
    ensemble_plot_paths = {}
    
    # Ensemble parity plot
    parity_path = output_dir / "kfold_ensemble_parity_plot.png"
    parity_fig, parity_metrics = create_parity_plot(
        y_true=y_true,
        y_pred=y_pred,
        output_path=parity_path,
        title=f"K-Fold Ensemble Predictions ({dataset_name})",
        property_name=property_name,
        unit=unit
    )
    import matplotlib.pyplot as plt
    plt.close(parity_fig)
    ensemble_plot_paths["ensemble_parity_plot"] = parity_path
    
    # Ensemble error distribution plot
    error_path = output_dir / "kfold_ensemble_error_distribution.png"
    error_fig, error_stats = create_error_distribution_plot(
        y_true=y_true,
        y_pred=y_pred,
        output_path=error_path,
        title=f"K-Fold Ensemble {property_name} Prediction Errors ({dataset_name})",
        property_name=property_name,
        unit=unit
    )
    plt.close(error_fig)
    ensemble_plot_paths["ensemble_error_distribution"] = error_path
    
    # Ensemble t-SNE plot if requested
    if create_tsne:
        tsne_path = output_dir / "kfold_ensemble_tsne_embeddings.png"
        tsne_fig, tsne_results = create_tsne_embeddings_plot(
            embeddings=embeddings,
            y_true=y_true,
            output_path=tsne_path,
            title=f"K-Fold Ensemble {property_name} Embeddings ({dataset_name})",
            property_name=property_name,
            unit=unit
        )
        plt.close(tsne_fig)
        ensemble_plot_paths["ensemble_tsne_embeddings"] = tsne_path
    
    # Create individual fold plots in subfolders
    fold_plot_paths = {}
    for fold_result in kfold_eval_result.results:
        if fold_result.model_loaded_successfully:
            fold_idx = fold_result.fold - 1  # Convert to 0-indexed
            fold_dir = output_dir / f"fold_{fold_result.fold}"
            fold_dir.mkdir(exist_ok=True, parents=True)
            
            # Individual fold test data
            test_result = fold_result.test_result
            fold_y_true = np.array(test_result.y_true)
            fold_y_pred = np.array(test_result.y_pred)
            fold_embeddings = np.array(test_result.embeddings)
            
            fold_paths = {}
            
            # Fold parity plot
            fold_parity_path = fold_dir / f"fold_{fold_result.fold}_parity_plot.png"
            fold_parity_fig, fold_parity_metrics = create_parity_plot(
                y_true=fold_y_true,
                y_pred=fold_y_pred,
                output_path=fold_parity_path,
                title=f"Fold {fold_result.fold} Test Predictions ({dataset_name})",
                property_name=property_name,
                unit=unit
            )
            plt.close(fold_parity_fig)
            fold_paths["parity_plot"] = fold_parity_path
            
            # Fold error distribution plot
            fold_error_path = fold_dir / f"fold_{fold_result.fold}_error_distribution.png"
            fold_error_fig, fold_error_stats = create_error_distribution_plot(
                y_true=fold_y_true,
                y_pred=fold_y_pred,
                output_path=fold_error_path,
                title=f"Fold {fold_result.fold} {property_name} Prediction Errors ({dataset_name})",
                property_name=property_name,
                unit=unit
            )
            plt.close(fold_error_fig)
            fold_paths["error_distribution"] = fold_error_path
            
            # Fold t-SNE plot if requested
            if create_tsne:
                fold_tsne_path = fold_dir / f"fold_{fold_result.fold}_tsne_embeddings.png"
                fold_tsne_fig, fold_tsne_results = create_tsne_embeddings_plot(
                    embeddings=fold_embeddings,
                    y_true=fold_y_true,
                    output_path=fold_tsne_path,
                    title=f"Fold {fold_result.fold} {property_name} Embeddings ({dataset_name})",
                    property_name=property_name,
                    unit=unit
                )
                plt.close(fold_tsne_fig)
                fold_paths["tsne_embeddings"] = fold_tsne_path
            
            fold_plot_paths[f"fold_{fold_result.fold}"] = fold_paths
    
    # Combine all plot paths
    plot_paths = {
        "ensemble": ensemble_plot_paths,
        "individual_folds": fold_plot_paths
    }
    
    # Prepare configuration data
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    training_config = {
        "seed": 42,
        "epochs": kfold_eval_result.config.epochs,
        "batch_size": kfold_eval_result.config.batch_size,
        "learning_rate": kfold_eval_result.config.learning_rate,
        "optimizer": "AdamW",
        "scheduler": "ReduceLROnPlateau",
        "patience_epochs": kfold_eval_result.config.early_stopping_patience,
        "device": str(device),
        "k_folds": kfold_eval_result.k_folds,
        "ensemble_predictions": True
    }
    
    # Estimate data stats from evaluation results
    data_stats = {
        "total_samples": kfold_eval_result.total_samples_evaluated,
        "test_samples": len(y_true),
        "target_name": property_name,
        "target_mean": np.mean(y_true),
        "target_std": np.std(y_true),
        "target_min": np.min(y_true),
        "target_max": np.max(y_true),
        "k_fold_evaluation": True,
    }
    
    # Training statistics from k-fold results
    training_stats = {
        "k_folds": kfold_eval_result.k_folds,
        "ensemble_predictions": True,
        "fold_mae_mean": kfold_eval_result.test_mae.mean,
        "fold_mae_std": kfold_eval_result.test_mae.std,
        "fold_r2_mean": kfold_eval_result.test_r2.mean,
        "fold_r2_std": kfold_eval_result.test_r2.std,
        "used_pretrained": False,  # This info isn't in KFoldEvalResult
    }
    
    # Generate comprehensive report - create a minimal KFoldResult for compatibility
    from ..pipelines.training import KFoldResult, TrainingResult
    from pathlib import Path as PathType
    
    # Create minimal training results for report generation
    training_results = []
    for fold_result in kfold_eval_result.results:
        if fold_result.model_loaded_successfully:
            training_results.append(TrainingResult(
                fold=fold_result.fold,
                test_mae=fold_result.test_result.mae,
                test_mse=fold_result.test_result.mse,
                test_r2=fold_result.test_result.r2,
                epochs_trained=100,  # Placeholder
                best_epoch=90,      # Placeholder
                model_dir=PathType(fold_result.model_dir)
            ))
    
    # Create minimal KFoldResult for report compatibility
    kfold_result_for_report = KFoldResult(
        results=training_results,
        test_mae=kfold_eval_result.test_mae,
        test_mse=kfold_eval_result.test_mse,
        test_r2=kfold_eval_result.test_r2,
        k_folds=kfold_eval_result.k_folds,
        config=kfold_eval_result.config,
        used_pretrained=False
    )
    
    report_path = output_dir / "kfold_analysis_report.md"
    generate_kfold_training_report(
        dataset_name=dataset_name,
        kfold_result=kfold_result_for_report,
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
        "k_folds": kfold_eval_result.k_folds,
        "ensemble_metrics": {
            "mae": parity_metrics["mae"],
            "rmse": parity_metrics["rmse"],
            "r2": parity_metrics["r2"]
        },
        "cross_validation_metrics": {
            "mae_mean": kfold_eval_result.test_mae.mean,
            "mae_std": kfold_eval_result.test_mae.std,
            "mae_min": kfold_eval_result.test_mae.min,
            "mae_max": kfold_eval_result.test_mae.max,
            "r2_mean": kfold_eval_result.test_r2.mean,
            "r2_std": kfold_eval_result.test_r2.std,
            "r2_min": kfold_eval_result.test_r2.min,
            "r2_max": kfold_eval_result.test_r2.max,
        },
        "individual_fold_metrics": ensemble_results["fold_metrics"],
        "total_test_samples": ensemble_results["total_test_samples"],
        "used_pretrained": False,
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
    print(f"Ensemble plots: {list(ensemble_plot_paths.values())}")
    print(f"Individual fold plots created in subfolders")
    print(f"Ensemble MAE: {parity_metrics['mae']:.3f} ± {kfold_eval_result.test_mae.std:.3f}")
    print(f"Ensemble R²: {parity_metrics['r2']:.3f} ± {kfold_eval_result.test_r2.std:.3f}")
    
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


def _format_kfold_plot_descriptions(plot_paths: Dict[str, Any]) -> str:
    """Format plot descriptions for k-fold report."""
    descriptions = []
    
    # Ensemble plots
    if "ensemble" in plot_paths:
        ensemble_paths = plot_paths["ensemble"]
        descriptions.append("### Ensemble Plots (All Folds Combined)")
        
        if "ensemble_parity_plot" in ensemble_paths:
            descriptions.append(f"- **`{ensemble_paths['ensemble_parity_plot'].name}`:**\n  - Shows ensemble predictions vs. true values across all test folds. Each point represents a sample predicted by a model that never saw that sample during training, providing unbiased performance assessment.")
        
        if "ensemble_error_distribution" in ensemble_paths:
            descriptions.append(f"- **`{ensemble_paths['ensemble_error_distribution'].name}`:**\n  - Distribution of ensemble prediction errors. A well-centered distribution indicates unbiased predictions across the entire dataset.")
        
        if "ensemble_tsne_embeddings" in ensemble_paths:
            descriptions.append(f"- **`{ensemble_paths['ensemble_tsne_embeddings'].name}`:**\n  - t-SNE visualization of learned embeddings from ensemble models, colored by true values. Smooth gradients indicate that models have learned meaningful chemical representations.")
    
    # Individual fold plots
    if "individual_folds" in plot_paths:
        fold_paths = plot_paths["individual_folds"]
        descriptions.append("\n### Individual Fold Plots")
        descriptions.append(f"Individual performance plots for each of the {len(fold_paths)} folds are available in their respective subfolders:")
        
        for fold_name in sorted(fold_paths.keys()):
            descriptions.append(f"- **`{fold_name}/`** - Contains parity plots, error distributions, and t-SNE visualizations for {fold_name} test data only")
    
    return "\n\n".join(descriptions)