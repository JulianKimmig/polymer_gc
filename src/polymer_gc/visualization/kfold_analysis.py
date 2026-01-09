"""
K-Fold cross-validation analysis and visualization functions.

This module provides functions to analyze and visualize results from k-fold
cross-validation training, including ensemble predictions and uncertainty estimation.
"""

from __future__ import annotations
import numpy as np
import torch
from pathlib import Path
from typing import List, Any, Dict, Optional, Tuple, TYPE_CHECKING
import json
from rich import print

if TYPE_CHECKING:
    from polymer_gc.pipelines.evaluation import KFoldEvalResult
    from polymer_gc.pipelines.training import KFoldResult
from .plots import (
    create_parity_plot,
    create_error_distribution_plot, 
    create_embeddings_plot,
)


def create_kfold_ensemble_predictions_from_eval_results(
    kfold_eval_result: KFoldEvalResult,
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
    all_entry_pos = []
    
    # Extract data from evaluation results (test split only for ensemble)
    for fold_result in kfold_eval_result.results:
        if fold_result.model_loaded_successfully:
            # Use test split results for ensemble
            test_result = fold_result.test_result
            
            all_fold_predictions.append(np.array(test_result.y_pred))
            all_fold_embeddings.append(np.array(test_result.embeddings))
            all_test_true.append(np.array(test_result.y_true))
            all_entry_pos.append(np.array(test_result.entry_pos))

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
    all_entry_pos = np.concatenate(all_entry_pos)
    return {
        "y_true": all_true_values,
        "y_pred": all_predictions,
        "embeddings": all_embeddings,
        "entry_pos": all_entry_pos,
        "fold_metrics": fold_metrics,
        "ensemble_size": kfold_eval_result.k_folds,
        "total_test_samples": len(all_predictions)
    }


def create_kfold_visualization_suite(
    kfold_eval_result: KFoldEvalResult,
    dataset_name: str,
    output_dir: Path,
    create_embeddings: bool = True,
    property_name: str = "Value",
    unit: str = "",
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Create comprehensive visualization suite for k-fold cross-validation results.
    
    Args:
        kfold_eval_result: K-fold evaluation results from evaluation pipeline
        dataset_name: Name of the dataset
        output_dir: Directory to save all outputs
        create_embeddings: Whether to create t-SNE/UMAP visualization
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
    entry_pos = ensemble_results["entry_pos"]

    
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
    
    # Ensemble t-SNE/UMAP plot if requested
    if create_embeddings:
        emb_path = output_dir / "kfold_ensemble_embeddings.png"
        emb_fig, emb_results = create_embeddings_plot(
            embeddings=embeddings,
            y_true=y_true,
            entry_pos=entry_pos,
            output_path=emb_path,
            title=f"K-Fold Ensemble {property_name} Embeddings ({dataset_name})",
            property_name=property_name,
            unit=unit
        )
        plt.close(emb_fig)
        ensemble_plot_paths["ensemble_embeddings"] = emb_path
    
    # Create individual fold plots in subfolders
    fold_plot_paths = {}
    for fold_result in kfold_eval_result.results:
        if fold_result.model_loaded_successfully:
            fold_idx = fold_result.fold - 1  # Convert to 0-indexed
            fold_dir = output_dir / f"fold_{fold_result.fold}"
            fold_dir.mkdir(exist_ok=True, parents=True)
            
            fold_paths = {}
            
            # Create plots for each split (train, val, test)
            splits_data = {
                "train": fold_result.train_result,
                "val": fold_result.val_result,
                "test": fold_result.test_result
            }
            
            split_paths = {}
            for split_name, split_result in splits_data.items():
                split_y_true = np.array(split_result.y_true)
                split_y_pred = np.array(split_result.y_pred)
                split_embeddings = np.array(split_result.embeddings)
                split_entry_pos = np.array(split_result.entry_pos)
                split_paths[split_name] = {}
                
                # Split parity plot
                split_parity_path = fold_dir / f"fold_{fold_result.fold}_{split_name}_parity_plot.png"
                split_parity_fig, split_parity_metrics = create_parity_plot(
                    y_true=split_y_true,
                    y_pred=split_y_pred,
                    output_path=split_parity_path,
                    title=f"Fold {fold_result.fold} {split_name.capitalize()} Predictions ({dataset_name})",
                    property_name=property_name,
                    unit=unit
                )
                plt.close(split_parity_fig)
                split_paths[split_name]["parity_plot"] = split_parity_path
                
                # Split error distribution plot
                split_error_path = fold_dir / f"fold_{fold_result.fold}_{split_name}_error_distribution.png"
                split_error_fig, split_error_stats = create_error_distribution_plot(
                    y_true=split_y_true,
                    y_pred=split_y_pred,
                    output_path=split_error_path,
                    title=f"Fold {fold_result.fold} {split_name.capitalize()} {property_name} Prediction Errors ({dataset_name})",
                    property_name=property_name,
                    unit=unit
                )
                plt.close(split_error_fig)
                split_paths[split_name]["error_distribution"] = split_error_path
                
                # Split t-SNE/UMAP plot if requested
                if create_embeddings:
                    split_emb_path = fold_dir / f"fold_{fold_result.fold}_{split_name}_embeddings.png"
                    split_emb_fig, split_emb_results = create_embeddings_plot(
                        embeddings=split_embeddings,
                        y_true=split_y_true,
                        entry_pos=split_entry_pos,
                        output_path=split_emb_path,
                        title=f"Fold {fold_result.fold} {split_name.capitalize()} {property_name} Embeddings ({dataset_name})",
                        property_name=property_name,
                        unit=unit
                    )
                    plt.close(split_emb_fig)
                    split_paths[split_name]["embeddings"] = split_emb_path
            
            # Create combined plots that include ALL splits (train + val + test)
            # Combine data from all splits for this fold
            combined_y_true = []
            combined_y_pred = []
            combined_embeddings = []
            combined_split_labels = []
            combined_entry_pos = []

            for split_name, split_result in splits_data.items():
                split_y_true = np.array(split_result.y_true)
                split_y_pred = np.array(split_result.y_pred)
                split_embeddings = np.array(split_result.embeddings)
                split_entry_pos = np.array(split_result.entry_pos)
                combined_y_true.append(split_y_true)
                combined_y_pred.append(split_y_pred)
                combined_embeddings.append(split_embeddings)
                combined_split_labels.extend([split_name] * len(split_y_true))
                combined_entry_pos.append(split_entry_pos)
            
            # Concatenate all splits
            all_y_true = np.concatenate(combined_y_true)
            all_y_pred = np.concatenate(combined_y_pred)
            all_embeddings = np.concatenate(combined_embeddings, axis=0)
            all_entry_pos = np.concatenate(combined_entry_pos)
            
            # Combined parity plot (all splits)
            fold_parity_path = fold_dir / f"fold_{fold_result.fold}_all_splits_combined_parity_plot.png"
            fold_parity_fig, fold_parity_metrics = create_parity_plot(
                y_true=all_y_true,
                y_pred=all_y_pred,
                output_path=fold_parity_path,
                title=f"Fold {fold_result.fold} All Splits Combined ({dataset_name})",
                property_name=property_name,
                unit=unit
            )
            plt.close(fold_parity_fig)
            fold_paths["all_splits_combined_parity_plot"] = fold_parity_path
            
            # Combined error distribution plot (all splits)
            fold_error_path = fold_dir / f"fold_{fold_result.fold}_all_splits_combined_error_distribution.png"
            fold_error_fig, fold_error_stats = create_error_distribution_plot(
                y_true=all_y_true,
                y_pred=all_y_pred,
                output_path=fold_error_path,
                title=f"Fold {fold_result.fold} All Splits Combined {property_name} Prediction Errors ({dataset_name})",
                property_name=property_name,
                unit=unit
            )
            plt.close(fold_error_fig)
            fold_paths["all_splits_combined_error_distribution"] = fold_error_path
            
            # Combined t-SNE/UMAP plot if requested (all splits)
            if create_embeddings:
                fold_emb_path = fold_dir / f"fold_{fold_result.fold}_all_splits_combined_embeddings.png"
                fold_emb_fig, fold_emb_results = create_embeddings_plot(
                    embeddings=all_embeddings,
                    y_true=all_y_true,
                    entry_pos=all_entry_pos,
                    output_path=fold_emb_path,
                    title=f"Fold {fold_result.fold} All Splits Combined {property_name} Embeddings ({dataset_name})",
                    property_name=property_name,
                    unit=unit
                )
                plt.close(fold_emb_fig)
                fold_paths["all_splits_combined_embeddings"] = fold_emb_path
            
            # Store both split-specific and combined paths
            fold_paths["splits"] = split_paths
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
        "metrics_path": kfold_metrics_path,
        "ensemble_results": ensemble_results,
        "parity_metrics": parity_metrics,
        "detailed_metrics": detailed_metrics,
    }
    
    print(f"K-fold visualization suite completed!")
    print(f"Ensemble plots: {list(ensemble_plot_paths.values())}")
    print(f"Individual fold plots created in subfolders")
    print(f"Ensemble MAE: {parity_metrics['mae']:.3f} ± {kfold_eval_result.test_mae.std:.3f}")
    print(f"Ensemble R²: {parity_metrics['r2']:.3f} ± {kfold_eval_result.test_r2.std:.3f}")
    
    return results
