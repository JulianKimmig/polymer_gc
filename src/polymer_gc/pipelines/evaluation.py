"""
Evaluation pipeline for loading trained models and making predictions.

This module provides functions to evaluate previously trained models using
the same k-fold cross-validation splits as training, returning predictions
and true values for each fold and split (train/val/test).
"""

import json
from typing import Any, Dict, Optional, Tuple, Union, List
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from rich import print

from polymer_gc.data.database import SessionManager
from polymer_gc.data.dataset import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pydantic import BaseModel, Field, model_validator, PrivateAttr
from polymer_gc.model.base import PolyGCBaseModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .training import (
    TrainingConf, 
    KFoldMetrics, 
    load_data, 
    get_kfold_splits, 
    create_fold_dataloaders
)
from ..visualization.kfold_analysis import create_kfold_visualization_suite


class SplitEvalResult(BaseModel):
    """Evaluation results for a single data split (train/val/test)."""
    split_name: str  # "train", "val", or "test"
    mae: float
    mse: float
    r2: float
    num_samples: int
    
    # Large arrays stored as private attributes (not serialized to JSON)
    _y_true: List[float] = PrivateAttr()
    _y_pred: List[float] = PrivateAttr()
    _embeddings: List[List[float]] = PrivateAttr()
    
    def __init__(self, y_true: List[float], y_pred: List[float], embeddings: List[List[float]], **data):
        super().__init__(**data)
        self._y_true = y_true
        self._y_pred = y_pred
        self._embeddings = embeddings
    
    @property
    def y_true(self) -> List[float]:
        return self._y_true
    
    @property
    def y_pred(self) -> List[float]:
        return self._y_pred
    
    @property
    def embeddings(self) -> List[List[float]]:
        return self._embeddings
    
    def save_arrays(self, output_dir: Path, fold_id: str = "") -> Dict[str, Path]:
        """Save large arrays as separate numpy files."""
        import numpy as np
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        prefix = f"{fold_id}_{self.split_name}" if fold_id else self.split_name
        
        # Save arrays
        y_true_path = output_dir / f"{prefix}_y_true.npy"
        y_pred_path = output_dir / f"{prefix}_y_pred.npy"
        embeddings_path = output_dir / f"{prefix}_embeddings.npy"
        
        np.save(y_true_path, np.array(self._y_true))
        np.save(y_pred_path, np.array(self._y_pred))
        np.save(embeddings_path, np.array(self._embeddings))
        
        return {
            "y_true_path": y_true_path,
            "y_pred_path": y_pred_path,
            "embeddings_path": embeddings_path
        }


class FoldEvalResult(BaseModel):
    """Evaluation results for a single fold."""
    fold: int
    train_result: SplitEvalResult
    val_result: SplitEvalResult
    test_result: SplitEvalResult
    model_dir: Path
    model_loaded_successfully: bool

    @model_validator(mode="after")
    def serialize_model_dir(self):
        self.model_dir = str(self.model_dir)
        return self


class KFoldEvalResult(BaseModel):
    """K-fold cross-validation evaluation results."""
    results: List[FoldEvalResult]
    
    # Aggregated metrics across all folds for each split
    train_mae: KFoldMetrics
    train_mse: KFoldMetrics
    train_r2: KFoldMetrics
    
    val_mae: KFoldMetrics
    val_mse: KFoldMetrics
    val_r2: KFoldMetrics
    
    test_mae: KFoldMetrics
    test_mse: KFoldMetrics
    test_r2: KFoldMetrics
    
    k_folds: int
    config: TrainingConf
    successfully_loaded_models: int
    total_samples_evaluated: int


def load_trained_model(
    model_dir: Path,
    model_config: PolyGCBaseModel.ModelConfig,
    fold_idx: int,
    device: torch.device
) -> Tuple[Optional[PolyGCBaseModel], bool]:
    """
    Load a trained model from a specific fold.
    
    Args:
        model_dir: Directory containing the trained model
        model_config: Model configuration
        fold_idx: Fold index for model identification
        device: Device to load model on
        
    Returns:
        Tuple of (loaded_model, success_flag)
    """
    try:
        # Construct model hash similar to training pipeline
        model_conf_dict = model_config.model_dump()
        model_conf_json = json.dumps(model_conf_dict, sort_keys=True, indent=2)
        
        from hashlib import md5
        model_config_hash = md5(model_conf_json.encode()).hexdigest()
        model_config_hash = f"{model_config_hash}_fold_{fold_idx}"
        
        sub_model_dir = model_dir / model_config_hash
        model_file = sub_model_dir / "best_model.pt"
        
        if not model_file.exists():
            # Also try the alternative path structure
            model_file = sub_model_dir / "model.pt"
            
        if not model_file.exists():
            print(f"Warning: Model file not found for fold {fold_idx + 1} at {sub_model_dir}")
            return None, False
            
        # Load model
        model = PolyGCBaseModel(config=model_config)
        model.load_state_dict(torch.load(model_file, map_location=device))
        model = model.to(device)
        model.eval()
        
        print(f"Successfully loaded model for fold {fold_idx + 1} from {model_file}")
        return model, True
        
    except Exception as e:
        print(f"Error loading model for fold {fold_idx + 1}: {e}")
        return None, False


def evaluate_model_on_split(
    model: PolyGCBaseModel,
    data_loader: DataLoader,
    split_name: str,
    device: torch.device
) -> SplitEvalResult:
    """
    Evaluate a model on a specific data split.
    
    Args:
        model: Trained model to evaluate
        data_loader: DataLoader for the split
        split_name: Name of the split ("train", "val", "test")
        device: Device for computation
        
    Returns:
        SplitEvalResult containing predictions and metrics
    """
    model.eval()
    
    all_true = []
    all_pred = []
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {split_name}", leave=False):
            batch = batch.to(device)
            
            # Get predictions
            preds = model.predict(batch)
            embeddings = model.predict_embedding(batch)
            
            all_true.append(batch.y.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_embeddings.append(embeddings.cpu().numpy())
    
    # Concatenate results
    y_true = np.concatenate(all_true).flatten()
    y_pred = np.concatenate(all_pred).flatten()
    embeddings_array = np.concatenate(all_embeddings, axis=0)
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return SplitEvalResult(
        split_name=split_name,
        y_true=y_true.tolist(),
        y_pred=y_pred.tolist(),
        embeddings=embeddings_array.tolist(),
        mae=mae,
        mse=mse,
        r2=r2,
        num_samples=len(y_true)
    )


def evaluate_fold(
    trained_models_dir: Path,
    conf: TrainingConf,
    fold_idx: int,
    fold_splits: List[Tuple],
    all_graph_data: List[Data],
    device: torch.device
) -> FoldEvalResult:
    """
    Evaluate models for a specific fold on train/val/test splits.
    
    Args:
        trained_models_dir: Directory containing trained models
        conf: Training configuration
        fold_idx: Index of the fold to evaluate
        fold_splits: List of fold splits from training
        all_graph_data: All graph data
        device: Device for computation
        
    Returns:
        FoldEvalResult containing evaluation results for all splits
    """
    print(f"\n{'=' * 20} Evaluating Fold {fold_idx + 1} {'=' * 20}")
    
    # Load the trained model for this fold
    model, model_loaded = load_trained_model(
        model_dir=trained_models_dir,
        model_config=conf.model_conf,
        fold_idx=fold_idx,
        device=device
    )
    
    if not model_loaded or model is None:
        # Create dummy results if model couldn't be loaded
        dummy_result = SplitEvalResult(
            split_name="dummy",
            y_true=[],
            y_pred=[],
            embeddings=[],
            mae=float('inf'),
            mse=float('inf'),
            r2=-float('inf'),
            num_samples=0
        )
        
        return FoldEvalResult(
            fold=fold_idx + 1,
            train_result=dummy_result,
            val_result=dummy_result,
            test_result=dummy_result,
            model_dir=trained_models_dir,
            model_loaded_successfully=False
        )
    
    # Create data loaders for this fold
    train_loader, val_loader, test_loader, _, _, _ = create_fold_dataloaders(
        all_graph_data=all_graph_data,
        fold_splits=fold_splits,
        fold_idx=fold_idx,
        batch_size=conf.batch_size
    )
    
    # Evaluate on each split
    train_result = evaluate_model_on_split(model, train_loader, "train", device)
    val_result = evaluate_model_on_split(model, val_loader, "val", device)
    test_result = evaluate_model_on_split(model, test_loader, "test", device)
    
    # Move model back to CPU to free GPU memory
    model = model.to("cpu")
    
    return FoldEvalResult(
        fold=fold_idx + 1,
        train_result=train_result,
        val_result=val_result,
        test_result=test_result,
        model_dir=trained_models_dir,
        model_loaded_successfully=True
    )


def evaluation_pipeline(
    ds_name: str,
    trained_models_dir: Path,
    db_path: Path,
    output_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    k_folds: int = 5,
    seed: int = 42,
    conf: Optional[TrainingConf] = None,
    device: Optional[str] = None,
    reduce_identical: bool = True,
    cached_data: Optional[List[Data]] = None,
    create_visualizations: bool = True,
    property_name: str = "Value",
    unit: str = "",
) -> Tuple[KFoldEvalResult, Dict[str, Any]]:
    """
    Evaluation pipeline for k-fold cross-validation trained models.
    
    This function loads previously trained models and evaluates them on the same
    data splits used during training, providing comprehensive evaluation results
    for each fold and each split (train/val/test).
    
    Args:
        ds_name: Dataset name
        trained_models_dir: Directory containing trained models
        db_path: Path to database
        output_dir: Directory to save evaluation results
        data_dir: Directory for cached data
        k_folds: Number of folds (must match training)
        seed: Random seed (must match training)
        conf: Training configuration (if None, will attempt to load from trained models)
        device: Device to use for evaluation
        reduce_identical: Whether to reduce identical features
        cached_data: Pre-loaded graph data
        
    Returns:
        Tuple of (KFoldEvalResult, additional_data_dict)
        
    Raises:
        FileNotFoundError: If trained_models_dir doesn't exist
        ValueError: If no models can be loaded
    """
    print(f"Starting evaluation pipeline for dataset: {ds_name}")
    
    # Validate inputs
    trained_models_dir = Path(trained_models_dir)
    if not trained_models_dir.exists():
        raise FileNotFoundError(f"Trained models directory not found: {trained_models_dir}")
    
    if not trained_models_dir.is_dir():
        raise NotADirectoryError(f"Trained models path is not a directory: {trained_models_dir}")
    
    # Set up output directory
    if output_dir is None:
        output_dir = trained_models_dir / "evaluation_results"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data (same as training pipeline)
    print("Loading data...")
    if cached_data is None:
        all_graph_data = load_data(
            ds_name, db_path, data_dir, reduce_indentical=reduce_identical
        )
    else:
        all_graph_data = cached_data
    
    # Load or create configuration
    if conf is None:
        # Try to load configuration from trained models directory
        kfold_result_path = trained_models_dir / "kfold_result.json"
        if kfold_result_path.exists():
            print(f"Loading training configuration from {kfold_result_path}")
            with open(kfold_result_path, 'r') as f:
                kfold_data = json.load(f)
                conf = TrainingConf(**kfold_data['config'])
        else:
            # Create default configuration
            print("No training configuration found, using defaults")
            conf = TrainingConf(
                model_conf=PolyGCBaseModel.ModelConfig(
                    monomer_features=all_graph_data[0].x.shape[1],
                )
            )
    else:
        # Handle configuration setup similar to training pipeline
        if isinstance(conf, dict):
            if "model_conf" in conf:
                if isinstance(conf["model_conf"], PolyGCBaseModel.ModelConfig):
                    conf["model_conf"] = conf["model_conf"].model_dump()
                conf["model_conf"] = PolyGCBaseModel.ModelConfig(
                    **{
                        **conf["model_conf"],
                        **{"monomer_features": all_graph_data[0].x.shape[1]},
                    }
                )
            else:
                conf["model_conf"] = {"monomer_features": all_graph_data[0].x.shape[1]}
            conf = TrainingConf(**conf)
    
    # Create the same k-fold splits as training
    print(f"Creating {k_folds}-fold splits (must match training splits)...")
    fold_splits = get_kfold_splits(all_graph_data, k_folds=k_folds, seed=seed)
    
    # Set up device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")
    
    # Evaluate each fold
    fold_eval_results = []
    successfully_loaded = 0
    total_samples = 0
    
    for fold_idx in range(k_folds):
        fold_result = evaluate_fold(
            trained_models_dir=trained_models_dir,
            conf=conf,
            fold_idx=fold_idx,
            fold_splits=fold_splits,
            all_graph_data=all_graph_data,
            device=device
        )
        
        fold_eval_results.append(fold_result)
        
        if fold_result.model_loaded_successfully:
            successfully_loaded += 1
            total_samples += (
                fold_result.train_result.num_samples +
                fold_result.val_result.num_samples +
                fold_result.test_result.num_samples
            )
        
        # Save individual fold results
        fold_result_path = output_dir / f"fold_eval_result_{fold_result.fold}.json"
        
        # Save large arrays separately
        arrays_dir = output_dir / "arrays"
        arrays_dir.mkdir(exist_ok=True, parents=True)
        
        if fold_result.model_loaded_successfully:
            fold_id = f"fold_{fold_result.fold}"
            fold_result.train_result.save_arrays(arrays_dir, fold_id)
            fold_result.val_result.save_arrays(arrays_dir, fold_id)
            fold_result.test_result.save_arrays(arrays_dir, fold_id)
        
        # Save metadata (without large arrays)
        with open(fold_result_path, "w") as f:
            json.dump(fold_result.model_dump(), f, indent=2, sort_keys=True)
    
    # Check if any models were loaded successfully
    if successfully_loaded == 0:
        raise ValueError("No models could be loaded successfully. Check model paths and configurations.")
    
    print(f"Successfully loaded {successfully_loaded}/{k_folds} models")
    
    # Calculate aggregated metrics across folds
    successful_results = [r for r in fold_eval_results if r.model_loaded_successfully]
    
    # Train split metrics
    train_maes = [r.train_result.mae for r in successful_results]
    train_mses = [r.train_result.mse for r in successful_results]
    train_r2s = [r.train_result.r2 for r in successful_results]
    
    # Val split metrics
    val_maes = [r.val_result.mae for r in successful_results]
    val_mses = [r.val_result.mse for r in successful_results]
    val_r2s = [r.val_result.r2 for r in successful_results]
    
    # Test split metrics
    test_maes = [r.test_result.mae for r in successful_results]
    test_mses = [r.test_result.mse for r in successful_results]
    test_r2s = [r.test_result.r2 for r in successful_results]
    
    # Create final results
    kfold_eval_result = KFoldEvalResult(
        results=fold_eval_results,
        train_mae=KFoldMetrics.from_list(train_maes),
        train_mse=KFoldMetrics.from_list(train_mses),
        train_r2=KFoldMetrics.from_list(train_r2s),
        val_mae=KFoldMetrics.from_list(val_maes),
        val_mse=KFoldMetrics.from_list(val_mses),
        val_r2=KFoldMetrics.from_list(val_r2s),
        test_mae=KFoldMetrics.from_list(test_maes),
        test_mse=KFoldMetrics.from_list(test_mses),
        test_r2=KFoldMetrics.from_list(test_r2s),
        k_folds=k_folds,
        config=conf,
        successfully_loaded_models=successfully_loaded,
        total_samples_evaluated=total_samples
    )
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Dataset: {ds_name}")
    print(f"Models loaded: {successfully_loaded}/{k_folds}")
    print(f"Total samples evaluated: {total_samples}")
    print(f"\nTest Set Performance:")
    print(f"  MAE: {kfold_eval_result.test_mae.mean:.3f} ± {kfold_eval_result.test_mae.std:.3f}")
    print(f"  R²:  {kfold_eval_result.test_r2.mean:.3f} ± {kfold_eval_result.test_r2.std:.3f}")
    print(f"\nValidation Set Performance:")
    print(f"  MAE: {kfold_eval_result.val_mae.mean:.3f} ± {kfold_eval_result.val_mae.std:.3f}")
    print(f"  R²:  {kfold_eval_result.val_r2.mean:.3f} ± {kfold_eval_result.val_r2.std:.3f}")
    print("="*50)
    
    # Save complete results
    results_path = output_dir / "kfold_evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(kfold_eval_result.model_dump(), f, indent=2, sort_keys=True)
    
    print(f"Complete evaluation results saved to: {results_path}")
    
    # Create visualizations if requested
    visualization_results = None
    if create_visualizations:
        print("\n" + "="*50)
        print("CREATING K-FOLD VISUALIZATION SUITE")
        print("="*50)
        
        # Create visualization suite using evaluation results directly
        visualization_results = create_kfold_visualization_suite(
            kfold_eval_result=kfold_eval_result,
            dataset_name=ds_name,
            output_dir=output_dir / "visualizations",
            create_tsne=True,
            property_name=property_name,
            unit=unit
        )
        
        print("K-fold visualization suite completed!")
    
    return kfold_eval_result, {
        "all_graph_data": all_graph_data,
        "fold_splits": fold_splits,
        "output_dir": output_dir,
        "visualization_results": visualization_results
    }

