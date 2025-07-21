import json
from typing import Any, Dict, Optional, Tuple, Union, List, Optional
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from rich import print

from polymer_gc.data.database import SessionManager
from polymer_gc.data.dataset import Dataset
from torch_geometric.data import Data
from hashlib import md5

import numpy as np
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
import torch
from pathlib import Path
from pydantic import BaseModel, Field, PrivateAttr, model_validator
from polymer_gc.model.base import PolyGCBaseModel
from torch import optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class TrainingConf(BaseModel):
    model_conf: PolyGCBaseModel.ModelConfig = Field(
        default_factory=PolyGCBaseModel.ModelConfig
    )
    learning_rate: float = Field(default=0.0001)
    batch_size: int = Field(default=32)
    epochs: int = Field(default=50)
    scheduler_factor: float = Field(default=0.5)
    early_stopping_patience: int = Field(default=10)
    scheduler_patience: int = Field(default=5)


class TrainingResult(BaseModel):
    fold: int = Field(default=0)
    test_mae: float
    test_mse: float
    test_r2: float
    epochs_trained: int
    best_epoch: int
    model_dir: Path
    # private model field that is not serialized
    _model: PolyGCBaseModel = PrivateAttr()

    # correctly serialize model_dir
    @model_validator(mode="after")
    def serialize_model_dir(self):
        self.model_dir = str(self.model_dir)
        return self


class KFoldMetrics(BaseModel):
    min: float
    max: float
    mean: float
    std: float

    @classmethod
    def from_list(cls, values: List[float]):
        return cls(
            min=np.min(values),
            max=np.max(values),
            mean=np.mean(values),
            std=np.std(values),
        )


class KFoldResult(BaseModel):
    results: List[TrainingResult]
    test_mae: KFoldMetrics
    test_mse: KFoldMetrics
    test_r2: KFoldMetrics
    k_folds: int
    config: TrainingConf
    used_pretrained: bool


def load_data(
    ds_name: str,
    db_path: Path,
    data_dir: Optional[Path] = None,
    reduce_indentical: bool = True,
)->List[Data]:
    """Load and cache the graph data for Jablonka dataset."""
    if data_dir is None:
        data_dir = db_path.parent / "data"

    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)
    db_path = Path(db_path)

    with SessionManager(db_path) as session:
        dataset = Dataset.get(name=ds_name)
        if dataset is None:
            raise ValueError(f"Dataset '{ds_name}' not found in {db_path}")
        dataset_dict = dataset.model_dump()
        dataset_json = json.dumps(dataset_dict, sort_keys=True, indent=2)
        dataset_hash = md5(dataset_json.encode()).hexdigest()

    dataset_path = data_dir / dataset_hash
    dataset_path.mkdir(exist_ok=True, parents=True)

    with open(dataset_path / "dataset_config.json", "w+") as f:
        f.write(dataset_json)

    graph_data_file = data_dir / dataset_hash / "data.pt"

    if graph_data_file.exists():
        print(f"Loading cached data from {graph_data_file}...")
        all_graph_data = torch.load(graph_data_file, weights_only=False)
        if reduce_indentical:
            # graph_data_x[i] shape is (n_nodes, n_features)
            # reduce_indentical will reduce the number of features by dropping the features that are the same for all nodes
            # and keep the features that are different for at least one node
            all_gx = torch.cat([g.x for g in all_graph_data], dim=0)
            is_identical = torch.all(all_gx == all_gx[0], dim=0)
            for i, g in enumerate(all_graph_data):
                g.x = g.x[:, ~is_identical]
            print(
                f"Dropped {is_identical.sum()} features, keeping {all_graph_data[0].x.shape[1]} features"
            )

    else:
        print("Processing data from database...")
        with SessionManager(db_path) as session:
            dataset = Dataset.get(name=ds_name)
            print(dataset)
            data = dataset.load_entries_data()

        strucid_to_idx = {val: idx for idx, val in enumerate(data["structure_ids"])}
        vec_strucid_to_idx = np.vectorize(strucid_to_idx.get)

        target_name = dataset.config.targets[0]
        targets_array = data["targets"][target_name]
        print(f"Target variable: '{target_name}'")

        all_graph_data = []
        for g_info in tqdm(data["graphs"], desc="Creating PyG Data objects"):
            structure_idx = vec_strucid_to_idx(g_info["nodes"])
            embeddings = data["all_embeddings"][structure_idx]
            edges = torch.tensor(g_info["edges"], dtype=torch.long).T
            target_value = targets_array[g_info["entry_pos"]]

            graph_data_obj = Data(
                x=torch.tensor(embeddings, dtype=torch.float32),
                edge_index=edges,
                y=torch.tensor([[target_value]], dtype=torch.float32),
                entry_pos=g_info["entry_pos"],
                mass_distribution=torch.tensor(
                    data["sec"][g_info["sec_id"]], dtype=torch.float32
                ).unsqueeze(0),
            )
            all_graph_data.append(graph_data_obj)

        if reduce_indentical:
            all_gx = torch.cat([g.x for g in all_graph_data], dim=0)
            is_identical = torch.all(all_gx == all_gx[0], dim=0)
            for i, g in enumerate(all_graph_data):
                g.x = g.x[:, ~is_identical]
            print(
                f"Dropped {is_identical.sum()} features, keeping {all_graph_data[0].x.shape[1]} features"
            )

        print(f"Total number of graph data points created: {len(all_graph_data)}")
        with open(graph_data_file, "wb") as f:
            torch.save(all_graph_data, f)

    return all_graph_data


def get_kfold_splits(all_graph_data, seed: int, k_folds: int):
    """
    Create k-fold splits based on unique entry indices to prevent data leakage.

    Args:
        all_graph_data: List of PyTorch Geometric Data objects
        k_folds: Number of folds for cross-validation
        random_state: Random seed for reproducibility

    Returns:
        List of (train_indices, val_indices, test_indices) tuples for each fold
    """

    # Get unique entry indices
    entry_indices = np.unique([g.entry_pos for g in all_graph_data])
    num_entries = len(entry_indices)

    # Shuffle entries consistently
    np.random.RandomState(seed).shuffle(entry_indices)

    # Create k-fold splitter
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

    fold_splits = []

    for fold_idx, (trainval_idx, test_idx) in enumerate(kf.split(entry_indices)):
        # Get test entries for this fold
        test_entry_indices = set(entry_indices[test_idx])

        # Split remaining entries into train/val (80/20 split of the remaining data)
        trainval_entries = entry_indices[trainval_idx]
        val_size = max(
            1, int(len(trainval_entries) * 0.2)
        )  # At least 1 entry for validation

        # Further split trainval into train and val
        np.random.RandomState(seed * fold_idx).shuffle(trainval_entries)
        train_entries = set(trainval_entries[val_size:])
        val_entries = set(trainval_entries[:val_size])

        # Convert entry indices to graph indices
        train_graph_indices = [
            i for i, g in enumerate(all_graph_data) if g.entry_pos in train_entries
        ]
        val_graph_indices = [
            i for i, g in enumerate(all_graph_data) if g.entry_pos in val_entries
        ]
        test_graph_indices = [
            i for i, g in enumerate(all_graph_data) if g.entry_pos in test_entry_indices
        ]

        fold_splits.append((train_graph_indices, val_graph_indices, test_graph_indices))

        print(f"Fold {fold_idx + 1}:")
        print(
            f"  Train: {len(train_graph_indices)} samples from {len(train_entries)} entries"
        )
        print(
            f"  Val: {len(val_graph_indices)} samples from {len(val_entries)} entries"
        )
        print(
            f"  Test: {len(test_graph_indices)} samples from {len(test_entry_indices)} entries"
        )

    return fold_splits


def create_fold_dataloaders(all_graph_data, fold_splits, fold_idx, batch_size):
    """
    Create DataLoaders for a specific fold.

    Args:
        all_graph_data: List of all graph data objects
        fold_splits: Output from get_kfold_splits
        fold_idx: Which fold to create loaders for (0-indexed)
        batch_size: Batch size for DataLoaders

    Returns:
        (train_loader, val_loader, test_loader, train_graphs, val_graphs, test_graphs)
    """
    train_indices, val_indices, test_indices = fold_splits[fold_idx]

    train_graphs = [all_graph_data[i] for i in train_indices]
    val_graphs = [all_graph_data[i] for i in val_indices]
    test_graphs = [all_graph_data[i] for i in test_indices]

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_graphs, val_graphs, test_graphs


def train_fold(
    conf: TrainingConf,
    result_dir: Path,
    fold_idx,
    fold_splits,
    all_graph_data,
    device: torch.device,
    pretrained_model_dir: Optional[Path] = None,
):
    """
    Train a model on a specific fold.

    Args:
        conf: Configuration dictionary
        fold_idx: Index of the current fold
        fold_splits: List of fold splits
        all_graph_data: All graph data
        pretrained_model: Optional pre-trained model to fine-tune
    """
    print(f"\n{'=' * 20} Training Fold {fold_idx + 1} {'=' * 20}")

    # Create dataloaders for this fold
    train_loader, val_loader, test_loader, train_graphs, val_graphs, test_graphs = (
        create_fold_dataloaders(all_graph_data, fold_splits, fold_idx, conf.batch_size)
    )

    # Create or use pre-trained model
    if pretrained_model_dir is not None:
        model, model_loaded = make_model(
            conf, result_dir=pretrained_model_dir, fold_idx=fold_idx
        )
    else:
        model, model_loaded = make_model(conf, result_dir=result_dir, fold_idx=fold_idx)
    model = model.to(device)
    if not model_loaded:
        print("Prefitting model...")
        model.prefit(
            x=torch.cat([g.x for g in train_graphs], dim=0).to(device),
            y=torch.cat([g.y for g in train_graphs], dim=0).to(device),
        )

    # Train the model
    train_result = train_model(
        model, train_loader, val_loader, test_loader, conf, device, result_dir
    )
    train_result.fold = fold_idx + 1

    with open(result_dir / f"fold_result_{train_result.fold}.json", "w") as f:
        json.dump(train_result.model_dump(), f, indent=2, sort_keys=True)

    return train_result


def make_model(conf: TrainingConf, result_dir: Path, fold_idx: int = None):
    model_conf_dict = conf.model_conf.model_dump()
    model_conf_json = json.dumps(model_conf_dict, sort_keys=True, indent=2)

    model_config_hash = md5(model_conf_json.encode()).hexdigest()
    if fold_idx is not None:
        model_config_hash = f"{model_config_hash}_fold_{fold_idx}"

    sub_result_dir = result_dir / model_config_hash
    sub_result_dir.mkdir(exist_ok=True)
    model_file = sub_result_dir / f"model.pt"
    model_config_file = sub_result_dir / f"model_config.json"
    with open(model_config_file, "w") as f:
        f.write(model_conf_json)

    model = PolyGCBaseModel(config=conf.model_conf)

    model_loaded = False
    if model_file.exists():
        try:
            print(f"Loading pre-trained model from {model_file}")
            model.load_state_dict(torch.load(model_file))
            model_loaded = True
        except Exception as e:
            print(f"Could not load model state dict: {e}. Starting with a fresh model.")
    model.model_config_hash = model_config_hash
    return model, model_loaded


def train_model(
    model: PolyGCBaseModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    conf: TrainingConf,
    device: torch.device,
    result_dir: Path,
) -> TrainingResult:
    optimizer = optim.AdamW(model.parameters(), lr=conf.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        factor=conf.scheduler_factor,
        patience=conf.scheduler_patience,
    )
    modeldir = result_dir / model.model_config_hash
    modeldir.mkdir(exist_ok=True, parents=True)
    model_file = modeldir / "model.pt"

    # --- Training Loop ---
    print("\n--- Starting Training ---")
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    patience_epochs = conf.early_stopping_patience
    final_epoch = 0

    epochs = conf.epochs
    try:
        for epoch in range(epochs):
            final_epoch = epoch + 1
            model.train()
            total_train_loss = 0
            for batch in tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1:03d}/{epochs} [Train]",
                leave=False,
            ):
                batch = batch.to(device)
                optimizer.zero_grad()
                # The model's batch_loss should internally use GaussianNLLLoss
                loss = model.batch_loss(batch, "train")
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * batch.num_graphs

            avg_train_loss = total_train_loss / len(train_loader.dataset)

            # --- Validation Step ---
            model.eval()
            total_val_loss = 0
            all_val_preds_mu = []
            all_val_true = []
            with torch.no_grad():
                for batch in tqdm(
                    val_loader,
                    desc=f"Epoch {epoch + 1:03d}/{epochs} [Val]",
                    leave=False,
                ):
                    batch = batch.to(device)
                    total_val_loss += (
                        model.batch_loss(batch, "val").item() * batch.num_graphs
                    )
                    # model.predict should return a tensor of shape [N, 2] for [mu, log_var]
                    mean_preds = model.predict(batch)
                    all_val_preds_mu.append(mean_preds.cpu())
                    all_val_true.append(batch.y.cpu())

            avg_val_loss = total_val_loss / len(val_loader.dataset)

            # Calculate validation metrics
            val_preds_mu = torch.cat(all_val_preds_mu).numpy()
            val_true = torch.cat(all_val_true).numpy()
            val_mae = mean_absolute_error(val_true, val_preds_mu)
            val_mse = mean_squared_error(val_true, val_preds_mu)
            val_r2 = r2_score(val_true, val_preds_mu)

            scheduler.step(avg_val_loss)

            print(
                f"Epoch {epoch + 1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val MAE: {val_mae:.2f} K"
            )

            # Save best model and check for early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), model_file)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_epochs:
                    print(
                        f"Early stopping triggered after {patience_epochs} epochs without improvement."
                    )
                    break
    except KeyboardInterrupt:
        print("Training interrupted by user.")

    # --- Final Evaluation on Test Set ---
    print("\n--- Final Evaluation on Test Set ---")
    # Load the best performing model
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    test_true_list = []
    test_preds_list = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on Test Set"):
            batch = batch.to(device)
            test_preds = model.predict(batch)
            test_true_list.append(batch.y.cpu().numpy())
            test_preds_list.append(test_preds.cpu().numpy())

    test_true = np.concatenate(test_true_list)
    test_preds = np.concatenate(test_preds_list)

    model = model.to("cpu")
    model.eval()

    return TrainingResult(
        _model=model,
        model_dir=modeldir,
        test_mae=mean_absolute_error(test_true, test_preds),
        test_mse=mean_squared_error(test_true, test_preds),
        test_r2=r2_score(test_true, test_preds),
        epochs_trained=epoch + 1,
        best_epoch=best_epoch + 1,
    )


def taining_pipeline(
    ds_name: str,
    result_dir: Path,
    db_path: Path,
    data_dir=None,
    k_folds=5,
    seed: int = 42,
    conf: Optional[TrainingConf] = None,
    pretrained_model_dir: Optional[Path] = None,
    device: Optional[str] = None,
    reduce_indentical: bool = True,
    cached_data:Optional[List[Data]] =None,
) -> Tuple[KFoldResult,Dict[str,Any]]:
    print("Generating data...")
    result_dir = Path(result_dir)
    result_dir.mkdir(exist_ok=True, parents=True)
    if cached_data is None:
        all_graph_data = load_data(
            ds_name, db_path, data_dir, reduce_indentical=reduce_indentical
        )
    else:
        all_graph_data = cached_data
    if conf is None:
        conf = TrainingConf(
            model_conf=PolyGCBaseModel.ModelConfig(
                monomer_features=all_graph_data[0].x.shape[1],
            )
        )
    else:
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

    print(f"\nCreating {k_folds}-fold splits...")
    fold_splits = get_kfold_splits(all_graph_data, k_folds=k_folds, seed=seed)
    if pretrained_model_dir is not None:
        if not pretrained_model_dir.exists():
            raise FileNotFoundError(
                f"pretrained model directory {pretrained_model_dir} does not exist"
            )

        if not pretrained_model_dir.is_dir():
            raise NotADirectoryError(
                f"pretrained model directory {pretrained_model_dir} is not a directory"
            )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Train on each fold
    fold_results = []
    all_models = []
    for fold_idx in range(k_folds):
        fold_result = train_fold(
            conf=conf,
            fold_idx=fold_idx,
            fold_splits=fold_splits,
            all_graph_data=all_graph_data,
            pretrained_model_dir=pretrained_model_dir,
            device=device,
            result_dir=result_dir,
        )
        fold_results.append(fold_result)

        print(fold_results)

    # Calculate summary statistics
    kfold_result = KFoldResult(
        results=fold_results,
        test_mae=KFoldMetrics.from_list([r.test_mae for r in fold_results]),
        test_mse=KFoldMetrics.from_list([r.test_mse for r in fold_results]),
        test_r2=KFoldMetrics.from_list([r.test_r2 for r in fold_results]),
        k_folds=k_folds,
        config=conf,
        used_pretrained=pretrained_model_dir is not None,
    )

    # Print summary
    print(kfold_result)
    # Save results
    with open(result_dir / "kfold_result.json", "w") as f:
        json.dump(kfold_result.model_dump(), f, indent=2, sort_keys=True)

    return kfold_result,{"all_graph_data":all_graph_data}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", type=str, required=True)
    parser.add_argument("--db_path", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, required=False)
    parser.add_argument("--prefix", type=str, required=False)
    parser.add_argument("--k_folds", type=int, required=False, default=5)
    parser.add_argument("--seed", type=int, required=False, default=42)
    parser.add_argument(
        "--pretrained_model_dir",
        help="path to pretrained model directory",
        required=False,
        type=Path,
    )
    parser.add_argument("--device", type=str, required=False)
    parser.add_argument("--conf", type=Path, required=False, help="path to conf file")
    parser.add_argument(
        "--result_dir", type=Path, required=False, help="path to result directory"
    )
    args = parser.parse_args()
    if args.conf is not None:
        conf = TrainingConf.model_validate_json(Path(args.conf).read_text())
    else:
        conf = None
    taining_pipeline(
        ds_name=args.ds_name,
        db_path=args.db_path,
        data_dir=args.data_dir,
        prefix=args.prefix,
        k_folds=args.k_folds,
        seed=args.seed,
        pretrained_model_dir=args.pretrained_model_dir,
        device=args.device,
        conf=conf,
        result_dir=args.result_dir,
    )


if __name__ == "__main__":
    main()
