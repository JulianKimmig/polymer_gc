import polymer_gc
from typing import Optional, List, Tuple
import torch
from sklearn.model_selection import train_test_split
from .load import load as load_model, load_modelconfig

import polymer_gc.model
import polymer_gc.model.base
import polymer_gc.model.data  # For robust splitting
import os
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm
from torch import optim
import importlib
import json
from pydantic import Field, BaseModel


def _split_source_dataset_entries(
    source_dataset_items: List[polymer_gc.dataset.DataSetEntry],
    split_ratios: Tuple[float, float, float],
    seed: int,
) -> Tuple[
    List[polymer_gc.dataset.DataSetEntry],
    List[polymer_gc.dataset.DataSetEntry],
    List[polymer_gc.dataset.DataSetEntry],
]:
    """Splits a list of DataSetEntry objects into train, validation, and test sets."""
    if not source_dataset_items:
        return [], [], []

    # Use a unique identifier for each entry for stratified splitting if needed,
    # or just for consistent splitting. Here, we'll use indices for simplicity.
    indices = list(range(len(source_dataset_items)))

    train_val_ratio = split_ratios[0] + split_ratios[1]
    if train_val_ratio == 0:  # No train or val, all test
        return [], [], list(source_dataset_items)
    if train_val_ratio == 1.0:  # No test set
        test_indices = []
        train_val_indices = indices
    else:
        train_val_indices, test_indices = train_test_split(
            indices,
            test_size=split_ratios[2],  # test_ratio
            random_state=seed,
            shuffle=True,
        )

    if not train_val_indices:  # If train+val is empty
        train_indices = []
        val_indices = []
    elif split_ratios[0] == 0:  # No train, all val from train_val
        train_indices = []
        val_indices = train_val_indices
    elif (
        split_ratios[1] == 0 or train_val_ratio == split_ratios[0]
    ):  # No val, all train from train_val
        train_indices = train_val_indices
        val_indices = []
    else:
        # Calculate validation proportion relative to the combined train+val set
        val_relative_ratio = split_ratios[1] / train_val_ratio
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_relative_ratio,
            random_state=seed,  # Use the same seed for reproducibility in this sub-split
            shuffle=True,
        )

    train_entries = [source_dataset_items[i] for i in train_indices]
    val_entries = [source_dataset_items[i] for i in val_indices]
    test_entries = [source_dataset_items[i] for i in test_indices]

    return train_entries, val_entries, test_entries


class TrainingConfig(BaseModel):
    lr: float = Field(default=1e-3, description="Learning rate for the optimizer.")
    epochs: int = Field(default=100, description="Number of training epochs.")
    batch_size: int = Field(default=32, description="Batch size for training.")
    device: Optional[str] = Field(
        default=None,
        description="Device to use for training (e.g., 'cuda' or 'cpu').",
    )
    seed: int = Field(default=42, description="Random seed for reproducibility.")
    num_workers: int = Field(
        default=0,
        description="Number of workers for data loading. Set to 0 for no multiprocessing.",
    )
    prefit: bool = Field(
        default=True,
        description="Whether to prefit the model before training.",
    )
    split_ratios: List[float] = Field(
        ...,
        description="Ratios for splitting the dataset into train, val, and test sets.",
        default_factory=lambda: [0.8, 0.1, 0.1],
    )


def train(
    model_config: polymer_gc.model.base.PolyGCBaseModel.ModelConfig,
    source_dataset: polymer_gc.dataset.Dataset,
    pg_dataset_root: str,  # Root directory for PolymerGraphDataset processed files
    pg_dataset_config: polymer_gc.dataset.PgDatasetConfig,
    train_config: TrainingConfig,
    training_path: Optional[str] = None,
    save_best: bool = True,
    load_best: bool = True,
    continue_training: bool = True,
):
    lr = train_config.lr
    epochs = train_config.epochs
    batch_size = train_config.batch_size
    device_str = train_config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed = train_config.seed
    num_workers = train_config.num_workers
    prefit = train_config.prefit

    split_ratios = train_config.split_ratios

    if not source_dataset or not source_dataset.items:
        print("Source dataset is empty. Cannot train.")
        return None, {}

    device = torch.device(device_str)

    print(f"Using device: {device}")

    model = load_modelconfig(
        model_config=model_config,
    )

    train_entries, val_entries, test_entries = _split_source_dataset_entries(
        source_dataset.items,
        split_ratios,
        seed,
    )

    train_source_ds_obj = polymer_gc.dataset.Dataset(
        name=f"{source_dataset.name}_train", items=train_entries
    )
    train_pg_dataset = polymer_gc.model.data.PolymerGraphDataset(
        dataset=train_source_ds_obj,
        root=os.path.join(pg_dataset_root, "train"),
        config=pg_dataset_config,
    )
    print("Preparing training dataset...")
    train_pg_dataset.prepare()
    print(f"Training dataset prepared with {len(train_pg_dataset)} entries.")

    val_pg_dataset = None
    if val_entries:
        val_source_ds_obj = polymer_gc.dataset.Dataset(
            name=f"{source_dataset.name}_val", items=val_entries
        )
        val_pg_dataset = polymer_gc.model.data.PolymerGraphDataset(
            dataset=val_source_ds_obj,
            root=os.path.join(pg_dataset_root, "val"),
            config=pg_dataset_config,
        )
        print("Preparing validation dataset...")
        val_pg_dataset.prepare()
        print(f"Validation dataset prepared with {len(val_pg_dataset)} entries.")
    else:
        print(
            "Warning: Validation set (source entries) is empty. No validation will be performed."
        )

    test_pg_dataset = None
    if test_entries:
        test_source_ds_obj = polymer_gc.dataset.Dataset(
            name=f"{source_dataset.name}_test", items=test_entries
        )
        test_pg_dataset = polymer_gc.model.data.PolymerGraphDataset(
            dataset=test_source_ds_obj,
            root=os.path.join(pg_dataset_root, "test"),
            config=pg_dataset_config,
        )
        print("Preparing test dataset...")
        test_pg_dataset.prepare()
        print(f"Test dataset prepared with {len(test_pg_dataset)} entries.")

    train_loader = PyGDataLoader(
        train_pg_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # Consider num_workers > 0 for speed
    )
    val_loader = (
        PyGDataLoader(
            val_pg_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        if val_pg_dataset and len(val_pg_dataset) > 0
        else None
    )
    test_loader = (
        PyGDataLoader(
            test_pg_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        if test_pg_dataset and len(test_pg_dataset) > 0
        else None
    )

    if training_path:
        if not os.path.exists(training_path):
            os.makedirs(training_path, exist_ok=True)

        with open(os.path.join(training_path, "model.txt"), "w") as f:
            f.write(str(model))

        with open(os.path.join(training_path, "train_config.json"), "w") as f:
            json.dump(train_config.model_dump(mode="json"), f, indent=4)
        with open(os.path.join(training_path, "model_config.json"), "w") as f:
            json.dump(model_config.model_dump(mode="json"), f, indent=4)
        with open(os.path.join(training_path, "pg_dataset_config.json"), "w") as f:
            json.dump(pg_dataset_config.model_dump(mode="json"), f, indent=4)

    if training_path is None:
        save_best = False

    if not save_best:
        load_best = False

    if training_path:
        model_path = os.path.join(training_path, "best_model.pth")
        optimizer_path = os.path.join(training_path, "optimizer.pth")
        training_data_path = os.path.join(training_path, "training_data.json")
        history_path = os.path.join(training_path, "history.json")
    training_data = {}
    history = {
        "train_loss": {},
        "val_loss": {},
        "val_mae": {},  # MAE on mean predictions
        "test_loss": None,
        "test_mae": None,
    }

    def save():
        if training_path:
            torch.save(model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optimizer_path)
            with open(training_data_path, "w") as f:
                json.dump(training_data, f, indent=4)
            with open(history_path, "w") as f:
                json.dump(history, f, indent=4)

    def load():
        model_loaded = False
        if training_path:
            model_loaded, optimizer_loaded = load_model(
                model=model,
                model_path=model_path,
                optimizer=optimizer,
                optimizer_path=optimizer_path,
            )
            if os.path.exists(training_data_path):
                with open(training_data_path, "r") as f:
                    training_data.update(json.load(f))

        return model_loaded

    print("Training model...")
    model = model.to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
    )

    model_loaded = False
    if continue_training:
        model_loaded = load()
        if os.path.exists(training_data_path):
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    history.update(json.load(f))

    if prefit and not model_loaded:
        print("Prefitting model...")
        prefit_y_list = []
        prefit_additional_inputs_list = []
        prefit_x_list = []
        # Only prefit on the training data to avoid leakage
        for data_batch in tqdm(
            train_loader,  # Use train_loader for prefit
            desc="Generating prefit data from training set",
            unit="batch",
            total=len(train_loader),
        ):
            prefit_y_list.append(data_batch.y)
            prefit_additional_inputs_list.append(data_batch.additional_features)
            prefit_x_list.append(data_batch.x)

        if not prefit_y_list:
            print("No data in train_loader for prefit. Check dataset generation.")
            prefit_y_tensor = torch.empty(
                0, model.config.num_target_properties
            )  # Adjust shape as needed
        else:
            prefit_y_tensor = torch.cat(prefit_y_list)

        if not prefit_x_list:
            print("No data in train_loader for prefit. Check dataset generation.")
            prefit_x_tensor = torch.empty(0, model.config.monomer_features)
        else:
            prefit_x_tensor = torch.cat(prefit_x_list)

        if not prefit_additional_inputs_list:
            prefit_additional_inputs_tensor = torch.empty(
                0, model.config.additional_features
            )
        else:
            prefit_additional_inputs_tensor = torch.cat(prefit_additional_inputs_list)

        model.prefit(
            x=prefit_x_tensor.to(device),
            y=prefit_y_tensor.to(device),
            additional_inputs=prefit_additional_inputs_tensor.to(device),
        )

    if save_best:
        best_model_loss = training_data.setdefault("best_model_loss", float("inf"))

    training_data.setdefault("epoch", 0)

    train_loader_length = len(train_loader.dataset)
    try:
        try:
            with tqdm(total=epochs, desc="Training", unit="epoch") as pbar:
                pbar.update(training_data["epoch"])
                bar_postfix_data = {}
                for epoch in range(training_data["epoch"], epochs):
                    # --- Training phase ---
                    model.train()
                    epoch_train_loss_sum = 0.0

                    for data in train_loader:
                        data = data.to(device)
                        optimizer.zero_grad()
                        # model.batch_loss internally calls self.target_scaler on data.y
                        train_loss = model.batch_loss(data, context="train")
                        if torch.isnan(train_loss).any():
                            raise ValueError("Loss is NaN. Check your model or data.")
                        train_loss.backward()
                        optimizer.step()
                        epoch_train_loss_sum += train_loss.item() * data.num_graphs
                        bar_postfix_data["train_loss"] = train_loss.item()
                        pbar.set_postfix(**bar_postfix_data)

                    avg_epoch_train_loss = epoch_train_loss_sum / train_loader_length
                    bar_postfix_data["avg_train_loss"] = avg_epoch_train_loss
                    history["train_loss"][str(epoch)] = avg_epoch_train_loss
                    pbar.set_postfix(**bar_postfix_data)
                    training_data["epoch"] = epoch + 1
                    # --- Validation phase ---
                    if val_loader:
                        model.eval()
                        epoch_val_loss_sum = 0.0
                        epoch_val_mae_sum = 0.0

                        with torch.no_grad():
                            for data in val_loader:
                                data = data.to(device)
                                # predict returns unscaled predictions

                                # Get unscaled predictions
                                preds = model.predict(data)
                                val_loss = model.validation_loss_function(preds, data.y)
                                epoch_val_loss_sum += val_loss.item() * data.num_graphs
                                if model.config.logits_output:
                                    out_mean_unscaled, out_log_var_unscaled = preds
                                else:
                                    out_mean_unscaled = preds
                                    out_log_var_unscaled = None
                                # MAE is on unscaled predictions vs unscaled targets
                                mae = torch.abs(out_mean_unscaled - data.y).mean()
                                epoch_val_mae_sum += mae.item() * data.num_graphs

                                bar_postfix_data["val_loss"] = val_loss.item()
                                bar_postfix_data["val_mae"] = mae.item()
                                pbar.set_postfix(**bar_postfix_data)

                        avg_epoch_val_loss = epoch_val_loss_sum / len(
                            val_loader.dataset
                        )
                        avg_epoch_val_mae = epoch_val_mae_sum / len(val_loader.dataset)
                        history["val_loss"][str(epoch)] = avg_epoch_val_loss
                        bar_postfix_data["avg_val_loss"] = avg_epoch_val_loss
                        history["val_mae"][str(epoch)] = avg_epoch_val_mae
                        bar_postfix_data["avg_val_mae"] = avg_epoch_val_mae
                        pbar.set_postfix(**bar_postfix_data)
                        if best_model_loss > avg_epoch_val_loss:
                            best_model_loss = avg_epoch_val_loss
                            training_data["best_model_loss"] = best_model_loss
                            if save_best:
                                save()

                    pbar.update(1)

        except KeyboardInterrupt:
            print("Training interrupted.")

        if not save_best:
            save()

        if load_best and os.path.exists(model_path):
            load()

        # 7. Testing phase
        if test_loader:
            print("Evaluating on Test Set...")
            model.eval()
            test_loss_sum = 0.0
            test_mae_sum = 0.0
            test_pbar = tqdm(test_loader, desc="Testing", leave=False, unit="batch")
            with torch.no_grad():
                for data in test_pbar:
                    data = data.to(device)
                    test_loss = model.batch_loss(data, context="test")
                    test_loss_sum += test_loss.item() * data.num_graphs
                    # Get unscaled predictions
                    preds = model.predict(data)
                    if model.config.logits_output:
                        out_mean_unscaled, out_log_var_unscaled = preds
                    else:
                        out_mean_unscaled = preds
                        out_log_var_unscaled = None
                    mae = torch.abs(out_mean_unscaled - data.y).mean()
                    test_mae_sum += mae.item() * data.num_graphs
                    test_pbar.set_postfix(loss=test_loss.item(), mae=mae.item())

            avg_test_loss = test_loss_sum / len(test_loader.dataset)
            avg_test_mae = test_mae_sum / len(test_loader.dataset)
            history["test_loss"] = avg_test_loss
            history["test_mae"] = avg_test_mae
            print(f"Test Results - Loss: {avg_test_loss:.4f}, MAE: {avg_test_mae:.4f}")

        print("Training finished.")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=4)
    finally:
        pass

    return model, {
        "history": history,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "val_loader": val_loader,
    }
