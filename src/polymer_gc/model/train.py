from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader as PyGDataLoader
from .base import PolyGCBaseModel
from .data import PolymerGraphDataset


def train_polymer_model(
    pg_dataset: "PolymerGraphDataset",  # Type hint with string if class not defined yet
    model: PolyGCBaseModel,  # Pass the PolyGCBaseModel class itself
    train_config: dict,
):
    """
    Trains a polymer graph classification/regression model.

    Args:
        pg_dataset: An instance of your PolymerGraphDataset.
        model_class: The class of the model to be trained (e.g., PolyGCBaseModel).
        model_config: Dictionary of arguments for initializing the model
                      (e.g., gc_features, num_gnn_layers, dropout_rate).
        train_config: Dictionary of training parameters
                      (e.g., lr, epochs, batch_size, device, split_ratios).

    Returns:
        Trained model and a dictionary containing training history.
    """

    # 1. Training Configuration
    device_str = train_config.get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_str)
    print(f"Using device: {device}")

    lr = train_config.get("lr", 1e-3)
    epochs = train_config.get("epochs", 100)
    batch_size = train_config.get("batch_size", 32)
    # train, validation, test split ratios
    split_ratios = train_config.get("split_ratios", (0.8, 0.1, 0.1))

    if not pg_dataset or len(pg_dataset) == 0:
        print("Dataset is empty. Cannot train.")
        return None, {}

    # 2. Data Splitting
    total_size = len(pg_dataset)
    train_size = int(split_ratios[0] * total_size)
    val_size = int(split_ratios[1] * total_size)
    test_size = total_size - train_size - val_size

    if train_size == 0:
        print(
            f"Training set size is 0 after splitting. Cannot train. Adjust split_ratios or dataset size."
        )
        return None, {}
    if val_size == 0:
        print("Warning: Validation set size is 0. No validation will be performed.")

    train_dataset, val_dataset, test_dataset = random_split(
        pg_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(
            train_config.get("seed", 42)
        ),  # for reproducibility
    )

    print(
        f"Dataset sizes: Train={len(train_dataset)}, Validation={len(val_dataset)}, Test={len(test_dataset)}"
    )

    # 3. DataLoaders
    train_loader = PyGDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    # Ensure val_dataset is not empty before creating DataLoader
    val_loader = (
        PyGDataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        if len(val_dataset) > 0
        else None
    )
    # Ensure test_dataset is not empty
    test_loader = (
        PyGDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        if len(test_dataset) > 0
        else None
    )

    print("Prefitting model...")

    prefit_y = []
    prefit_additional_inputs = []
    for data in tqdm(
        train_loader,
        desc="Genrating prefit data",
        unit="batch",
        total=len(train_loader),
    ):
        prefit_y.append(data.y)
        prefit_additional_inputs.append(data.additional_features)

    model.prefit(
        y=torch.cat(prefit_y),  # Assuming y is a tensor
        additional_inputs=torch.cat(
            prefit_additional_inputs
        ),  # Assuming additional_features is a tensor
    )
    print("Training model...")

    # Instantiate the model
    # **IMPORTANT**: Ensure PolyGCBaseModel and its GCBlock are correctly defined.
    # Potential issues in the provided PolyGCBaseModel/GCBlock:
    # 1. GCBlock: `self.linear1` is assigned twice. It should be `self.lin1` and `self.lin2`.
    #    In `forward`, `y=self.lin1(y1)` and `y=self.lin2(y)` (or similar distinct names).
    # 2. GCBlock: `SAGEConv` does not have `concat=True`. If GAT was intended, syntax differs.
    #    Assuming a functional SAGEConv or that this is a placeholder.
    # 3. PolyGCBaseModel: `GCBlock` is called with `dropout=dropout_rate`. The `GCBlock`
    #    constructor expects `dropout_rate=...`.
    # These need to be fixed in your model definition for the training to run.
    model = model.to(device)

    # 5. Optimizer and Loss Function
    print(lr)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
    }  # MAE on mean predictions

    def prepdata(data):
        if data.y.ndim == 1:
            data.y = data.y.unsqueeze(1)
        data = data.to(device)
        return data

    print(f"Starting training for {epochs} epochs...")
    # 6. Training Loop
    with tqdm(total=epochs, desc="Training", unit="epoch") as pbar:
        bar_postfix_data = {}
        for epoch in range(epochs):
            # --- Training phase ---
            model.train()
            epoch_train_loss_sum = 0.0

            # Use tqdm for progress bar
            for data in train_loader:
                data = prepdata(data)
                optimizer.zero_grad()
                loss = model.batch_loss(data)
                if torch.isnan(loss).any():
                    print("Loss is NaN. Skipping this batch.")
                    raise ValueError("Loss is NaN. Check your model or data.")
                loss.backward()
                optimizer.step()
                epoch_train_loss_sum += (
                    loss.item() * data.num_graphs
                )  # loss.item() is avg loss for batch
                bar_postfix_data["train_loss"] = loss.item()
                pbar.set_postfix(**bar_postfix_data)

            avg_epoch_train_loss = epoch_train_loss_sum / len(train_loader.dataset)
            bar_postfix_data["avg_train_loss"] = avg_epoch_train_loss
            history["train_loss"].append(avg_epoch_train_loss)

            # --- Validation phase ---
            avg_epoch_val_loss = None
            avg_epoch_val_mae = None
            if val_loader:  # Check if val_loader was created
                model.eval()
                epoch_val_loss_sum = 0.0
                epoch_val_mae_sum = 0.0

                with torch.no_grad():
                    for data in val_loader:
                        data = prepdata(data)
                        out = model.predict(data)
                        loss = model.loss(out, data.y)
                        epoch_val_loss_sum += loss.item() * data.num_graphs

                        if model.logits_output:
                            mean_preds = out[0]
                        else:
                            mean_preds = out
                        mae = torch.abs(
                            mean_preds - data.y
                        ).mean()  # MAE over properties and batch
                        epoch_val_mae_sum += mae.item() * data.num_graphs
                        bar_postfix_data["val_loss"] = loss.item()
                        bar_postfix_data["val_mae"] = mae.item()
                        pbar.set_postfix(**bar_postfix_data)

                avg_epoch_val_loss = epoch_val_loss_sum / len(val_loader.dataset)
                avg_epoch_val_mae = epoch_val_mae_sum / len(val_loader.dataset)
                history["val_loss"].append(avg_epoch_val_loss)
                history["val_mae"].append(avg_epoch_val_mae)
            else:  # No validation set
                history["val_loss"].append(None)
                history["val_mae"].append(None)

            pbar.update(1)

    # 7. Testing phase (optional, after training)
    if test_loader:
        print("Evaluating on Test Set...")
        model.eval()
        test_loss_sum = 0.0
        test_mae_sum = 0.0

        test_pbar = tqdm(test_loader, desc="Testing", leave=False)
        with torch.no_grad():
            for data in test_pbar:
                data = prepdata(data)
                out = model.predict(data)
                loss = model.loss(out, data.y)
                test_loss_sum += loss.item() * data.num_graphs
                if model.logits_output:
                    mean_preds = out[0]
                else:
                    mean_preds = out
                mae = torch.abs(mean_preds - data.y).mean()
                test_mae_sum += mae.item() * data.num_graphs
                test_pbar.set_postfix(loss=loss.item(), mae=mae.item())

        avg_test_loss = test_loss_sum / len(test_loader.dataset)
        avg_test_mae = test_mae_sum / len(test_loader.dataset)
        history["test_loss"] = avg_test_loss
        history["test_mae"] = avg_test_mae
        print(f"Test Results - Loss: {avg_test_loss:.4f}, MAE: {avg_test_mae:.4f}")
    else:
        history["test_loss"] = None
        history["test_mae"] = None

    print("Training finished.")
    return model, history
