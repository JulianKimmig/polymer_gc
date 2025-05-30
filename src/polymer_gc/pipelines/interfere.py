from typing import Optional
import os
import polymer_gc
import polymer_gc.model
import polymer_gc.model.base
import polymer_gc.model.data  # For robust splitting
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader
from pydantic import BaseModel, Field
import numpy as np
from tqdm import tqdm
from .load import load_modelconfig, load


class InterfereConfig(BaseModel):
    batch_size: int = Field(default=32, description="Batch size for interfereing.")
    device: Optional[str] = Field(
        default=None,
        description="Device to use for interfering (e.g., 'cuda' or 'cpu').",
    )
    num_workers: int = Field(
        default=0,
        description="Number of workers for data loading. Set to 0 for no multiprocessing.",
    )


def interfere(
    model_config: polymer_gc.model.base.PolyGCBaseModel.ModelConfig,
    interfere_dataset: polymer_gc.dataset.Dataset,
    base_path: str,
    pg_dataset_config: polymer_gc.dataset.PgDatasetConfig,
    datadir: str,
    interfere_config: Optional[InterfereConfig] = None,
):
    if interfere_config is None:
        interfere_config = InterfereConfig()

    batch_size = interfere_config.batch_size
    device_str = interfere_config.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    num_workers = interfere_config.num_workers

    device = torch.device(device_str)

    print(f"Using device: {device}")
    model = load_modelconfig(model_config)

    model_path = os.path.join(base_path, "best_model.pth")
    model_loaded = load(model=model, model_path=model_path)[0]
    if not model_loaded:
        raise ValueError(
            f"Model {model_config.model} could not be loaded from {model_path}."
        )

    interfere_dataset = polymer_gc.model.data.PolymerGraphDataset(
        dataset=interfere_dataset, config=pg_dataset_config, root=datadir
    )

    print("Preparing interfere dataset...")
    interfere_dataset.prepare()
    print(f"Interfere dataset prepared with {len(interfere_dataset)} entries.")

    model.to(device)
    model.eval()

    interfere_loader = PyGDataLoader(
        interfere_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    value_preds = []
    if model.config.logits_output:
        std_preds = []
    with torch.no_grad():
        for data in tqdm(
            interfere_loader,
            desc="Interfereing",
            unit="batch",
            total=len(interfere_loader),
        ):
            data = data.to(device)
            preds = model.predict(data)
            if model.config.logits_output:
                out_mean_unscaled, out_log_var_unscaled = preds
                std_preds.append(torch.exp(out_log_var_unscaled / 2).cpu().numpy())
            else:
                out_mean_unscaled = preds

            out_mean_unscaled = out_mean_unscaled.cpu().numpy()
            value_preds.append(out_mean_unscaled)

    value_preds = np.concatenate(value_preds, axis=0)
    if model.config.logits_output:
        std_preds = np.concatenate(std_preds, axis=0)

        return value_preds, std_preds
    return value_preds
