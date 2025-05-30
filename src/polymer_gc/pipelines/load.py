import torch
import os
from typing import Optional
import polymer_gc
import importlib

MODELS = {"PolyGCBaseModel": polymer_gc.model.base.PolyGCBaseModel}


def load(
    model: Optional[torch.nn.Module] = None,
    model_path: Optional[str] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    optimizer_path: Optional[str] = None,
):
    model_loaded = False
    optimizer_loaded = False
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model_loaded = True
    if optimizer_path and os.path.exists(optimizer_path):
        optimizer.load_state_dict(torch.load(optimizer_path))
        optimizer_loaded = True

    return model_loaded, optimizer_loaded


def load_modelconfig(
    model_config: polymer_gc.model.base.PolyGCBaseModel.ModelConfig,
) -> polymer_gc.model.base.PolyGCBaseModel:
    model_cls = MODELS.get(model_config.model, None)

    # try import name if it is given as module string e.g. polymer_gc.model.base.PolyGCBaseModel
    if model_cls is None:
        try:
            module_name, class_name = model_config.model.rsplit(".", 1)
            module = importlib.import_module(module_name)
            model_cls = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Model {model_config.model} not found. Error: {e}")
    if model_cls is None:
        raise ValueError(f"Model {model_config.model} not found in available models.")
    # Remove model name from config

    # Initialize the model with the provided config

    model_cls_config = model_cls.ModelConfig(**model_config.model_dump())

    model = model_cls(config=model_cls_config)

    if not isinstance(model, polymer_gc.model.base.PolyGCBaseModel):
        raise ValueError(f"Model {model_config.model} is not a valid PolyGCBaseModel.")
    return model
