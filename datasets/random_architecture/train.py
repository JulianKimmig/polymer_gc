from model import RandomArchitectureDataset, RandomArchitectureEntry
from pathlib import Path
import json
from polymer_gc.model.base import PolyGCBaseModel
from polymer_gc.pipelines.training import train, TrainingConfig
from polymer_gc.pipelines.stats import (
    plot_embeddings_map,
    plot_model_embeddings,
    plot_true_vs_pred,
)
from polymer_gc import dataset
from hashlib import md5

datapath = Path(__file__).parent / "data"
datapath.mkdir(exist_ok=True)


# from json
with open("RandomArchitecture.json", "r") as f:
    pg_dataset = RandomArchitectureDataset(**json.load(f))


pg_dataset_config = dataset.PgDatasetConfig(
    embedding="random_64",
    num_bins=100,
    log_start=1,
    log_end=7,
    n_graphs=2,
    targets=["hot_encoded_architecture", "hot_encoded_sequence"],
    additional_features=[],
)

train_config = TrainingConfig(lr=1e-4, epochs=500, batch_size=64)

model_config = PolyGCBaseModel.ModelConfig(
    model="PolyGCBaseModel",
    monomer_features=64,  # 600 from PolyBERT
    num_target_properties=1,
    gc_features=64,
    num_gnn_layers=3,
    dropout_rate=0.2,
    mass_distribution_buckets=pg_dataset_config.num_bins,
    mass_distribution_reduced=6,
    additional_features=0,
    logits_output=False,
    mlp_layer=3,
    validation_loss="mae",
)

outpath = (
    datapath.parent
    / "trains"
    / md5(model_config.model_dump_json().encode("utf-8")).hexdigest()
)

outpath.mkdir(parents=True, exist_ok=True)

plot_embeddings_map(
    source_dataset=pg_dataset,
    pg_dataset_config=pg_dataset_config,
    path=outpath / "stats",
)

model, data = train(
    model_config=model_config,
    source_dataset=pg_dataset,
    pg_dataset_root=datapath,
    pg_dataset_config=pg_dataset_config,
    train_config=train_config,
    training_path=outpath,
)


plot_model_embeddings(
    model=model,
    pg_dataset_config=pg_dataset_config,
    path=outpath / "stats",
    train_loader=data["train_loader"],
    test_loader=data["test_loader"],
    val_loader=data["val_loader"],
)


plot_true_vs_pred(
    model=model,
    pg_dataset_config=pg_dataset_config,
    path=outpath / "stats",
    train_loader=data["train_loader"],
    test_loader=data["test_loader"],
    val_loader=data["val_loader"],
)
