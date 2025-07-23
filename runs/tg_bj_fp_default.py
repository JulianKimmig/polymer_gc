from polymer_gc.data.dataset import PgDatasetConfig
from polymer_gc.model.base import PolyGCBaseModel
from polymer_gc.pipelines.training import taining_pipeline,TrainingConf
from pathlib import Path
from typing import Optional, List, Any
from polymer_gc.pipelines.training import taining_pipeline,TrainingConf
from polymer_gc.pipelines.optimize import optimize,OptimizationConf,OptimizationVar
from rich import print
import torch
import numpy as np
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader

from polymer_gc.datasets.tg_bayreuth_jena import populate as fill_db_bj
from polymer_gc.datasets.tg_jablonka import populate as fill_db_jb
from polymer_gc.visualization import create_kfold_visualization_suite
from polymer_gc.pipelines.evaluation import evaluation_pipeline
from pathlib import Path

db_path = Path(__file__).parent / "database.db"

basefolder = Path(__file__).parent / "results" / "tg_bayreuth_default"
db_path = Path(__file__).parent / "database.db"
fill_db_jb(
    db_path=db_path,
    pg_dataset_config=PgDatasetConfig(
        embedding="RDKitFP",
        targets=["Tg"],
    ),
)
fill_db_bj(
    db_path=db_path,
    pg_dataset_config=PgDatasetConfig(
        embedding="RDKitFP",
        targets=["Tg"],
    ),
)
fill_db_bj(
    db_path=db_path,
    flory_fox=False,
    pg_dataset_config=PgDatasetConfig(
        embedding="RDKitFP",
        targets=["Tg"],
    ),
)

model_conf=PolyGCBaseModel.ModelConfig(
            task_type="regression",
            monomer_features=2048, 
            gc_features=256, 
            num_target_properties=1,
            num_gnn_layers=4,
            mlp_layer=3,
            dropout_rate=0.2,
            pooling_layers=[{"type": "mean"}, {"type": "max"}, {"type": "sum"}],
            logits_output=False,
            mass_distribution_reduced=10,
        )


res_jb = taining_pipeline(
    ds_name="tg_jablonka",
    result_dir=basefolder /"jb",
    conf=TrainingConf(
        batch_size=32,
        epochs=200,
        model_conf=model_conf,
    ),
    db_path= db_path,
    reduce_indentical=False,
)

eval_result_jb, eval_data_jb = evaluation_pipeline(
    ds_name="tg_jablonka",
    trained_models_dir=basefolder / "jb",
    db_path=db_path,
    reduce_identical=False,
    output_dir=basefolder / "jb" / "evaluation",
    property_name="Tg",
    unit="K"
)

res_wflory = taining_pipeline(
    ds_name="tg_bayreuth_jena",
    result_dir=basefolder /"wflory",
    pretrained_model_dir=basefolder /"jb",
    conf=TrainingConf(
        batch_size=32,
        epochs=50,
        model_conf=model_conf,
    ),
    db_path= db_path,
    reduce_indentical=False,
)

eval_result_wflory, eval_data_wflory = evaluation_pipeline(
    ds_name="tg_bayreuth_jena",
    trained_models_dir=basefolder / "wflory",
    db_path=db_path,
    k_folds=5,
    seed=42,
    reduce_identical=False,
    output_dir=basefolder / "wflory" / "evaluation",
    property_name="Tg",
    unit="K"
)

res = taining_pipeline(
    ds_name="tg_bayreuth_jena_no_flory_fox",
    result_dir=basefolder /"fin",
    pretrained_model_dir=basefolder /"wflory",
    conf=TrainingConf(
        batch_size=32,
        epochs=4,
        model_conf=model_conf,
    ),
    db_path= db_path,
    reduce_indentical=False,
)

eval_result_fin, eval_data_fin = evaluation_pipeline(
    ds_name="tg_bayreuth_jena_no_flory_fox",
    trained_models_dir=basefolder / "fin",
    db_path=db_path,
    k_folds=5,
    seed=42,
    reduce_identical=False,
    output_dir=basefolder / "fin" / "evaluation",
    property_name="Tg",
    unit="K"
)

