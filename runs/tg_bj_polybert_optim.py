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
from polymer_gc.visualization import create_kfold_visualization_suite
from pathlib import Path

db_path = Path(__file__).parent / "database.db"


model_conf=PolyGCBaseModel.ModelConfig(
            task_type="regression",
            monomer_features=600, 
            gc_features=41, 
            num_target_properties=1,
            num_gnn_layers=2,
            mlp_layer=2,
            dropout_rate=0.155,
            pooling_layers=[{"type": "mean"}, {"type": "max"}, {"type": "sum"}],
            logits_output=False,
            mass_distribution_reduced=8
        )


res = taining_pipeline(
    ds_name="tg_jablonka",
    result_dir=Path(__file__).parent / "results" / "tg_bayreuth_optimum"/"jb",
    conf=TrainingConf(
        batch_size=32,
        epochs=150,
        model_conf=model_conf,
    ),
    db_path= db_path,
    reduce_indentical=False,
)

# Create visualizations for tg_jablonka training
kfold_result, data_dict = res
create_kfold_visualization_suite(
    kfold_result=kfold_result,
    all_graph_data=data_dict["all_graph_data"],
    dataset_name="tg_jablonka",
    output_dir=Path(__file__).parent / "results" / "tg_bayreuth_optimum"/"jb" / "visualizations",
    create_tsne=True
)


res = taining_pipeline(
    ds_name="tg_bayreuth_jena",
    result_dir=Path(__file__).parent / "results" / "tg_bayreuth_optimum"/"wflory",
    pretrained_model_dir=Path(__file__).parent / "results" / "tg_bayreuth_optimum"/"jb",
    conf=TrainingConf(
        batch_size=32,
        epochs=50,
        model_conf=model_conf,
    ),
    db_path= db_path,
    reduce_indentical=False,
)

res = taining_pipeline(
    ds_name="tg_bayreuth_jena_no_flory_fox",
    result_dir=Path(__file__).parent / "results" / "tg_bayreuth_optimum"/"fin",
    pretrained_model_dir=Path(__file__).parent / "results" / "tg_bayreuth_optimum"/"wflory",
    conf=TrainingConf(
        batch_size=32,
        epochs=4,
        model_conf=model_conf,
    ),
    db_path= db_path,
    reduce_indentical=False,
)