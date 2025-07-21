from pathlib import Path
from typing import Optional
from polymer_gc.pipelines.training import taining_pipeline,TrainingConf
from polymer_gc.pipelines.optimize import optimize,OptimizationConf,OptimizationVar
from rich import print

from polymer_gc.datasets.tg_jablonka import populate as fill_db_jb
from polymer_gc.datasets.tg_bayreuth_jena import populate as fill_db_bj
from pathlib import Path

db_path = Path(__file__).parent / "database.db"
fill_db_jb(
    db_path=db_path,
)

optimization_conf = OptimizationConf(
    vars=[
        OptimizationVar(
            name="model_gc_features",
            attribute="gc_features",
            type="int",
            min=2**4,
            max=2**8,
            log=True,
            path=["model_conf"],
        ),
        OptimizationVar(
            name="model_num_gnn_layers",
            attribute="num_gnn_layers",
            type="int",
            min=2,
            max=6,
            step=1,
            path=["model_conf"],
        ),
        OptimizationVar(
            name="model_mlp_layer",
            attribute="mlp_layer",
            type="int",
            min=2,
            max=6,
            step=1,
            path=["model_conf"],
        ),
        OptimizationVar(
            name="model_dropout_rate",
            attribute="dropout_rate",
            type="float",
            min=0.1,
            max=0.2,
            step=0.01,
            path=["model_conf"],
        ),
        OptimizationVar(
            name="model_pooling",
            attribute="pooling_layers",
            type="k_out_of_n",
            choices=[{"type": "max"}, {"type": "mean"}, {"type": "sum"}, {"type": "add"}],
            choice_names=["max", "mean", "sum", "add"],
            k_min=1,
            k_max=4,
            path=["model_conf"],
        ),
        OptimizationVar(
            name="learning_rate",
            attribute="learning_rate",
            type="float",
            min=1e-4,
            max=1e-3,
            log=True,
        ),
        OptimizationVar(
            name="epochs",
            attribute="epochs",
            type="int",
            min=10,
            max=150,
            step=5,
        ),
    ]
)


optimize(
    result_dir=Path(__file__).parent / "results" / "tg_jablonka",
    study_name="tg_jablonka",
    optimization_conf=optimization_conf,
    k_folds=5,
    ds_name="tg_jablonka", db_path=db_path,
    conf={"batch_size":2**6},
    storage=f"sqlite:///{Path(__file__).parent /  'optimization.db'}",
)


# DB_PATH = (
#     Path(__file__).parent.parent / "datasets" / "end_to_end_tg_polybert" / "database_br.db" # / "database_jb.db"
# )
# PRE_DS_NAME = "Tg_Prediction_from_CSV"
# DS_NAME = "tg_bayreuth"# "Tg_Prediction_from_CSV"



# taining_pipeline(
#     ds_name=DS_NAME, db_path=DB_PATH, data_dir=DB_PATH.parent / "data", k_folds=5,
#     result_dir=Path(__file__).parent / "results" / DS_NAME,
#     pretrained_model_dir=Path(__file__).parent / "results" / PRE_DS_NAME ,
#     conf={"epochs":2},
# )
