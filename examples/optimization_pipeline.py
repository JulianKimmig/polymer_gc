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
fill_db_bj(
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
            min=0.15,
            max=0.3,
            step=0.01,
            path=["model_conf"],
        ),
        OptimizationVar(
            name="model_pooling",
            attribute="pooling_layers",
            type="k_out_of_n",
            choices=[{"type": "max"}, {"type": "mean"}, {"type": "sum"}, {"type": "add"}],
            k_min=1,
            k_max=3,
            path=["model_conf"],
        ),
        OptimizationVar(
            name="model_mass_distribution_reduced",
            attribute="mass_distribution_reduced",
            type="int",
            min=2**0,
            max=2**4,
            step=1,
            log=True,
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
            name="batch_size",
            attribute="batch_size",
            type="int",
            min=2**4.5,
            max=2**5.5,
            step=1,
            log=True,
        ),
        OptimizationVar(
            name="epochs_jb",
            attribute="epochs_jb",
            type="int",
            min=50,
            max=150,
            step=1,
        ),
        OptimizationVar(
            name="epochs_bj",
            attribute="epochs_bj",
            type="int",
            min=1,
            max=10,
            step=1,
        ),
    ]
)



def custom_training_pipeline(
    ds_name:str,
    conf:dict,
    result_dir:Path,
    pretrained_model_dir:Optional[Path]=None,
    cached_data:Optional[dict]=None,
    **kwargs,
):
    
    if cached_data is None:
        cached_data = {}
    
    conf["epochs"] = conf.pop("epochs_jb")
    pretrained_model_dir = None
    pretrain_result,data = taining_pipeline(
        ds_name="tg_jablonka",
        result_dir=result_dir / "tg_jablonka",
        pretrained_model_dir=pretrained_model_dir,
        conf=conf,
        cached_data=cached_data.get("tg_jablonka"),
        **kwargs,
    )
    cached_data.setdefault("tg_jablonka",data["all_graph_data"])
   
    conf["epochs"] = conf.pop("epochs_bj")
    pretrained_model_dir = result_dir / "tg_jablonka"
    train_result,data = taining_pipeline(
        ds_name="tg_bayreuth_jena",
        result_dir=result_dir / "tg_bayreuth_jena",
        pretrained_model_dir=pretrained_model_dir,
        conf=conf,
        cached_data=cached_data.get("tg_bayreuth_jena"),
        **kwargs,
    )
    cached_data.setdefault("tg_bayreuth_jena",data["all_graph_data"])

    return train_result,{"all_graph_data":cached_data}


optimize(
    result_dir=Path(__file__).parent / "results" / "tg_jablonka_br",
    study_name="tg_jablonka_br",
    optimization_conf=optimization_conf,
    k_folds=5,
    ds_name="tg_bayreuth_jena", db_path=db_path,
    training_pipeline_=custom_training_pipeline,
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
