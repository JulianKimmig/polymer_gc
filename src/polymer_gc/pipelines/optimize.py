import optuna
from copy import deepcopy
from typing import Optional, Union, List, TypeAlias, Dict, Any
from pathlib import Path
import numpy as np
from optuna.samplers import TPESampler
from pydantic import BaseModel, Field, model_validator
from .training import taining_pipeline, TrainingConf    
from hashlib import md5
class OptimizationVar(BaseModel):
    name: str = Field()
    attribute: str = Field()
    type: str = Field(default="float")
    min: Optional[float] = Field(default=None)
    max: Optional[float] = Field(default=None)
    step: Optional[float] = Field(default=1)
    log: Optional[bool] = Field(default=False)
    path: List[str] = Field(default_factory=list)
    choices: Optional[List[Any]] = Field(None)
    k: Optional[int] = Field(default=1)
    k_min: Optional[int] = Field(None)
    k_max: Optional[int] = Field(None)

    @model_validator(mode="after")
    def check_k_min_max(self):
        if self.type in ["int", "float"]:
            assert self.min is not None and self.max is not None, (
                "min and max must be set for int and float"
            )
            assert self.min < self.max, "min must be < max"
        return self


class OptimizationConf(BaseModel):
    vars: List[OptimizationVar]

    @model_validator(mode="after")
    def check_vars(self):
        # make shure each var has a unique name
        names = [var.name for var in self.vars]
        assert len(names) == len(set(names)), (
            f"Each var must have a unique name: {names}"
        )
        return self


def make_objective_kfold(
        optimization_conf: OptimizationConf,
        result_dir:Path,
        conf: Optional[Union[TrainingConf, dict]] = None,
        training_pipeline_=taining_pipeline,
        **kwargs,
    ):
    if isinstance(conf, TrainingConf):
        conf = conf.model_dump()
    if conf is None:
        conf = {}
    cached_data:Optional[list] = None
    def objective_kfold(
        trial: optuna.Trial,
    ):
        nonlocal cached_data
        
        for var in optimization_conf.vars:
            print(var)
            if var.type == "int":
                value = trial.suggest_int(
                    var.name,
                    var.min,
                    var.max,
                    step=var.step,
                    log=var.log,
                )

            elif var.type == "float":
                value = trial.suggest_float(
                    var.name,
                    var.min,
                    var.max,
                    step=var.step if not var.log else None,
                    log=var.log,
                )
            elif var.type == "categorical":
                value = trial.suggest_categorical(var.name, var.choices)
            elif var.type == "k_out_of_n":
                if var.k_min is not None or var.k_max is not None:
                    assert var.k_min is not None and var.k_max is not None, (
                        "k_min and k_max must be set both or not for k_out_of_n"
                    )
                    assert var.k_min >= 0, "k_min must be >= 0"
                    assert var.k_max >= var.k_min, "k_max must be >= k_min"
                    k = trial.suggest_int(f"{var.name}_k", var.k_min, var.k_max)
                else:
                    k = var.k

                scores = [
                    trial.suggest_float(f"{var.name}_pick_{i}", 0, 1)
                    for i in range(len(var.choices))
                ]
                selected_indices = np.argsort(scores)[-k:]
                value = [var.choices[i] for i in selected_indices]

            subconf = conf
            for path in var.path:
                if path not in subconf:
                    subconf[path] = {}
                subconf = subconf[path]
            subconf[var.attribute] = value

        res,data = training_pipeline_(conf=deepcopy(conf),result_dir=result_dir/f"trial_{trial.number}",cached_data=cached_data, **kwargs)
        cached_data = data["all_graph_data"]
        return res.test_mae.mean

    return objective_kfold



def optimize(
    result_dir: Path,
    study_name: str,
    optimization_conf: OptimizationConf,
    storage: Optional[str] = None,
    direction: str = "minimize",
    k_folds: int = 5,
    n_trials: int = 100,
    training_pipeline_=taining_pipeline,
    **kwargs,
):
    conf_dict = optimization_conf.model_dump()
    conf_dict["k_folds"] = k_folds
    conf_dict.update({k:str(v) for k,v in kwargs.items()})
    hashed_conf = md5(str(conf_dict).encode()).hexdigest()
    study_name = study_name or result_dir.name
    study_name += f"_{hashed_conf}"
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    if storage is None:
        storage = f"sqlite:///{result_dir}/optimization.db"

    def default_gamma(x: int) -> int:
        return min(int(np.ceil(0.25 * x)), 25)

    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=TPESampler(
            gamma=default_gamma,
            multivariate=True,
        ),
    )
    for k,v in conf_dict.items():
        study.set_user_attr(k,v)

    # Create objective function with fixed k_folds
    objective_func = make_objective_kfold(
        k_folds=k_folds,
        optimization_conf=optimization_conf,
        result_dir=result_dir,
        training_pipeline_=training_pipeline_,
        **kwargs,
    )

    study.optimize(objective_func, n_trials=n_trials)

    print(f"\n{'=' * 60}")
    print(f"HYPERPARAMETER OPTIMIZATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value (Mean Test MAE): {study.best_value:.4f} K")
    print(f"Best params: {study.best_params}")

    # Print additional metrics
    best_trial = study.best_trial
    if hasattr(best_trial, "user_attrs"):
        print(f"Jablonka MAE: {best_trial.user_attrs.get('jablonka_mae', 'N/A'):.4f} K")
        print(f"Bayreuth MAE: {best_trial.user_attrs.get('bayreuth_mae', 'N/A'):.4f} K")
        print(
            f"Final MAE Std: {best_trial.user_attrs.get('final_mae_std', 'N/A'):.4f} K"
        )

    return study
