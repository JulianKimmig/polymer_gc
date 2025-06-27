from typing import Optional, List, Tuple, Literal, Dict, Any
import torch
from torch_geometric.nn import GATConv, Sequential, SAGEConv
from torch.nn import LayerNorm
from torch.nn import ELU
from torch.nn import Dropout
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from pydantic import BaseModel, Field, model_validator, field_validator
from torch_geometric.utils import degree


import torch.nn as nn


class MeanDiffFeatureBlock(torch.nn.Module):
    def __init__(self, features):
        super(MeanDiffFeatureBlock, self).__init__()
        self.linear = Linear(features * 2, features)

    def forward(self, x, batch):
        x_mean = global_mean_pool(x, batch)
        x_mean = x_mean[batch]
        x_diff = x - x_mean
        return self.linear(torch.cat([x, x_diff], dim=-1))


class GCBlock(torch.nn.Module):
    def __init__(self, features, dropout_rate=0.3):
        super(GCBlock, self).__init__()

        self.gat = SAGEConv(in_channels=features, out_channels=features)
        self.lnorm1 = LayerNorm(features)
        self.linear1 = Linear(features, features)
        self.activation = ELU()
        self.dropout = Dropout(dropout_rate)
        self.linear2 = Linear(features, features)
        self.lnorm2 = LayerNorm(features)

    def forward(self, x, edge_index):
        y = self.gat(x, edge_index)
        y1 = y + x
        y1 = self.lnorm1(y1)  # store in differnt var for res

        y = self.linear1(y1)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.linear2(y)
        y = y + y1
        y = self.lnorm2(y)
        return y


class PoolingLayer(torch.nn.Module):
    def __init__(self, layer_size, aggr, dropout_rate=0.3):
        super().__init__()

        self.aggr = aggr
        self.lnorm1 = LayerNorm(layer_size)
        self.linear1 = Linear(layer_size, layer_size)
        self.activation = ELU()
        self.dropout = Dropout(dropout_rate)
        self.linear2 = Linear(layer_size, layer_size)
        self.lnorm2 = LayerNorm(layer_size)

    def forward(self, x, graph_index):
        h = self.aggr(x, graph_index)
        h = self.lnorm1(h)
        y = self.linear1(h)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.linear2(y)
        y = y + h
        y = self.lnorm2(y)
        return y


class LogitsOutput(nn.Module):
    def __init__(self, input_dim, num_target_properties=1, dropout_rate=0.2):
        super(LogitsOutput, self).__init__()
        mlp_out_features = 2 * num_target_properties
        self.num_target_properties = num_target_properties
        self.ll = nn.Linear(input_dim, mlp_out_features)

    def forward(self, x):
        raw_output = self.ll(x)

        reshaped_output = raw_output.view(-1, self.num_target_properties, 2)

        mean_preds = reshaped_output[
            ..., 0
        ]  # Shape: [batch_size, num_target_properties]
        log_var_preds = reshaped_output[
            ..., 1
        ]  # Shape: [batch_size, num_target_properties]

        return mean_preds, log_var_preds


class StandartScaler(nn.Module):
    def __init__(self, n, eps=1e-8):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("std", torch.ones(n))
        self.register_buffer("eps", torch.tensor(eps))

    def forward(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def inverse(self, x):
        return x * (self.std + self.eps) + self.mean

    def fit(self, x):
        """
        Fit the scaler to the data.
        Args:
            x (torch.Tensor): Input data.
        """
        if x.ndim != 2:
            raise ValueError("x must be 2D (batch_size, num_features).")
        if x.shape[1] == 0:
            self.mean = torch.zeros(x.shape[1]).to(x.device)
            self.std = torch.ones(x.shape[1]).to(x.device)
            return self
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0)

        return self


class GraphIdentity(nn.Module):
    """
    A simple identity layer for graph data.
    This is useful when you want to skip the GNN layers.
    """

    def forward(self, x, edge_index):
        return x


class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood Loss.
    Assumes model outputs mean and log_variance for each target.
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, mean_preds_log_pair, targets):
        """
        Args:
            mean_preds (torch.Tensor): Predicted means, shape [batch_size, num_target_properties].
            log_var_preds (torch.Tensor): Predicted log variances, shape [batch_size, num_target_properties].
            targets (torch.Tensor): Ground truth values, shape [batch_size, num_target_properties].
        """
        mean_preds, log_var_preds = mean_preds_log_pair
        # Clamp log_var_preds for numerical stability (adjust range as needed)
        log_var_preds = torch.clamp(log_var_preds, -10, 10)

        # Calculate NLL components for each target property
        # NLL = 0.5 * (log(sigma^2) + (y - mu)^2 / sigma^2)
        # sigma^2 = exp(log_var_preds)
        loss_components = 0.5 * (
            log_var_preds + ((targets - mean_preds) ** 2 / torch.exp(log_var_preds))
        )

        # Sum loss over the target properties dimension
        loss_per_sample = torch.sum(loss_components, dim=1)  # Shape: [batch_size]

        # Apply reduction over the batch dimension
        if self.reduction == "mean":
            return torch.mean(loss_per_sample)
        elif self.reduction == "sum":
            return torch.sum(loss_per_sample)
        else:  # 'none' or invalid
            return loss_per_sample


class RSquaredLoss(nn.Module):
    """
    R-squared loss for regression tasks.
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (torch.Tensor): Predicted values, shape [batch_size, num_target_properties].
            y_true (torch.Tensor): Ground truth values, shape [batch_size, num_target_properties].
        """
        # Calculate the total sum of squares
        ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)

        # Calculate the residual sum of squares
        ss_residual = torch.sum((y_true - y_pred) ** 2)

        # Calculate R-squared
        r_squared = torch.abs(1 - (ss_residual / ss_total))

        # Apply reduction
        if self.reduction == "mean":
            return torch.mean(r_squared)
        elif self.reduction == "sum":
            return torch.sum(r_squared)
        else:  # 'none' or invalid
            return r_squared


# loss based on variance shortfall
class VarianceLoss(nn.Module):
    """
    Variance loss for regression tasks.
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (torch.Tensor): Predicted values, shape [batch_size, num_target_properties].
            y_true (torch.Tensor): Ground truth values, shape [batch_size, num_target_properties].
        """
        # Calculate the variance of the true and predicted values
        var_true = torch.var(y_true)
        var_pred = torch.var(y_pred)

        # Calculate the penalty for variance shortfall
        penalty = torch.relu(var_true - var_pred)
        penalty = (penalty**2) / (var_true**2 + 1e-8)

        # Apply reduction
        if self.reduction == "mean":
            return torch.mean(penalty)
        elif self.reduction == "sum":
            return torch.sum(penalty)
        else:  # 'none' or invalid
            return penalty


class LossMixer(nn.Module):
    """
    A module that combines multiple loss functions.
    """

    def __init__(self, loss_functions: List[Tuple[nn.Module, float]]):
        super().__init__()
        self.loss_functions = nn.ModuleList([loss_fn for loss_fn, _ in loss_functions])
        loss_weights = [weight for _, weight in loss_functions]
        if len(self.loss_functions) != len(loss_weights):
            raise ValueError(
                "The number of loss functions must match the number of weights."
            )
        self.register_buffer(
            "loss_weights_tensor",
            torch.tensor(loss_weights, dtype=torch.float32) / sum(loss_weights),
        )

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred (torch.Tensor): Predicted values.
            y_true (torch.Tensor): Ground truth values.
        """
        # Calculate the loss for each function
        losses = [loss_fn(y_pred, y_true) for loss_fn in self.loss_functions]

        # Combine the losses using the weights
        combined_loss = torch.sum(torch.stack(losses) * self.loss_weights_tensor, dim=0)

        return combined_loss


class RMSELoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.mse = torch.nn.MSELoss(**kwargs)

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))


class PolyGCBaseModel(nn.Module):
    class ModelConfig(BaseModel):
        model: Literal["PolyGCBaseModel"] = Field(
            "PolyGCBaseModel",
            description="Model name.",
        )
        task_type: Literal["regression", "classification"] = Field(
            "regression",
            description="Type of task: regression or classification.",
        )
        monomer_features: int = Field(
            64,
            description="Number of features for each monomer.",
            gt=0,
        )
        gc_features: Optional[int] = Field(
            None,
            description="Number of features for the GNN layers. If None, it will be set to monomer_features.",
            gt=0,
        )
        num_target_properties: int = Field(
            1,
            description="Number of target properties to predict.",
            gt=0,
        )
        num_gnn_layers: int = Field(
            3,
            description="Number of GNN layers to use.",
            ge=0,
        )
        dropout_rate: float = Field(
            0.3,
            description="Dropout rate for the GNN layers.",
            ge=0.0,
            le=1.0,
        )
        mass_distribution_buckets: int = Field(
            100,
            description="Number of buckets for the mass distribution histogram.",
            ge=0,
        )
        mass_distribution_reduced: int = Field(
            6,
            description="Number of features to reduce the mass distribution to.",
            ge=0,
        )
        additional_features: int = Field(
            0,
            description="Number of additional features to include in the model.",
            ge=0,
        )

        mlp_layer: int = Field(
            3,
            description="Number of layers in the MLP.",
            ge=0,
        )
        validation_loss: Optional[Literal["mae", "mse", "rmse"]] = Field(
            None,
            description="Validation loss function to use. If None, the same loss function as training will be used.",
        )
        training_loss: Optional[Literal["mae", "mse", "rmse"]] = Field(
            None,
            description="Training loss function to use. If None, the default loss function will be used.",
        )
        test_loss: Optional[Literal["mae", "mse", "rmse"]] = Field(
            None,
            description="Test loss function to use. If None, the same loss function as training will be used.",
        )

        num_classes_per_task: Optional[List[int]] = Field(
            None,
            description="List with the number of classes for each classification task. E.g., [10, 5] for two tasks with 10 and 5 classes.",
        )
        logits_output: bool = Field(
            False,  # <-- Set to False, we'll use a simple Linear layer for logits
            description="Whether to use logits output or not.",
        )

        pooling_layers: List[Dict[str, Any]] = Field(
            default_factory=lambda: [{"type": "mean"}],
            description=(
                "A list of pooling layers to apply. Each dict specifies the 'type' "
                "and any parameters. Supported types: 'mean', 'max', 'add'. "
                "Example: [{'type': 'mean'}, {'type': 'max'}]"
            ),
        )

        @model_validator(mode="before")
        @classmethod
        def interfere_missing_values(cls, values):
            """
            Interfere missing values in the model config.
            """
            if values.get("gc_features") is None:
                values["gc_features"] = values["monomer_features"]
            if (
                values.get("task_type") == "classification"
                and values.get("num_classes_per_task") is None
            ):
                raise ValueError(
                    "`num_classes_per_task` must be provided for classification tasks."
                )
            return values

    def __init__(
        self,
        config: "PolyGCBaseModel.ModelConfig" = None,
        loss_function=None,
        training_loss_function=None,
        validation_loss_function=None,
        test_loss_function=None,
    ):
        super().__init__()
        self.config = self.ModelConfig(**config) if isinstance(config, dict) else config

        # Only create target scaler for regression tasks
        if self.config.task_type == "regression":
            self.target_scaler = StandartScaler(self.config.num_target_properties)
        else:
            self.target_scaler = None

        self.is_classification = self.config.task_type == "classification"

        self.additional_inputs_scaler = StandartScaler(self.config.additional_features)
        self.input_scaler = StandartScaler(self.config.monomer_features)

        self.inilinear = nn.Linear(
            self.config.monomer_features + 1, self.config.gc_features
        )

        if self.config.num_gnn_layers <= 0:
            self.conv = GraphIdentity()
        else:
            self.conv = Sequential(
                "x, edge_index",
                [
                    (
                        GCBlock(
                            self.config.gc_features,
                            dropout_rate=self.config.dropout_rate,
                        ),
                        "x, edge_index -> x",
                    )
                    for _ in range(self.config.num_gnn_layers)
                ],
            )

        self.mean_diff_feature_block = MeanDiffFeatureBlock(self.config.gc_features)

        # Create pooling layers as proper modules
        self.pooling_layers = nn.ModuleList()
        pooling_factory = {
            "mean": lambda: PoolingLayer(
                self.config.gc_features, global_mean_pool, self.config.dropout_rate
            ),
            "max": lambda: PoolingLayer(
                self.config.gc_features, global_max_pool, self.config.dropout_rate
            ),
            "add": lambda: PoolingLayer(
                self.config.gc_features, global_add_pool, self.config.dropout_rate
            ),
            "sum": lambda: PoolingLayer(
                self.config.gc_features, global_add_pool, self.config.dropout_rate
            ),
        }
        for pool_config in self.config.pooling_layers:
            pool_type = pool_config.get("type")
            if pool_type in pooling_factory:
                self.pooling_layers.append(pooling_factory[pool_type]())
            else:
                raise ValueError(f"Unsupported pooling type: {pool_type}")
        pooling_output_dim = self.config.gc_features * len(self.pooling_layers)

        if self.config.mass_distribution_buckets == 0:
            self.config.mass_distribution_reduced = 0

        if self.config.mass_distribution_reduced:
            self.mass_dist_reducer = nn.Linear(
                self.config.mass_distribution_buckets,
                self.config.mass_distribution_reduced,
            )
        else:
            self.mass_dist_reducer = None

        mlp_in_features = (
            pooling_output_dim
            + self.config.mass_distribution_reduced
            + self.config.additional_features
        )

        self.mlp = nn.Sequential(
            *[
                nn.Sequential(
                    *[
                        nn.Linear(mlp_in_features, mlp_in_features),
                        nn.ELU(),
                        nn.Dropout(self.config.dropout_rate),
                    ]
                )
                for _ in range(self.config.mlp_layer)
            ]
        )

        if self.is_classification:
            # The total number of output neurons is the sum of classes in all tasks
            total_output_classes = sum(self.config.num_classes_per_task)
            self.readout = nn.Linear(mlp_in_features, total_output_classes)
            self.loss = nn.CrossEntropyLoss()  # Use CrossEntropyLoss
        elif self.config.logits_output:
            self.readout = LogitsOutput(
                mlp_in_features,
                self.config.num_target_properties,
                dropout_rate=self.config.dropout_rate,
            )
            self.loss = GaussianNLLLoss(reduction="mean")
        else:
            self.readout = nn.Linear(mlp_in_features, self.config.num_target_properties)
            self.loss = RMSELoss(reduction="mean")

        if loss_function is not None:
            self.loss = loss_function

        if training_loss_function is None:
            self.training_loss_function = self.loss
        else:
            self.training_loss_function = training_loss_function

        if validation_loss_function is None:
            self.validation_loss_function = self.loss
        else:
            self.validation_loss_function = validation_loss_function

        if test_loss_function is None:
            self.test_loss_function = self.loss
        else:
            self.test_loss_function = test_loss_function

    @property
    def logits_output(self):
        return self.config.logits_output

    def batch_loss(self, batch, context: Literal["train", "val", "test"] = "train"):
        """
        Calculate the loss for a batch of data.
        Args:
            batch (torch_geometric.data.Batch): Batch of data.
        Returns:
            torch.Tensor: Loss value.
        """
        y = batch.y

        # Only scale targets for regression tasks

        lossf = getattr(self, f"{context}_loss_function", self.loss)

        y_pred = self(
            batch.x,
            batch.edge_index,
            batch.batch,
            getattr(batch, "mass_distribution", None),
            getattr(batch, "additional_features", None),
        )

        if self.is_classification:
            logits_per_task = torch.split(
                y_pred, self.config.num_classes_per_task, dim=1
            )
            total_loss = 0.0
            num_tasks = len(self.config.num_classes_per_task)

            for i in range(num_tasks):
                # Get the logits for the current task
                task_logits = logits_per_task[i]
                # Get the ground truth labels for the current task (long type is required for CrossEntropyLoss)
                task_labels = y[:, i].long()

                # Calculate loss for this task
                task_loss = lossf(task_logits, task_labels)
                total_loss += task_loss

            # Average the loss across the tasks
            return total_loss / num_tasks
        else:
            y_scaled = self.target_scaler(y)
            loss = lossf(y_pred, y_scaled)
            return loss

    def predict_embedding(self, batch):
        self.eval()
        with torch.no_grad():
            return self(
                batch.x,
                batch.edge_index,
                batch.batch,
                getattr(batch, "mass_distribution", None),
                getattr(batch, "additional_features", None),
                return_embedding=True,
            )

    def predict_proba(self, batch):
        self.eval()
        if not self.is_classification:
            raise ValueError(
                "predict_proba is only available for classification tasks."
            )
        with torch.no_grad():
            y_pred = self(
                batch.x,
                batch.edge_index,
                batch.batch,
                getattr(batch, "mass_distribution", None),
                getattr(batch, "additional_features", None),
            )
            logits_per_task = torch.split(
                y_pred, self.config.num_classes_per_task, dim=1
            )

            probs_per_task = [
                torch.softmax(logits, dim=1) for logits in logits_per_task
            ]
            return probs_per_task

    def predict(self, batch):
        self.eval()
        with torch.no_grad():
            y_pred = self(
                batch.x,
                batch.edge_index,
                batch.batch,
                getattr(batch, "mass_distribution", None),
                getattr(batch, "additional_features", None),
            )

            # Only apply inverse scaling for regression tasks
            if self.is_classification:
                # --- START NEW LOGIC FOR CLASSIFICATION ---
                logits_per_task = torch.split(
                    y_pred, self.config.num_classes_per_task, dim=1
                )

                predicted_indices = []
                for task_logits in logits_per_task:
                    # Get the index of the max logit for each sample in the batch
                    preds = torch.argmax(task_logits, dim=1)
                    predicted_indices.append(preds.unsqueeze(1))

                # Stack the predictions to get a [batch_size, 2] tensor
                return torch.cat(predicted_indices, dim=1)
                # --- END NEW LOGIC FOR CLASSIFICATION ---

            else:
                if self.config.logits_output:
                    mean_preds, log_var_preds = y_pred
                    mean_preds = self.target_scaler.inverse(mean_preds)
                    # Adjust variance for unscaling
                    std = self.target_scaler.std + self.target_scaler.eps
                    log_var_preds = log_var_preds + 2 * torch.log(std)
                    return (
                        mean_preds,
                        log_var_preds,
                    )  # or exp(log_var_preds) for actual variance
                else:
                    y_pred = self.target_scaler.inverse(y_pred)
                    return y_pred

    def forward(
        self,
        x,
        edge_index,
        batch,
        mass_distribution=None,
        additional_features=None,
        return_embedding=False,
    ):
        x = self.input_scaler(x)

        degree_x = degree(edge_index[0], num_nodes=x.shape[0])
        x = torch.cat([x, degree_x.unsqueeze(1)], dim=-1)

        nx = self.inilinear(x)
        nx = self.conv(nx, edge_index)
        nx = self.mean_diff_feature_block(nx, batch)
        pooled_outputs = [pool_func(nx, batch) for pool_func in self.pooling_layers]
        x_pooled_cat = torch.cat(pooled_outputs, dim=-1)

        cat_tensors = [x_pooled_cat]
        if additional_features is not None:
            cat_tensors.append(self.additional_inputs_scaler(additional_features))
        if self.mass_dist_reducer is not None and mass_distribution is not None:
            # batchwise area normalization of mass_distribution

            mass_distribution = mass_distribution / torch.sum(
                mass_distribution, dim=1, keepdim=True
            )
            cat_tensors.append(self.mass_dist_reducer(mass_distribution))

        x_merged = torch.cat(
            cat_tensors,
            dim=-1,
        )

        x_readout = self.mlp(x_merged)

        if return_embedding:
            return x_readout
        else:
            return self.readout(x_readout)

    def prefit(self, x=None, y=None, additional_inputs=None):
        """
        Fit the target scaler to the data.
        Args:
            y (torch.Tensor): Target data.
        """
        if x is not None:
            if x.ndim != 2:
                raise ValueError("x must be 2D (batch_size, features).")
            if x.shape[1] != self.config.monomer_features:
                raise ValueError(
                    f"x must have {self.config.monomer_features} features."
                )
            if x.shape[0] > 0:
                # Fit the input scaler to the data
                self.input_scaler.fit(x)
        if y is not None and not self.is_classification:
            if y.ndim != 2:
                raise ValueError("y must be 2D (batch_size, num_target_properties).")
            if y.shape[1] != self.config.num_target_properties:
                raise ValueError(
                    f"y must have {self.config.num_target_properties} target properties."
                )
            if y.shape[0] > 0:
                # Fit the target scaler to the data (only for regression)
                self.target_scaler.fit(y)
        if additional_inputs is not None:
            if additional_inputs.ndim != 2:
                raise ValueError("additional_inputs must be 2D (batch_size, features).")
            if additional_inputs.shape[1] != self.config.additional_features:
                raise ValueError(
                    f"additional_inputs must have {self.config.additional_features} features."
                )
            if additional_inputs.shape[0] > 0:
                self.additional_inputs_scaler.fit(additional_inputs)
        return self
