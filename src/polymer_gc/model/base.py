from typing import Optional, List, Tuple, Literal
import torch
from torch_geometric.nn import GATConv, Sequential, SAGEConv
from torch.nn import LayerNorm
from torch.nn import ELU
from torch.nn import Dropout
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
from pydantic import BaseModel, Field, model_validator, field_validator


import torch.nn as nn


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


class MeanPooling(torch.nn.Module):
    def __init__(self, layer_size, dropout_rate=0.3):
        super().__init__()

        self.aggr = global_mean_pool
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
    def __init__(self, mean=0, std=1, eps=1e-8):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))
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
            gt=0,
        )
        mass_distribution_reduced: int = Field(
            6,
            description="Number of features to reduce the mass distribution to.",
            gt=0,
        )
        additional_features: int = Field(
            0,
            description="Number of additional features to include in the model.",
            ge=0,
        )
        logits_output: bool = Field(
            True,
            description="Whether to use logits output or not.",
        )

        mlp_layer: int = Field(
            3,
            description="Number of layers in the MLP.",
            ge=0,
        )

        @model_validator(mode="before")
        @classmethod
        def interfere_missing_values(cls, values):
            """
            Interfere missing values in the model config.
            """
            if values.get("gc_features") is None:
                values["gc_features"] = values["monomer_features"]
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

        self.target_scaler = StandartScaler()
        self.additional_inputs_scaler = StandartScaler()

        self.inilinear = nn.Linear(
            self.config.monomer_features, self.config.gc_features
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

        self.pooling = MeanPooling(
            self.config.gc_features, dropout_rate=self.config.dropout_rate
        )
        self.mass_dist_reducer = nn.Linear(
            self.config.mass_distribution_buckets, self.config.mass_distribution_reduced
        )
        mlp_in_features = (
            self.config.gc_features
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

        if self.config.logits_output:
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
        y_scaled = self.target_scaler(y)
        y_pred = self(
            batch.x,
            batch.edge_index,
            batch.batch,
            batch.mass_distribution,
            batch.additional_features,
        )

        lossf = getattr(self, f"{context}_loss_function", self.loss)
        loss = lossf(y_pred, y_scaled)
        return loss

    def predict(self, batch):
        self.eval()
        with torch.no_grad():
            y_pred = self(
                batch.x,
                batch.edge_index,
                batch.batch,
                batch.mass_distribution,
                batch.additional_features,
            )

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

    def forward(self, x, edge_index, batch, mass_distribution, additional_features):
        nx = self.inilinear(x)
        nx = self.conv(nx, edge_index)

        # batchwise area normalization of mass_distribution
        mass_distribution = mass_distribution / torch.sum(
            mass_distribution, dim=1, keepdim=True
        )
        x_pooled = self.pooling(nx, batch)
        x_merged = torch.cat(
            (
                x_pooled,
                self.mass_dist_reducer(mass_distribution),
                self.additional_inputs_scaler(additional_features),
            ),
            dim=-1,
        )

        x_readout = self.mlp(x_merged)

        return self.readout(x_readout)

    def prefit(self, y=None, additional_inputs=None):
        """
        Fit the target scaler to the data.
        Args:
            y (torch.Tensor): Target data.
        """
        if y is not None:
            if y.ndim != 2:
                raise ValueError("y must be 2D (batch_size, num_target_properties).")
            if y.shape[1] != self.config.num_target_properties:
                raise ValueError(
                    f"y must have {self.config.num_target_properties} target properties."
                )
            if y.shape[0] > 0:
                # Fit the target scaler to the data
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
