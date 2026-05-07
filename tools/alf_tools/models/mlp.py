# Copyright 2023 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from alf_core import BaseModel, Candidate, LabelledCandidates, Modality, Predictions, Results
from alf_core.dataclasses.surrogate_epoch_metrics import SurrogateEpochMetrics
from torch.utils.data import DataLoader, TensorDataset

from alf_tools.models.utils import get_device

logger = logging.getLogger("alf-tools")


@dataclass
class MLPModelConfig:
    """Configuration for MLP model architecture.

    Args:
        hidden_dims: Sizes of hidden layers; length determines depth.
        activation: Activation function applied after each hidden layer's norm.
        norm: Normalisation applied before activation; "none" skips it.
        dropout: Dropout probability applied after activation in each hidden layer.
        n_mc_passes: Number of stochastic forward passes at inference for MC dropout.
            0 disables MC dropout and returns means only.
        model_seed: Global seed for weight initialisation and training data shuffling.
            Also used as the dropout generator seed when dropout_seed is None.
            Note: two models with the same seed will have identical initial weights —
            pass distinct seeds for ensemble members.
        dropout_seed: If set, overrides model_seed exclusively for the MC dropout
            pass generator.
    """

    hidden_dims: list[int] = field(default_factory=lambda: [256, 128])
    activation: Literal["relu", "gelu", "silu"] = "relu"
    norm: Literal["none", "batch", "layer"] = "none"
    dropout: float = 0.0
    n_mc_passes: int = 0
    model_seed: int = 0
    dropout_seed: int | None = None

    def __post_init__(self) -> None:
        """Validate that n_mc_passes > 0 requires dropout > 0.

        Raises:
            ValueError: If n_mc_passes > 0 and dropout is not positive.
        """
        if self.n_mc_passes < 0:
            raise ValueError(f"n_mc_passes must be >= 0, got {self.n_mc_passes}")
        if self.n_mc_passes > 0 and self.dropout <= 0.0:
            raise ValueError(
                f"dropout must be > 0 when n_mc_passes > 0, got dropout={self.dropout}"
            )


@dataclass
class MLPTrainConfig:
    """Configuration for MLP training.

    Args:
        learning_rate: Learning rate for the optimiser.
        batch_size: Mini-batch size.
        num_epochs: Number of training epochs. Must be >= 1.
        optimizer: Optimiser type; "adam" or "adamw".
        weight_decay: L2 regularisation coefficient.
        log_frequency: Log a training summary every this many epochs.
    """

    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 50
    optimizer: Literal["adam", "adamw"] = "adam"
    weight_decay: float = 0.0
    log_frequency: int = 10

    def __post_init__(self) -> None:
        """Validate training configuration.

        Raises:
            ValueError: If num_epochs < 1.
        """
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")


class MLP(nn.Module):
    """Feedforward MLP for scalar regression on pre-computed feature vectors.

    Architecture:
        input → [Linear → Norm → Activation → Dropout] × depth → Linear → scalar
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activation: Literal["relu", "gelu", "silu"],
        norm: Literal["none", "batch", "layer"],
        dropout: float,
        model_seed: int = 0,
    ):
        """Build the hidden block and output layer, seeded deterministically."""
        super().__init__()
        torch.manual_seed(model_seed)

        _activation_map: dict[str, type[nn.Module]] = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        activation_cls = _activation_map[activation]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if norm == "batch":
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif norm == "layer":
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.hidden_block = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return scalar predictions of shape (batch,).

        Returns:
            1-D tensor of scalar predictions, one per input row.
        """
        return self.output_layer(self.hidden_block(x)).squeeze(-1)


class MLPModel(BaseModel):
    """Surrogate model wrapping MLP for pre-computed vector inputs.

    Featurisation is a passthrough — inputs must arrive as TABULAR or EMBEDDING
    candidates whose data is a numpy array or torch tensor.
    """

    def __init__(
        self,
        name: str = "mlp_model",
        model_config: MLPModelConfig | None = None,
        train_config: MLPTrainConfig | None = None,
        device: str | None = None,
    ):
        """Initialise MLPModel with optional config overrides; network is built lazily."""
        self.name = name
        self.model_config = model_config or MLPModelConfig()
        self.train_config = train_config or MLPTrainConfig()
        self.device = get_device(device)
        self.net: MLP | None = None
        self.input_dim: Optional[int] = None
        self.training_metrics: dict[str, Union[float, int, np.number]] = {}
        self._epoch_metrics: list[SurrogateEpochMetrics] = []

    def featurise(self, inputs: Union[LabelledCandidates, list[Candidate]]) -> torch.Tensor:
        """Convert TABULAR or EMBEDDING candidates to a float32 tensor.

        Args:
            inputs: Either LabelledCandidates or a list of Candidates. Each
                candidate's data must be a numpy array or torch tensor.

        Returns:
            Float32 tensor of shape (n_candidates, feature_dim).

        Raises:
            ValueError: If any candidate has an unsupported modality, or if
                inputs is not a LabelledCandidates or list.
        """
        if isinstance(inputs, LabelledCandidates):
            candidates = inputs.candidates
        elif isinstance(inputs, list):
            candidates = inputs
        else:
            raise ValueError("Input must be LabelledCandidates or list of Candidate")

        for c in candidates:
            if c.modality not in (Modality.TABULAR, Modality.EMBEDDING):
                raise ValueError(
                    f"MLPModel only supports TABULAR and EMBEDDING modalities, got {c.modality}"
                )

        arrays = []
        for c in candidates:
            if isinstance(c.data, torch.Tensor):
                arrays.append(c.data.float().cpu().numpy())
            else:
                arrays.append(np.asarray(c.data, dtype=np.float32))

        return torch.tensor(np.stack(arrays), dtype=torch.float32)

    def sample(self, condition: Any | None = None) -> list[Candidate]:
        """Not implemented; raises NotImplementedError."""
        raise NotImplementedError("Sampling is not implemented for MLPModel.")

    def get_epoch_metrics(self) -> list[SurrogateEpochMetrics]:
        """Return per-epoch metrics recorded during the most recent train() call.

        Returns:
            List of SurrogateEpochMetrics, one per epoch; empty before first train().
        """
        return self._epoch_metrics

    def get_training_summary_metrics(self) -> dict[str, Union[float, int, np.number]]:
        """Return scalar summary metrics from the most recent train() call.

        Returns:
            Dict of metric names to values; empty before first train().
        """
        return self.training_metrics

    def _train_epoch(
        self, loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module
    ) -> dict:
        """Run one full training epoch.

        Args:
            loader: DataLoader over the training set.
            optimizer: Optimiser used for gradient updates.
            criterion: Loss function.

        Returns:
            Dict with keys ``loss`` (float) and any additional metrics from
            ``Results.metrics`` (e.g. ``spearman``, ``mse``).
        """
        assert self.net is not None
        self.net.train()
        train_losses: list[float] = []
        train_preds_list: list[np.ndarray] = []
        train_targets_list: list[np.ndarray] = []

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = self.net(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_preds_list.append(preds.detach().cpu().numpy())
            train_targets_list.append(batch_y.detach().cpu().numpy())

        avg_loss = float(np.mean(train_losses))
        all_preds = np.concatenate(train_preds_list)
        all_targets = np.concatenate(train_targets_list)

        if len(all_preds) >= 2:
            metrics = Results(predictions=Predictions(means=all_preds), targets=all_targets).metrics
        else:
            metrics = {"mse": float(np.mean((all_preds - all_targets) ** 2))}

        return {"loss": avg_loss, **metrics}

    def _validate_epoch(self, loader: DataLoader, criterion: nn.Module) -> dict:
        """Run one full validation epoch.

        Args:
            loader: DataLoader over the validation set.
            criterion: Loss function.

        Returns:
            Dict with key ``loss`` (float) and any additional metrics from
            ``Results.metrics`` (e.g. ``spearman``, ``mse``).
        """
        assert self.net is not None
        self.net.eval()
        val_losses: list[float] = []
        val_preds_list: list[np.ndarray] = []
        val_targets_list: list[np.ndarray] = []

        with torch.no_grad():
            for batch_x, batch_y in loader:
                preds = self.net(batch_x)
                loss = criterion(preds, batch_y)
                val_losses.append(loss.item())
                val_preds_list.append(preds.cpu().numpy())
                val_targets_list.append(batch_y.cpu().numpy())

        avg_loss = float(np.mean(val_losses))
        all_preds = np.concatenate(val_preds_list)
        all_targets = np.concatenate(val_targets_list)

        if len(all_preds) >= 2:
            metrics = Results(predictions=Predictions(means=all_preds), targets=all_targets).metrics
        else:
            metrics = {"mse": float(np.mean((all_preds - all_targets) ** 2))}

        return {"loss": avg_loss, **metrics}

    def train(
        self,
        train_data: LabelledCandidates,
        val_data: LabelledCandidates | None = None,
    ) -> None:
        """Fit the MLP to train_data, optionally tracking val_data metrics per epoch.

        The network is initialised on the first call and re-used on subsequent calls
        (warm-start). Epoch metrics are reset at the start of each call.

        Args:
            train_data: Labelled candidates used for gradient updates.
            val_data: Optional labelled candidates for per-epoch validation loss.

        Raises:
            ValueError: If called after a previous train() with data of a different
                feature dimension without calling cleanup() first.
        """
        self._epoch_metrics = []

        # Featurise first so modality errors surface here with a clear message,
        # and so we know the true input_dim before building the network.
        x_train = self.featurise(train_data).to(self.device)

        # Guard against warm-start with mismatched feature dimension.
        if self.net is not None and x_train.shape[1] != self.input_dim:
            raise ValueError(
                f"Input dimension changed from {self.input_dim} to {x_train.shape[1]}. "
                "Call cleanup() before retraining with different-dimensional data."
            )

        if self.net is None:
            input_dim = x_train.shape[1]
            self.input_dim = input_dim
            self.net = MLP(
                input_dim=input_dim,
                hidden_dims=self.model_config.hidden_dims,
                activation=self.model_config.activation,
                norm=self.model_config.norm,
                dropout=self.model_config.dropout,
                model_seed=self.model_config.model_seed,
            ).to(self.device)
            logger.info(
                f"MLPModel '{self.name}' initialised with input_dim={input_dim}, "
                f"hidden_dims={self.model_config.hidden_dims}, "
                f"params={sum(p.numel() for p in self.net.parameters()):,}"
            )

        y_train = torch.tensor(train_data.labels, dtype=torch.float32).to(self.device)

        # Use an isolated generator for DataLoader shuffle — avoids clobbering the
        # process-wide RNG used by other models or library code.
        g = torch.Generator()
        g.manual_seed(self.model_config.model_seed)
        train_loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=self.train_config.batch_size,
            shuffle=True,
            generator=g,
        )

        val_loader: DataLoader | None = None
        if val_data is not None and len(val_data) > 0:
            x_val = self.featurise(val_data).to(self.device)
            y_val = torch.tensor(val_data.labels, dtype=torch.float32).to(self.device)
            val_loader = DataLoader(
                TensorDataset(x_val, y_val),
                batch_size=self.train_config.batch_size,
                shuffle=False,
            )

        if self.train_config.optimizer == "adamw":
            optimizer: optim.Optimizer = optim.AdamW(
                self.net.parameters(),
                lr=self.train_config.learning_rate,
                weight_decay=self.train_config.weight_decay,
            )
        else:
            optimizer = optim.Adam(
                self.net.parameters(),
                lr=self.train_config.learning_rate,
                weight_decay=self.train_config.weight_decay,
            )

        criterion = nn.MSELoss()

        train_metrics: dict = {}
        val_metrics: dict = {}
        avg_train_loss = 0.0
        avg_val_loss: float | None = None

        for epoch in range(self.train_config.num_epochs):
            train_result = self._train_epoch(train_loader, optimizer, criterion)
            avg_train_loss = train_result["loss"]
            train_metrics = {k: v for k, v in train_result.items() if k != "loss"}

            if val_loader is not None:
                val_result = self._validate_epoch(val_loader, criterion)
                avg_val_loss = val_result["loss"]
                val_metrics = {k: v for k, v in val_result.items() if k != "loss"}

            additional: dict[str, float] = {}
            if (v := train_metrics.get("spearman")) is not None:
                additional["train_spearman"] = float(v)
            if (v := train_metrics.get("mse")) is not None:
                additional["train_mse"] = float(v)
            if val_loader is not None:
                if (v := val_metrics.get("spearman")) is not None:
                    additional["val_spearman"] = float(v)
                if (v := val_metrics.get("mse")) is not None:
                    additional["val_mse"] = float(v)

            self._epoch_metrics.append(
                SurrogateEpochMetrics(
                    epoch=epoch,
                    train_loss=avg_train_loss,
                    val_loss=avg_val_loss,
                    additional_metrics=additional,
                )
            )

            if epoch % self.train_config.log_frequency == 0:
                log_msg = (
                    f"MLPModel '{self.name}' epoch {epoch}/{self.train_config.num_epochs - 1} "
                    f"train_loss={avg_train_loss:.4f}"
                )
                if avg_val_loss is not None:
                    log_msg += f" val_loss={avg_val_loss:.4f}"
                logger.debug(log_msg)

        self.training_metrics = {"final_train_loss": avg_train_loss}
        self.training_metrics.update({f"final_train_{k}": v for k, v in train_metrics.items()})
        if val_loader is not None:
            self.training_metrics["final_val_loss"] = avg_val_loss  # type: ignore[assignment]
            self.training_metrics.update({f"final_val_{k}": v for k, v in val_metrics.items()})

        self.net.eval()

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Return mean predictions, with MC dropout empirical distribution if configured.

        Args:
            candidate_points: Candidates to predict for.

        Returns:
            Predictions with means always set. If n_mc_passes > 0, variances and
            empirical_dist are also populated from stochastic forward passes.

        Raises:
            RuntimeError: If train() has not been called yet.
        """
        if self.net is None:
            raise RuntimeError("Model not trained. Call train() first.")

        x = self.featurise(candidate_points).to(self.device)

        if self.model_config.n_mc_passes == 0:
            self.net.eval()
            with torch.no_grad():
                preds = self.net(x).cpu().numpy()
            return Predictions(means=preds)

        # MC dropout: eval mode to freeze BatchNorm stats, then selectively
        # enable only Dropout modules so stochastic sampling remains active.
        self.net.eval()
        for m in self.net.modules():
            if isinstance(m, nn.Dropout):
                m.train()
        seed = (
            self.model_config.dropout_seed
            if self.model_config.dropout_seed is not None
            else self.model_config.model_seed
        )
        torch.manual_seed(seed)
        passes: list[np.ndarray] = []
        with torch.no_grad():
            for _ in range(self.model_config.n_mc_passes):
                passes.append(self.net(x).cpu().numpy())

        # Restore full eval state — dropout submodules were put in train() above.
        self.net.eval()

        empirical_dist = np.stack(passes, axis=1)  # (N, T)
        means = empirical_dist.mean(axis=1)
        variances = empirical_dist.var(axis=1)
        return Predictions(means=means, variances=variances, empirical_dist=empirical_dist)
