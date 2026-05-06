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
from typing import Any, Literal, Union

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
        num_epochs: Number of training epochs.
        optimizer: Optimiser type; "adam" or "adamw".
        weight_decay: L2 regularisation coefficient.
    """

    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 50
    optimizer: Literal["adam", "adamw"] = "adam"
    weight_decay: float = 0.0


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
        self.name = name
        self.model_config = model_config or MLPModelConfig()
        self.train_config = train_config or MLPTrainConfig()
        self.device = get_device(device)
        self.net: MLP | None = None
        self.training_metrics: dict[str, Union[float, int, np.number]] = {}
        self._epoch_metrics: list[SurrogateEpochMetrics] = []

    def featurise(self, inputs: Union[LabelledCandidates, list[Candidate]]) -> torch.Tensor:
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
        raise NotImplementedError("Sampling is not implemented for MLPModel.")

    def get_epoch_metrics(self) -> list[SurrogateEpochMetrics]:
        return self._epoch_metrics

    def get_training_summary_metrics(self) -> dict[str, Union[float, int, np.number]]:
        return self.training_metrics

    def train(
        self,
        train_data: LabelledCandidates,
        val_data: LabelledCandidates | None = None,
    ) -> None:
        raise NotImplementedError("train() not yet implemented")

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        raise NotImplementedError("predict() not yet implemented")
