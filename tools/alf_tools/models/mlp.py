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
from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from alf_core import BaseModel, Candidate, LabelledCandidates, Predictions, Results
from alf_core.dataclasses.surrogate_epoch_metrics import SurrogateEpochMetrics
from torch.utils.data import DataLoader, TensorDataset

from alf_tools.models.utils import get_device

logger = logging.getLogger("alf-tools")


@dataclass
class MLPModelConfig:
    """Configuration for MLP model architecture.

    Args:
        hidden_dims: List of hidden layer sizes.
        dropout: Dropout rate applied after each hidden layer.
    """

    hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    dropout: float = 0.2


@dataclass
class MLPTrainConfig:
    """Configuration for MLP training.

    Args:
        learning_rate: Learning rate for AdamW optimizer.
        batch_size: Batch size for training.
        num_epochs: Number of training epochs.
        log_frequency: Frequency of logging epoch metrics.
    """

    learning_rate: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 50
    log_frequency: int = 10


class MolecularMLP(nn.Module):
    """Feed-forward MLP for molecular property regression.

    Architecture: Linear -> GELU -> LayerNorm -> Dropout, repeated per hidden layer,
    then a final Linear with no activation (raw scalar output).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.GELU(),
                nn.LayerNorm(h_dim),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning scalar predictions of shape (batch_size,)."""
        return self.network(x).squeeze(-1)


class MLPModel(BaseModel):
    """MLP surrogate for molecular property prediction."""

    def __init__(
        self,
        name: str = "mlp_model",
        model_config: MLPModelConfig | None = None,
        train_config: MLPTrainConfig | None = None,
        device: str | None = None,
    ):
        self.model_config = model_config or MLPModelConfig()
        self.train_config = train_config or MLPTrainConfig()
        self.device = get_device(device)
        self.model: MolecularMLP | None = None
        self.training_metrics: dict[str, Union[float, int, np.number]] = {}
        self._epoch_metrics: list[SurrogateEpochMetrics] = []

    def featurise(self, inputs: Union[LabelledCandidates, list[Candidate]]) -> torch.Tensor:
        """Convert candidates to feature tensor (implemented in Task 3)."""
        raise NotImplementedError

    def train(self, train_data: LabelledCandidates, val_data: LabelledCandidates | None = None) -> None:
        """Train the MLP model (implemented in Task 4)."""
        raise NotImplementedError

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Make predictions (implemented in Task 4)."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        raise NotImplementedError

    def sample(self, *args: Any, **kwargs: Any) -> list[Candidate]:
        """Sample candidate points from the model."""
        raise NotImplementedError("Sampling is not implemented for this model.")

    def get_epoch_metrics(self) -> list[SurrogateEpochMetrics]:
        """Return per-epoch metrics from the most recent train() call."""
        return self._epoch_metrics

    def get_training_summary_metrics(self) -> dict[str, Union[float, int, np.number]]:
        """Return training summary metrics."""
        return self.training_metrics
