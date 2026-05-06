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
    """Placeholder for MLP torch module (implemented in future tasks)."""

    pass


class MLPModel(BaseModel):
    """Placeholder for MLPModel wrapper (implemented in future tasks)."""

    pass
