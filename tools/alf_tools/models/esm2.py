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
from dataclasses import dataclass
from typing import Any, Literal, Union

import numpy as np
import torch
from alf_core import BaseModel, Candidate, LabelledCandidates, Predictions
from alf_core.dataclasses.surrogate_epoch_metrics import SurrogateEpochMetrics
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

from alf_tools.models.utils import get_device

logger = logging.getLogger("alf-tools")


@dataclass
class ESM2ModelConfig:
    """Configuration for ESM-2 model architecture.

    Args:
        model_id: HuggingFace model identifier, e.g. 'facebook/esm2_t6_8M_UR50D'.
        pooling: Strategy for reducing per-token hidden states to a sequence embedding.
        repr_layer: Transformer layer index to extract embeddings from. -1 = final layer.
    """

    model_id: str
    pooling: Literal["mean", "cls", "last_hidden_state"] = "mean"
    repr_layer: int = -1


@dataclass
class ESM2TrainConfig:
    """Configuration for ESM-2 training.

    Args:
        freeze_backbone: When True, train() is a no-op (pure embedding extractor).
        learning_rate: Learning rate for the optimizer.
        optimizer_type: Which optimizer to use ('adam' or 'adamw').
        batch_size: Batch size for training.
        num_epochs: Number of epochs to train for.
        mask_probability: Fraction of non-special tokens to randomly mask (MLM).
        log_frequency: Record epoch metrics every N epochs.
    """

    freeze_backbone: bool = True
    learning_rate: float = 1e-4
    optimizer_type: Literal["adam", "adamw"] = "adamw"
    batch_size: int = 8
    num_epochs: int = 10
    mask_probability: float = 0.15
    log_frequency: int = 1


class ESM2Model(BaseModel):
    """ESM-2 protein language model wrapper.

    Loads a pre-trained ESM-2 checkpoint from HuggingFace and exposes it as a
    BaseModel. predict() returns per-sequence embeddings. Optionally fine-tunes
    the backbone with masked language modelling (MLM).
    """

    def __init__(
        self,
        name: str,
        model_config: ESM2ModelConfig,
        train_config: ESM2TrainConfig | None = None,
        device: str | None = None,
    ):
        raise NotImplementedError

    def featurise(self, inputs: Union[LabelledCandidates, list[Candidate]]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        raise NotImplementedError

    def train(self, train_data: LabelledCandidates, val_data: LabelledCandidates | None = None) -> None:
        raise NotImplementedError

    def sample(self, *args: Any, **kwargs: Any) -> list[Candidate]:
        raise NotImplementedError("Sampling is not implemented for this model.")

    def get_epoch_metrics(self) -> list[SurrogateEpochMetrics]:
        raise NotImplementedError

    def get_training_summary_metrics(self) -> dict[str, Union[float, int, np.number]]:
        raise NotImplementedError
