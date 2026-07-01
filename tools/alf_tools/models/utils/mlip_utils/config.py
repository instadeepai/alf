# Copyright 2026 InstaDeep Ltd. All rights reserved.
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

"""Configuration dataclasses for the ALF MLIP wrapper."""

from dataclasses import dataclass, field
from pathlib import Path

from alf_core import BaseTrainConfig
from mlip.models.config import MLIPNetworkConfig
from mlip.models.loss import Loss, MSELoss
from mlip.training import TrainingLoop
from mlip.training.optimizer_config import OptimizerConfig

from alf_tools.models.utils.mlip_utils.model_registry import MLIPModelType


@dataclass
class MLIPModelConfig:
    """Configuration for the MLIP model architecture.

    Args:
        model_path: Path to a pretrained model zip file, or None to train from scratch.
        model_type: mlip architecture class to use when loading or training a model.
            Supported values are "mace", "nequip", "visnet", and "esen".
        network_config: Architecture-specific mlip network config used when
            training from scratch. Defaults to the selected model's default Config.
        graph_cutoff_angstrom: Graph cutoff distance in Angstrom. Only used when
            training from scratch (model_path=None); when finetuning, the pretrained
            model's cutoff is used.
    """

    model_path: str | Path | None = None
    model_type: MLIPModelType = "mace"
    network_config: MLIPNetworkConfig | None = None
    graph_cutoff_angstrom: float = 5.0


@dataclass(kw_only=True)
class MLIPTrainConfig(BaseTrainConfig):
    """Configuration for MLIP model training.

    Args:
        batch_size: Batch size for training.
        inference_batch_size: Batch size for prediction. Defaults to batch_size when None.
        optimizer_config: Required native mlip optimizer config.
        training_loop_config: Required native mlip training loop config.
        loss: Optional native mlip loss. Defaults to ``MSELoss``.
    """

    optimizer_config: OptimizerConfig
    training_loop_config: TrainingLoop.Config
    batch_size: int = 8
    inference_batch_size: int | None = None
    loss: Loss = field(default_factory=MSELoss)
