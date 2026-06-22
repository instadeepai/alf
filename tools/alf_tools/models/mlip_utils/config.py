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

from dataclasses import dataclass
from typing import Optional

from alf_core import BaseTrainConfig


@dataclass
class MLIPModelConfig:
    """Configuration for the MLIP model architecture.

    Args:
        model_path: Path to a pretrained model zip file (relative to the models/ directory),
            or None to train from scratch. Defaults to the MLIP-1 foundation model.
        num_channels: Number of channels in the MACE architecture. Only used when
            training from scratch (model_path=None).
        correlation: Correlation order for the MACE architecture. Only used when
            training from scratch (model_path=None).
        graph_cutoff_angstrom: Graph cutoff distance in Angstrom. Only used when
            training from scratch (model_path=None); when finetuning, the pretrained
            model's cutoff is used.
    """

    model_path: Optional[str] = "fine_tuning/mlip-1416.zip"
    num_channels: int = 128
    correlation: int = 3
    graph_cutoff_angstrom: float = 5.0


@dataclass
class MLIPTrainConfig(BaseTrainConfig):
    """Configuration for MLIP model training.

    When dynamic_training=True, batch_size, learning_rate, and epochs are
    automatically set based on training size to keep the total number of
    gradient updates constant (~1000 updates):

    - 1-20 samples:  batch_size=1, learning_rate=0.001
    - 21-100 samples: batch_size=2, learning_rate=0.005
    - 100+ samples:  batch_size=4, learning_rate=0.01

    Epochs = max(10, ceil(1000 * batch_size / train_size))

    Args:
        epochs: Number of training epochs. Ignored when dynamic_training=True.
        batch_size: Batch size for training. Ignored when dynamic_training=True.
        learning_rate: Learning rate for the optimizer. Ignored when dynamic_training=True.
        dynamic_training: Automatically tune batch_size, learning_rate, and epochs
            based on training set size to maintain ~1000 gradient updates.
        use_weight_flip: Use a piecewise schedule that starts energy-weighted and
            switches to forces-weighted at flip_epoch.
        flip_epoch: Epoch at which to flip the energy/forces loss weights.
            Defaults to 70% of total epochs when None.
        energy_weight: Energy loss weight. Only used when use_weight_flip=False.
        forces_weight: Forces loss weight. Only used when use_weight_flip=False.
    """

    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 1e-3
    dynamic_training: bool = False
    use_weight_flip: bool = True
    flip_epoch: Optional[int] = None
    energy_weight: float = 1.0
    forces_weight: float = 1.0
