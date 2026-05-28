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

from alf_tools.models.cnn import CNNModel, CNNModelConfig, CNNTrainConfig
from alf_tools.models.esm2 import ESM2Model, ESM2ModelConfig, ESM2TrainConfig
from alf_tools.models.gp import FeaturizerConfig, GPModel, GPModelConfig, GPTrainConfig
from alf_tools.models.mlp import MLP, MLPModel, MLPModelConfig, MLPTrainConfig
from alf_tools.models.utils import (
    create_char_to_idx_mapping,
    extract_sequences_from_inputs,
    get_device,
    one_hot_encode,
)

__all__ = [
    "CNNModel",
    "CNNModelConfig",
    "CNNTrainConfig",
    "ESM2Model",
    "ESM2ModelConfig",
    "ESM2TrainConfig",
    "create_char_to_idx_mapping",
    "extract_sequences_from_inputs",
    "get_device",
    "one_hot_encode",
    "FeaturizerConfig",
    "GPModelConfig",
    "GPModel",
    "GPTrainConfig",
    "MLP",
    "MLPModel",
    "MLPModelConfig",
    "MLPTrainConfig",
]
