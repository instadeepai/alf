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

from alf_tools.models.botorch_exact_gp_model import BoTorchGPModel, BoTorchTrainConfig
from alf_tools.models.cnn import CNNModel, CNNModelConfig, CNNTrainConfig
from alf_tools.models.ensemble import EnsembleWrapper, EnsembleWrapperConfig, SubsampleConfig
from alf_tools.models.gp import FeaturizerConfig, GPModel, GPModelConfig, GPTrainConfig
from alf_tools.models.mlp import MLP, MLPModel, MLPModelConfig, MLPTrainConfig
from alf_tools.models.utils import (
    build_from_target,
    create_char_to_idx_mapping,
    extract_sequences_from_inputs,
    get_device,
    one_hot_encode,
)

__all__ = [
    "BoTorchGPModel",
    "BoTorchTrainConfig",
    "build_from_target",
    "CNNModel",
    "CNNModelConfig",
    "CNNTrainConfig",
    "create_char_to_idx_mapping",
    "extract_sequences_from_inputs",
    "FeaturizerConfig",
    "get_device",
    "GPModel",
    "GPModelConfig",
    "GPTrainConfig",
    "one_hot_encode",
    "EnsembleWrapper",
    "EnsembleWrapperConfig",
    "SubsampleConfig",
    "MLP",
    "MLPModel",
    "MLPModelConfig",
    "MLPTrainConfig",
]


try:
    from alf_tools.models.esmfold import ESMFoldModel, ESMFoldModelConfig

    _esmfold_available = True
except ModuleNotFoundError:
    _esmfold_available = False

if _esmfold_available:
    __all__ += ["ESMFoldModelConfig", "ESMFoldModel"]

_chemprop_available = False
try:
    import chemprop as _chemprop  # noqa: F401

    _chemprop_available = True
except ImportError:
    pass

if _chemprop_available:
    from alf_tools.models.chemprop import ChempropModel, ChempropModelConfig, ChempropTrainConfig

    __all__ += ["ChempropModel", "ChempropModelConfig", "ChempropTrainConfig"]
