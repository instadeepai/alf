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
from alf_tools.models.gp import (
    FeaturizerConfig,
    GPModelConfig,
    GPModelTrainer,
    GPTrainConfig,
)
from alf_tools.models.model_utils import (
    create_char_to_idx_mapping,
    extract_sequences_from_inputs,
    get_device,
    one_hot_encode,
)

# Lazy import for PyRosetta to avoid heavy downloads on module import
# PyRosetta will only be imported when explicitly accessed
__all__ = [
    "CNNModel",
    "CNNModelConfig",
    "CNNTrainConfig",
    "FeaturizerConfig",
    "GPModelConfig",
    "GPModelTrainer",
    "GPTrainConfig",
    "create_char_to_idx_mapping",
    "extract_sequences_from_inputs",
    "get_device",
    "one_hot_encode",
    "PyRosetta",
]


def __getattr__(name: str):
    """Lazy import for heavy dependencies.
    
    This defers the import of PyRosetta until it's actually accessed,
    avoiding the heavy download and initialization overhead if it's not needed.
    
    TODO: I think that there should be a script for downloading and installing rosetta that should be 
    triggered if PyRosetta is being imported.
    """
    if name == "PyRosetta":
        try:
            from alf_tools.models.pyrosetta import PyRosetta
            return PyRosetta
        except ImportError as e:
            # Provide a clear, actionable error message
            raise ImportError(
                "PyRosetta is not installed or could not be imported.\n"
                "To install PyRosetta, run:\n"
                "  python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'\n"
                "\n"
                "Note: PyRosetta is a large package (~1-2GB) and installation may take 20-30 minutes.\n"
                f"Original error: {e}"
            ) from e
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")