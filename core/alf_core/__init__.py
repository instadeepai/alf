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

# This file makes alf_core a Python package
from alf_core.dataclasses import (
    Candidate,
    LabelledCandidates,
    Modality,
    Predictions,
    Results,
    State,
)
from alf_core.dataset.base_dataset import BaseDataset, BaseDatasetConfig
from alf_core.model.base_model import BaseModel, BaseTrainConfig
from alf_core.model.normaliser import InputNormaliser, OutputStandardiser
from alf_core.optimizer.acquisition_function import AcquisitionFunction
from alf_core.optimizer.optimizer import Optimizer
from alf_core.optimizer.search import (
    BaseSearch,
    DatasetSearch,
    GeneratorSearch,
    ModelProtocolSearch,
    ProtocolSearch,
    SearchProtocol,
)
from alf_core.oracle.oracle import Oracle
from alf_core.surrogate.surrogate import Surrogate
from alf_core.tasks.base_task import BaseTask
from alf_core.tasks.design_task import DesignTask
from alf_core.tasks.supervised_task import SupervisedTask
from alf_core.tasks.zeroshot_task import ZeroShotTask
from alf_core.utils.state_logger import (
    FileStateLogger,
    StateLogger,
    TerminalStateLogger,
)
