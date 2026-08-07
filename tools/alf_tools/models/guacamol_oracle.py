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

from alf_tools.datasets.guacamol.guacamol_scoring import get_task_scorer
from alf_tools.datasets.guacamol.guacamol_utils import GuacaMolTaskName
from alf_tools.models.molecule_oracle import MoleculeOracleModel


@dataclass
class GuacaMolOracleModelConfig:
    """Configuration for GuacaMolOracleModel.

    Args:
        task_name: Name of the GuacaMol composite/MPO benchmark task to score
            candidates against (e.g. "osimertinib_mpo"). See GuacaMolTaskName
            for all supported values.
    """

    task_name: GuacaMolTaskName


class GuacaMolOracleModel(MoleculeOracleModel):
    """Online oracle that scores arbitrary SMILES via a GuacaMol composite/MPO task.

    A thin preset over `MoleculeOracleModel`: the scorer is fixed to one of GuacaMol's
    named benchmark tasks instead of an arbitrary user-supplied function. Unlike
    GuacaMol(BaseDataset), which scores a fixed corpus once and stores the results as
    static labels, this model calls the RDKit-based scorer fresh on every predict()
    call, so it can evaluate any SMILES a search protocol proposes — including
    molecules that never appeared in any corpus.
    """

    def __init__(self, config: GuacaMolOracleModelConfig) -> None:
        """Initialize the oracle model with the target GuacaMol task.

        Args:
            config: Configuration selecting which GuacaMol task to score against.
        """
        self.config = config
        super().__init__(scorer=get_task_scorer(config.task_name))
