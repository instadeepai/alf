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

import numpy as np
from alf_core import BaseModel, Candidate, LabelledCandidates, Predictions

from alf_tools.datasets.guacamol.guacamol_scoring import get_task_scorer
from alf_tools.datasets.guacamol.guacamol_utils import GuacaMolTaskName


@dataclass
class GuacaMolOracleConfig:
    """Configuration for GuacaMolOracle.

    Args:
        task_name: Name of the GuacaMol composite/MPO benchmark task to score
            candidates against (e.g. "osimertinib_mpo"). See GuacaMolTaskName
            for all supported values.
    """

    task_name: GuacaMolTaskName


class GuacaMolOracle(BaseModel):
    """Online oracle that scores arbitrary SMILES via a GuacaMol composite/MPO task.

    Unlike GuacaMol(BaseDataset), which scores a fixed corpus once and stores the
    results as static labels, this model calls the RDKit-based scorer fresh on
    every predict() call, so it can evaluate any SMILES a search protocol
    proposes — including molecules that never appeared in any corpus.
    """

    def __init__(self, config: GuacaMolOracleConfig) -> None:
        """Initialize the oracle model with the target GuacaMol task.

        Args:
            config: Configuration selecting which GuacaMol task to score against.
        """
        self.config = config
        self._scorer = get_task_scorer(config.task_name)

    def featurise(self, inputs: list[Candidate]) -> None:
        """No-op — the scorer computes RDKit descriptors internally per SMILES.

        Args:
            inputs: Candidates to featurize (unused).
        """

    def train(
        self,
        train_data: LabelledCandidates,
        val_data: LabelledCandidates | None = None,
    ) -> None:
        """No-op — the oracle is a fixed RDKit-computed function, not fit to data.

        Args:
            train_data: Training data (unused).
            val_data: Validation data (unused).
        """

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Score each candidate's SMILES with the configured GuacaMol task.

        Args:
            candidate_points: Candidates whose `.data` is a SMILES string.

        Returns:
            Predictions with one score in [0, 1] per candidate.
        """
        means = np.array([self._scorer(c.data) for c in candidate_points], dtype=float)
        return Predictions(means=means)

    def sample(self, condition: object | None = None) -> list[Candidate]:
        """Not supported — this oracle scores candidates, it does not generate them.

        Args:
            condition: Unused.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("GuacaMolOracle scores candidates; it does not generate them.")
