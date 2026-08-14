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

    For benchmark tasks, GuacaMol(BaseDataset).query() computes scores the same way —
    there's no corpus lookup for that mode either, every SMILES is re-scored by the task
    function. What this class avoids is everything else that comes with
    GuacaMol(BaseDataset): constructing one downloads and RDKit-scores the entire
    ~1.6M-molecule corpus up front, even if that corpus is never otherwise used, and it
    makes Oracle classify the scorer as offline by type. This is a plain BaseModel — no
    corpus, near-instant to construct, and it degrades gracefully on invalid SMILES
    (returns 0.0) rather than raising, unlike GuacaMol.query()'s benchmark-task path.
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
            inputs: Candidates to featurise (unused).
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
