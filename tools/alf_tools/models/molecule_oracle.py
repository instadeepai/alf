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

from typing import Callable

import numpy as np
from alf_core import BaseModel, Candidate, LabelledCandidates, Predictions


class MoleculeOracleModel(BaseModel):
    """Generic online oracle for molecules: scores arbitrary SMILES with any
    user-supplied `(smiles: str) -> float` function, computed fresh on every
    predict() call rather than looked up from a fixed, precomputed corpus.

    The scorer can be anything that maps a SMILES string to a float: a GuacaMol
    benchmark scorer (see `GuacaMolOracleModel`, a thin preset built on top of
    this class), a raw RDKit descriptor (e.g. `Descriptors.MolLogP`), a
    hand-written composite function, or an ML-based property predictor. This
    class has no RDKit or GuacaMol dependency itself — whatever the scorer
    needs is on the caller.
    """

    def __init__(self, scorer: Callable[[str], float]) -> None:
        """Initialize the oracle model with a SMILES-scoring function.

        Args:
            scorer: Callable mapping a SMILES string to a scalar score.
        """
        self._scorer = scorer

    def featurise(self, inputs: list[Candidate]) -> None:
        """No-op — the scorer computes whatever it needs internally per SMILES.

        Args:
            inputs: Candidates to featurize (unused).
        """

    def train(
        self,
        train_data: LabelledCandidates,
        val_data: LabelledCandidates | None = None,
    ) -> None:
        """No-op — the oracle is a fixed function, not fit to data.

        Args:
            train_data: Training data (unused).
            val_data: Validation data (unused).
        """

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Score each candidate's SMILES with the configured scorer.

        Args:
            candidate_points: Candidates whose `.data` is a SMILES string.

        Returns:
            Predictions with one score per candidate.
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
        raise NotImplementedError(
            "MoleculeOracleModel scores candidates; it does not generate them."
        )
