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

"""Unit tests for Surrogate.featurise() delegation."""

from typing import Any

import numpy as np
import pytest
from alf_core import BaseModel, Candidate, LabelledCandidates, Predictions, Surrogate


class _FeaturiseModel(BaseModel):
    """Minimal model that returns a fixed 2-D embedding from featurise()."""

    def featurise(self, inputs: list[Candidate] | LabelledCandidates) -> np.ndarray:
        """Return a (n, 2) array of ones for the given inputs.

        Args:
            inputs: Candidates or labelled candidates to featurise.

        Returns:
            Array of shape (n, 2) filled with ones.
        """
        candidates = inputs if isinstance(inputs, list) else inputs.candidates
        return np.ones((len(candidates), 2))

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Return zero-mean predictions.

        Args:
            candidate_points: Candidates to predict for.

        Returns:
            Predictions with zero means.
        """
        return Predictions(means=np.zeros(len(candidate_points)))

    def train(self, train_data: LabelledCandidates, val_data: Any = None) -> None:
        """No-op training.

        Args:
            train_data: Training data (unused).
            val_data: Validation data (unused).
        """

    def sample(self, condition: Any = None) -> list[Candidate]:
        """Not implemented for this test model.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError


class _RaisingFeaturiseModel(_FeaturiseModel):
    """Model whose featurise() always raises RuntimeError."""

    def featurise(self, inputs: list[Candidate] | LabelledCandidates) -> np.ndarray:
        """Raise RuntimeError unconditionally.

        Args:
            inputs: Ignored.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("featurise failed intentionally")


def _make_candidates(n: int = 2) -> list[Candidate]:
    """Create n simple sequence candidates.

    Args:
        n: Number of candidates.

    Returns:
        List of n Candidate objects.
    """
    return [Candidate(data=f"seq_{i}", modality="sequence") for i in range(n)]


def test_surrogate_featurise_delegates_to_model() -> None:
    """Surrogate.featurise() returns whatever the underlying model returns."""
    surrogate = Surrogate(model=_FeaturiseModel())
    candidates = _make_candidates(3)
    result = surrogate.featurise(candidates)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 2)
    assert np.all(result == 1.0)


def test_surrogate_featurise_accepts_labelled_candidates() -> None:
    """Surrogate.featurise() works when passed a LabelledCandidates object."""
    surrogate = Surrogate(model=_FeaturiseModel())
    candidates = _make_candidates(2)
    labelled = LabelledCandidates(candidates=candidates, labels=np.array([1.0, 2.0]))
    result = surrogate.featurise(labelled)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)


def test_surrogate_featurise_propagates_model_exception() -> None:
    """Surrogate.featurise() does not swallow exceptions from model.featurise()."""
    surrogate = Surrogate(model=_RaisingFeaturiseModel())
    with pytest.raises(RuntimeError, match="featurise failed intentionally"):
        surrogate.featurise(_make_candidates(1))
