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

from typing import Any

import numpy as np
import pytest
from alf_core import Candidate, LabelledCandidates, Predictions, State
from alf_core.model.base_model import BaseModel
from alf_core.surrogate.surrogate import Surrogate

from alf_tools.optimizer.acquisition_functions.core_set import CoreSet


class EmbeddingModel(BaseModel):
    """Test model that returns a pre-set numpy embedding array."""

    def __init__(self, embeddings: np.ndarray) -> None:
        """Initialise with a fixed embedding array.

        Args:
            embeddings: Array of shape (n_total, d) returned slice-by-count from featurise.
        """
        self._embeddings = embeddings

    def featurise(self, inputs: LabelledCandidates | list[Candidate]) -> np.ndarray:
        """Return the first n rows of the pre-set embeddings.

        Args:
            inputs: Candidates to featurise.

        Returns:
            Embedding array of shape (n, d).
        """
        n = len(inputs) if isinstance(inputs, list) else len(inputs.candidates)
        return self._embeddings[:n]

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Return zero-mean predictions.

        Args:
            candidate_points: Candidates to predict for.

        Returns:
            Predictions with zero means.
        """
        return Predictions(means=np.zeros(len(candidate_points)))

    def train(self, train_data: LabelledCandidates, val_data: LabelledCandidates) -> None:
        """No-op training.

        Args:
            train_data: Training data (unused).
            val_data: Validation data (unused).
        """
        pass

    def sample(self, condition: Any | None = None) -> list[Candidate]:
        """Not implemented for this test model.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError


class NullFeaturiseModel(BaseModel):
    """Test model whose featurise returns None."""

    def featurise(self, inputs: LabelledCandidates | list[Candidate]) -> None:
        """Return None to trigger the ValueError path.

        Args:
            inputs: Candidates (unused).

        Returns:
            None.
        """
        return None

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Return zero-mean predictions.

        Args:
            candidate_points: Candidates to predict for.

        Returns:
            Predictions with zero means.
        """
        return Predictions(means=np.zeros(len(candidate_points)))

    def train(self, train_data: LabelledCandidates, val_data: LabelledCandidates) -> None:
        """No-op training.

        Args:
            train_data: Training data (unused).
            val_data: Validation data (unused).
        """
        pass

    def sample(self, condition: Any | None = None) -> list[Candidate]:
        """Not implemented for this test model.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError


class MockDataset:
    """Minimal dataset-like object exposing only train_dataset."""

    def __init__(self, train_candidates: list[Candidate], train_labels: np.ndarray) -> None:
        """Initialise with training candidates and labels.

        Args:
            train_candidates: List of training candidates.
            train_labels: Labels for training candidates.
        """
        self.train_dataset = LabelledCandidates(
            candidates=train_candidates,
            labels=train_labels,
        )


def _make_state(
    embeddings: np.ndarray,
    n_train: int,
    acq_batch_size: int,
) -> tuple[State, list[Candidate]]:
    """Build a State and search_candidates list for testing.

    The embeddings array has shape (n_train + n_cands, d). The first n_train rows
    are assigned to the training set; the remaining rows to the search candidates.
    The EmbeddingModel returns ``embeddings[:n]`` for any call to featurise with n inputs,
    so CoreSet receives the full array when it calls featurise(training + search).

    Args:
        embeddings: Full embedding array, training rows first.
        n_train: Number of training points.
        acq_batch_size: Acquisition batch size for the State.

    Returns:
        Tuple of (State, search_candidates).
    """
    train_candidates = [Candidate(data=f"train_{i}", modality="sequence") for i in range(n_train)]
    n_cands = len(embeddings) - n_train
    search_candidates = [Candidate(data=f"cand_{i}", modality="sequence") for i in range(n_cands)]

    dataset = MockDataset(train_candidates=train_candidates, train_labels=np.zeros(n_train))
    surrogate = Surrogate(model=EmbeddingModel(embeddings))
    state = State(dataset=dataset, surrogate=surrogate, acq_batch_size=acq_batch_size)
    return state, search_candidates


class TestCoreSet:
    """Unit tests for the CoreSet acquisition function."""

    def test_farthest_candidate_gets_highest_score(self) -> None:
        """The candidate geometrically farthest from the training set gets the highest score.

        Setup (2-D):
            Training:   (0, 0)
            Candidates: (1, 0), (0, 3), (2, 2)
            Min dists:   1.0,    3.0,    sqrt(8) ≈ 2.83

        With acq_batch_size=1 only candidate index 1 (dist 3.0) is selected.
        """
        embeddings = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 3.0], [2.0, 2.0]])
        state, search_candidates = _make_state(embeddings, n_train=1, acq_batch_size=1)

        result = CoreSet()(search_candidates, state)

        assert result.labels[1] == pytest.approx(3.0)
        assert result.labels[0] == 0.0
        assert result.labels[2] == 0.0

    def test_scores_decrease_by_selection_order(self) -> None:
        """Scores are non-increasing: each subsequent greedy pick scores <= its predecessor.

        Setup (2-D):
            Training:   (0, 0)
            Candidates: (1, 0), (0, 3), (2, 2)
            acq_batch_size = 2

        Step 1: select (0, 3) with score 3.0.
        After update, min_dists become [1.0, 0.0, sqrt(5)] (new centre at (0,3)).
        Step 2: select (2, 2) with score sqrt(5) ≈ 2.24.
        Candidate (1, 0) is not selected (score 0).
        """
        embeddings = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 3.0], [2.0, 2.0]])
        state, search_candidates = _make_state(embeddings, n_train=1, acq_batch_size=2)

        result = CoreSet()(search_candidates, state)

        assert result.labels[1] == pytest.approx(3.0)
        assert result.labels[2] == pytest.approx(np.sqrt(5))
        assert result.labels[0] == 0.0
        assert result.labels[1] >= result.labels[2]

    def test_unselected_candidates_score_zero(self) -> None:
        """Candidates not reached within acq_batch_size steps receive a score of 0."""
        embeddings = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 3.0], [2.0, 2.0]])
        state, search_candidates = _make_state(embeddings, n_train=1, acq_batch_size=1)

        result = CoreSet()(search_candidates, state)

        assert int((result.labels > 0).sum()) == 1

    def test_batch_size_exceeds_candidates(self) -> None:
        """When acq_batch_size >= n_cands, all candidates receive a positive score."""
        embeddings = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 3.0], [2.0, 2.0]])
        state, search_candidates = _make_state(embeddings, n_train=1, acq_batch_size=10)

        result = CoreSet()(search_candidates, state)

        assert (result.labels > 0).all()

    def test_raises_on_none_features(self) -> None:
        """ValueError is raised when the model's featurise returns None."""
        train_candidates = [Candidate(data="train_0", modality="sequence")]
        search_candidates = [Candidate(data="cand_0", modality="sequence")]
        dataset = MockDataset(train_candidates=train_candidates, train_labels=np.zeros(1))
        state = State(
            dataset=dataset,
            surrogate=Surrogate(model=NullFeaturiseModel()),
            acq_batch_size=1,
        )

        with pytest.raises(ValueError, match="featurise returned None"):
            CoreSet()(search_candidates, state)
