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

import numpy as np
import pytest
from alf_core.dataclasses import Candidate, LabelledCandidates, Predictions, State
from alf_core.model.base_model import BaseModel
from alf_core.oracle.oracle import Oracle


class SimpleModel(BaseModel):
    """Minimal model stub that returns fixed predictions."""

    def predict(self, candidates):
        """Return fixed predictions for the given candidates."""
        return Predictions(means=np.ones(len(candidates)))

    def featurise(self, inputs):
        """No-op featurisation."""

    def train(self, train_data, val_data):
        """No-op training."""

    def sample(self, condition=None):
        """Not implemented."""
        raise NotImplementedError


@pytest.fixture
def model_oracle():
    """Oracle backed by a simple model scorer.

    Returns:
        An Oracle wrapping a SimpleModel instance.
    """
    return Oracle(scorer=SimpleModel())


@pytest.fixture
def dataset_oracle(dummy_dataset):
    """Oracle backed by a dataset scorer.

    Returns:
        An Oracle wrapping the dummy dataset.
    """
    return Oracle(scorer=dummy_dataset)


@pytest.fixture
def model_candidates():
    """Five synthetic candidates for model-backed oracle tests.

    Returns:
        A list of five sequence Candidate objects.
    """
    return [Candidate(data=f"seq{i}", modality="sequence") for i in range(5)]


@pytest.fixture
def dataset_candidates(dummy_dataset):
    """Real candidates that exist in the dataset.

    Returns:
        First five candidates from the dataset candidate pool.
    """
    return dummy_dataset.candidate_pool.candidates[:5]


@pytest.fixture
def state(dummy_dataset, dummy_surrogate):
    """Minimal task state for evaluate() tests.

    Returns:
        A State instance initialised with the dummy dataset and surrogate.
    """
    return State(dataset=dummy_dataset, surrogate=dummy_surrogate)


class TestOracleEvaluate:
    """Tests for Oracle.evaluate()."""

    def test_evaluate_with_model_scorer_returns_labelled_candidates(
        self, model_oracle, model_candidates, state
    ):
        """Test that evaluate() returns LabelledCandidates and records oracle_time."""
        labelled, new_state = model_oracle.evaluate(model_candidates, state)
        assert isinstance(labelled, LabelledCandidates)
        assert len(labelled.candidates) == len(model_candidates)
        assert "oracle_time" in new_state.round_metrics.metrics

    def test_evaluate_with_dataset_scorer_returns_labelled_candidates(
        self, dataset_oracle, dataset_candidates, state
    ):
        """Test that evaluate() works correctly when backed by a dataset scorer."""
        labelled, new_state = dataset_oracle.evaluate(dataset_candidates, state)
        assert isinstance(labelled, LabelledCandidates)
        assert len(labelled.candidates) == len(dataset_candidates)
        assert "oracle_time" in new_state.round_metrics.metrics

    def test_evaluate_model_scorer_labels_are_ndarray(self, model_oracle, model_candidates, state):
        """Test that labels returned by evaluate() are a numpy array."""
        labelled, _ = model_oracle.evaluate(model_candidates, state)
        assert isinstance(labelled.labels, np.ndarray)
