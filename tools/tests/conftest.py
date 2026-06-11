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

"""Shared pytest fixtures for tools tests.

This module provides fixtures needed by the data-conversion utility tests
(test_botorch_utils.py) and GP model tests (test_gp.py,
test_botorch_model_adapter.py). Acquisition and dataset fixtures are added
in later PRs.
"""

import math
from dataclasses import dataclass
from typing import Union

import numpy as np
import pytest
import torch
from alf_core import (
    Candidate,
    LabelledCandidates,
    Modality,
    Predictions,
    Surrogate,
)
from alf_core.model.base_model import BaseModel
from alf_tools.models.gp import FeaturizerConfig, GPModel, GPTrainConfig
from botorch.models import SingleTaskGP


@pytest.fixture
def test_candidates_2d():
    """Create 2D test candidates (tabular data).

    Returns:
        List of 3 Candidate objects with 2D numpy arrays.

    Example:
        >>> def test_with_candidates(test_candidates_2d):
        ...     assert len(test_candidates_2d) == 3
        ...     assert test_candidates_2d[0].data.shape == (2,)
    """
    return [
        Candidate(data=np.array([0.5, 0.5], dtype=np.float32), modality=Modality.TABULAR),
        Candidate(data=np.array([0.3, 0.7], dtype=np.float32), modality=Modality.TABULAR),
        Candidate(data=np.array([0.8, 0.2], dtype=np.float32), modality=Modality.TABULAR),
    ]


@pytest.fixture
def test_tensor_2d():
    """Create 2D test tensor for BoTorch operations.

    Returns:
        Torch tensor of shape (3, 2) with float32 dtype.

    Example:
        >>> def test_with_tensor(test_tensor_2d):
        ...     posterior = model.posterior(test_tensor_2d)
        ...     assert posterior.mean.shape[0] == 3
    """
    return torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.8, 0.2]], dtype=torch.float32)


@pytest.fixture
def test_bounds_2d():
    """Create 2D bounds for optimization problems.

    Returns:
        List of tuples representing bounds: [(0.0, 1.0), (0.0, 1.0)].

    Example:
        >>> def test_optimization(test_bounds_2d):
        ...     bounds_tensor = get_bounds_tensor(test_bounds_2d)
        ...     assert bounds_tensor.shape == torch.Size([2, 2])
    """
    return [(0.0, 1.0), (0.0, 1.0)]


@pytest.fixture
def test_tensor_3d():
    """Create 3D test tensor (batch format) for BoTorch operations.

    Returns:
        Torch tensor of shape (2, 3, 2) representing 2 batches of 3 points each.

    Example:
        >>> def test_batched_posterior(test_tensor_3d):
        ...     posterior = model.posterior(test_tensor_3d)
    """
    return torch.tensor(
        [
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            [[0.7, 0.8], [0.9, 0.1], [0.2, 0.3]],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def random_seed():
    """Set a fixed random seed for reproducibility.

    Yields:
        The seed value (42), and resets numpy/torch random state after test.

    Example:
        >>> def test_reproducible(random_seed):
        ...     data = np.random.randn(100)
    """
    seed = 42
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    yield seed
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)


# =============================================================================
# Mock ALF Models
# =============================================================================


class MockALFModelWithVariances(BaseModel):
    """Mock ALF BaseModel that provides variances for testing.

    Attributes:
        mean_offset: Offset to add to predictions (default: 0.0).
        variance_value: Fixed variance value for all predictions (default: 0.1).
    """

    def __init__(self, mean_offset: float = 0.0, variance_value: float = 0.1):
        """Initialize mock model.

        Args:
            mean_offset: Offset to add to predictions.
            variance_value: Fixed variance value for predictions.
        """
        self.mean_offset = mean_offset
        self.variance_value = variance_value

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Generate predictions with variances.

        Args:
            candidate_points: List of candidates to predict.

        Returns:
            Predictions with means and variances.
        """
        n = len(candidate_points)
        means = np.arange(n, dtype=np.float32) + self.mean_offset
        variances = np.full(n, self.variance_value, dtype=np.float32)
        return Predictions(means=means, variances=variances)

    def featurise(self, inputs: Union[LabelledCandidates, list[Candidate]]) -> None:
        """Not implemented for mock model."""
        pass

    def train(
        self, train_data: LabelledCandidates, val_data: LabelledCandidates | None = None
    ) -> None:
        """Not implemented for mock model."""
        pass

    def sample(self, *args: object, **kwargs: object) -> list[Candidate]:
        """Not implemented for mock model."""
        raise NotImplementedError()


class MockALFModelWithoutVariances(BaseModel):
    """Mock ALF BaseModel that does NOT provide variances (deterministic).

    Predictions return sequential means (0, 1, 2, ...) with no variances.
    """

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Generate predictions WITHOUT variances.

        Args:
            candidate_points: List of candidates to predict.

        Returns:
            Predictions with only means (no variances).
        """
        n = len(candidate_points)
        means = np.arange(n, dtype=np.float32)
        return Predictions(means=means, variances=None)

    def featurise(self, inputs: Union[LabelledCandidates, list[Candidate]]) -> None:
        """Not implemented for mock model."""
        pass

    def train(
        self, train_data: LabelledCandidates, val_data: LabelledCandidates | None = None
    ) -> None:
        """Not implemented for mock model."""
        pass

    def sample(self, *args: object, **kwargs: object) -> list[Candidate]:
        """Not implemented for mock model."""
        raise NotImplementedError()


@pytest.fixture
def mock_alf_model_with_variances():
    """Create an ALF model that provides variances.

    Returns:
        MockALFModelWithVariances with offset=5.0 and variance=0.2.
    """
    return MockALFModelWithVariances(mean_offset=5.0, variance_value=0.2)


@pytest.fixture
def mock_alf_model_without_variances():
    """Create an ALF model that does NOT provide variances.

    Returns:
        MockALFModelWithoutVariances instance.
    """
    return MockALFModelWithoutVariances()


# =============================================================================
# BoTorch Model Fixtures
# =============================================================================


@pytest.fixture
def simple_train_data():
    """Create simple training data for BoTorch models.

    Returns:
        Tuple of (train_X, train_Y) tensors where:
        - train_X: (10, 2) tensor of random inputs in [0, 1]
        - train_Y: (10, 1) tensor of sin/cos function outputs
    """
    torch.manual_seed(42)
    train_X = torch.rand(10, 2, dtype=torch.float32)
    train_Y = torch.sin(train_X[:, 0] * 3.14159) + torch.cos(train_X[:, 1] * 3.14159)
    train_Y = train_Y.unsqueeze(-1)
    return train_X, train_Y


@pytest.fixture
def botorch_gp_model(simple_train_data):
    """Create a native BoTorch SingleTaskGP model.

    Args:
        simple_train_data: Fixture providing training data.

    Returns:
        Trained SingleTaskGP model.
    """
    train_X, train_Y = simple_train_data
    return SingleTaskGP(train_X.double(), train_Y.double())


# =============================================================================
# ALF GP Model Fixtures
# =============================================================================


@dataclass
class _TrainedGPModel:
    """Container for a trained GPModel and the candidates it was trained on."""

    model: GPModel
    train_candidates: list[Candidate]


@pytest.fixture
def trained_gp_model():
    """Train a GPModel on a handful of 2-feature tabular candidates.

    Returns:
        _TrainedGPModel with `model` (a trained GPModel exposing a
        `botorch_model` property) and `train_candidates` (the training
        candidates, e.g. for building an `X_baseline`).
    """
    X_train = np.array([[0.1, 0.2], [0.4, 0.5], [0.7, 0.8], [0.3, 0.6]], dtype=np.float64)
    y_train = np.array([1.0, 2.0, 1.5, 1.8])
    candidates = [Candidate(data=x, modality=Modality.TABULAR) for x in X_train]
    train_data = LabelledCandidates(candidates=candidates, labels=y_train)

    model = GPModel(
        train_config=GPTrainConfig(num_iterations=10),
        featurizer_config=FeaturizerConfig(featurizer_type="precomputed"),
        device="cpu",
    )
    model.train(train_data)
    return _TrainedGPModel(model=model, train_candidates=candidates)


# =============================================================================
# Dataset Fixtures
# =============================================================================


@dataclass
class _SimpleBraninDataset:
    """Minimal dataset container with train and test splits for GP tests."""

    train_dataset: LabelledCandidates
    test_dataset: LabelledCandidates


def _branin(x1: float, x2: float) -> float:
    """Evaluate the Branin function at (x1, x2).

    Args:
        x1: First input, typically in [-5, 10].
        x2: Second input, typically in [0, 15].

    Returns:
        Branin function value.
    """
    return (
        (x2 - (5.1 / (4 * math.pi**2)) * x1**2 + (5 / math.pi) * x1 - 6) ** 2
        + 10 * (1 - 1 / (8 * math.pi)) * math.cos(x1)
        + 10
    )


@pytest.fixture
def branin_dataset():
    """Create inline Branin evaluations for GP training tests.

    Generates 20 evaluations on a grid spanning the Branin domain
    (x1 in [-5, 10], x2 in [0, 15]) without using BoTorchSyntheticDataset.

    Returns:
        _SimpleBraninDataset with train_dataset (25 points) and
        test_dataset (10 points), both as LabelledCandidates.
    """
    x1_train = np.linspace(-5.0, 10.0, 5)
    x2_train = np.linspace(0.0, 15.0, 5)
    inputs = [(x1, x2) for x1 in x1_train for x2 in x2_train]  # 25 train points

    x1_test = np.linspace(-4.0, 9.0, 5)
    x2_test = np.linspace(1.0, 14.0, 5)
    test_inputs = [(x1, x2) for x1, x2 in zip(x1_test, x2_test)] + [
        (x1, x2) for x1, x2 in zip(x1_test, reversed(x2_test))
    ]  # 10 test points

    def make_labelled(pts: list[tuple[float, float]]) -> LabelledCandidates:
        candidates = [
            Candidate(data=np.array([x1, x2], dtype=np.float32), modality=Modality.TABULAR)
            for x1, x2 in pts
        ]
        labels = np.array([_branin(x1, x2) for x1, x2 in pts], dtype=np.float32)
        return LabelledCandidates(candidates=candidates, labels=labels)

    return _SimpleBraninDataset(
        train_dataset=make_labelled(inputs),
        test_dataset=make_labelled(test_inputs),
    )


# =============================================================================
# Surrogate Fixtures
# =============================================================================


@pytest.fixture
def trained_surrogate(branin_dataset):
    """Create and train a GP surrogate on the Branin dataset.

    Args:
        branin_dataset: Fixture providing a Branin dataset.

    Returns:
        Surrogate with trained GPModel.
    """
    gp_model = GPModel(
        featurizer_config=FeaturizerConfig(featurizer_type="precomputed"),
        train_config=GPTrainConfig(num_iterations=50, learning_rate=0.1),
        device="cpu",
    )
    surrogate = Surrogate(model=gp_model)
    surrogate.fit(branin_dataset.train_dataset, branin_dataset.test_dataset)
    return surrogate
