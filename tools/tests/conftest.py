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

This module provides fixtures for BoTorch GP model, acquisition, dataset,
and integration tests.
"""

from typing import Union

import numpy as np
import pytest
import torch
from alf_core import (
    BaseDatasetConfig,
    Candidate,
    LabelledCandidates,
    Modality,
    Optimizer,
    Oracle,
    Predictions,
    State,
    Surrogate,
)
from alf_core.model.base_model import BaseModel
from alf_tools.datasets.botorch_synthetic_dataset import BoTorchSyntheticDataset
from alf_tools.models.botorch_exact_gp_model import BoTorchGPModel
from alf_tools.optimizer.acquisition_functions.botorch_acquisition import BoTorchAcquisition
from alf_tools.optimizer.acquisition_functions.botorch_samplers import BoTorchMCSampler
from alf_tools.optimizer.search.botorch_search_functions import ContinuousSearch
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
    return SingleTaskGP(train_X, train_Y)


# =============================================================================
# Dataset Fixtures
# =============================================================================


@pytest.fixture
def branin_dataset():
    """Create a Branin BoTorchSyntheticDataset for GP training and e2e tests.

    Generates 500 random evaluations of the Branin function, split into:
    - train: ~40 points  (train_ratio=0.1, validation_frac=0.2)
    - validation: ~10 points
    - test: ~200 points  (test_ratio=0.4, enough for discrete pool tests up to 120)
    - candidate_pool: remainder

    Returns:
        BoTorchSyntheticDataset with train/validation/test splits ready for use.
    """
    config = BaseDatasetConfig(
        name="branin",
        modality=Modality.TABULAR,
        seed=42,
        train_ratio=0.1,
        validation_frac=0.2,
        test_ratio=0.4,
        split_type="random",
        problem_type="regression",
    )
    dataset = BoTorchSyntheticDataset(
        config=config,
        function_name="branin",
        noise_std=0.0,
        n_initial_samples=500,
        negate=True,
    )
    dataset.setup()
    return dataset


# =============================================================================
# Surrogate Fixtures
# =============================================================================


@pytest.fixture
def trained_surrogate(branin_dataset):
    """Create and train a GP surrogate on the Branin dataset.

    Args:
        branin_dataset: Fixture providing a Branin dataset.

    Returns:
        Surrogate with trained BoTorchGPModel.
    """
    gp_model = BoTorchGPModel(num_iterations=50, learning_rate=0.1)
    surrogate = Surrogate(model=gp_model)
    surrogate.fit(branin_dataset.train_dataset, branin_dataset.test_dataset)
    return surrogate


@pytest.fixture
def gp_surrogate():
    """Create an untrained GP surrogate for design task e2e tests.

    Returns:
        Surrogate wrapping an untrained BoTorchGPModel (model attribute is None).
    """
    gp_model = BoTorchGPModel(num_iterations=50, learning_rate=0.1)
    return Surrogate(model=gp_model)


@pytest.fixture
def botorch_optimizer(branin_dataset):
    """Create a BoTorch optimizer for continuous Branin optimization.

    Uses qEI with Sobol sampler and ContinuousSearch (empty candidate pool),
    triggering gradient-based acquisition optimization in the Branin bounds.

    Args:
        branin_dataset: Fixture providing the Branin dataset (for bounds).

    Returns:
        Optimizer with BoTorchAcquisition (qEI, batch_size=2) and ContinuousSearch.
    """
    # Build [[lower_i, upper_i], ...] from branin_dataset.bounds shape (2, dim)
    bounds = [
        [float(branin_dataset.bounds[0][i]), float(branin_dataset.bounds[1][i])]
        for i in range(branin_dataset.dim)
    ]
    sampler = BoTorchMCSampler(sampler_type="sobol", num_samples=64, seed=42)
    acq = BoTorchAcquisition(
        acquisition_type="qEI",
        sampler=sampler,
        bounds=bounds,
        batch_size=2,
        num_restarts=3,
        raw_samples=64,
    )
    return Optimizer(acquisition_fn=acq, search_fn=ContinuousSearch())


@pytest.fixture
def branin_oracle(branin_dataset):
    """Create an oracle that evaluates candidates on the Branin function.

    Args:
        branin_dataset: Fixture providing the Branin dataset (used as scorer).

    Returns:
        Oracle wrapping the BoTorchSyntheticDataset query method.
    """
    return Oracle(scorer=branin_dataset)


@pytest.fixture
def state(branin_dataset, trained_surrogate):
    """Create a task State with the Branin dataset and a trained GP surrogate.

    Used by TestBoTorchAcquisitionDiscreteScoring tests that call
    acquisition_fn(candidates, state) directly.

    Args:
        branin_dataset: Fixture providing the Branin dataset.
        trained_surrogate: Fixture providing a trained GP surrogate.

    Returns:
        State with dataset and trained surrogate (round=0, acq_batch_size=0).
    """
    return State(dataset=branin_dataset, surrogate=trained_surrogate)
