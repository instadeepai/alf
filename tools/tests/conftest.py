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

"""Shared pytest fixtures for all tools tests.

This module provides commonly used fixtures that can be reused across all test
files in the tools/tests directory, including:
- Mock ALF models (with and without variances)
- BoTorch model fixtures (SingleTaskGP, training data)
- Dataset fixtures (Branin synthetic datasets)
- Surrogate and task state fixtures
- Common test data (candidates, tensors)
"""

from typing import Any, List, Union

import numpy as np
import pytest
import torch
from alf_core import (
    BaseDatasetConfig,
    Candidate,
    LabelledCandidates,
    Modality,
    Predictions,
    Surrogate,
)
from alf_core.dataclasses.task_state import TaskState
from alf_core.model.base_model import BaseModel
from alf_tools.datasets.botorch_test_functions import BoTorchSyntheticDataset
from alf_tools.models.botorch_exact_gp_model import BoTorchGPModel
from botorch.models import SingleTaskGP

# =============================================================================
# Mock ALF Models
# =============================================================================


class MockALFModelWithVariances(BaseModel):
    """Mock ALF BaseModel that provides variances for testing.

    This model is useful for testing code that requires uncertainty estimates.
    Predictions return sequential means (0, 1, 2, ...) plus an offset, and
    constant variances.

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

    def predict(self, candidate_points: List[Candidate]) -> Predictions:
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

    def featurise(self, inputs: Union[LabelledCandidates, List[Candidate]]) -> Any:
        """Not implemented for mock model."""
        pass

    def train(self, train_data: LabelledCandidates, val_data: LabelledCandidates) -> None:
        """Not implemented for mock model."""
        pass

    def sample(self, *args: Any, **kwargs: Any) -> List[Candidate]:
        """Not implemented for mock model."""
        raise NotImplementedError()


class MockALFModelWithoutVariances(BaseModel):
    """Mock ALF BaseModel that does NOT provide variances (deterministic).

    This model is useful for testing error handling when models don't provide
    uncertainty estimates. Some models like CNNs may not provide variances.

    Predictions return sequential means (0, 1, 2, ...) with no variances.
    """

    def predict(self, candidate_points: List[Candidate]) -> Predictions:
        """Generate predictions WITHOUT variances.

        Args:
            candidate_points: List of candidates to predict.

        Returns:
            Predictions with only means (no variances).
        """
        n = len(candidate_points)
        means = np.arange(n, dtype=np.float32)
        return Predictions(means=means, variances=None)

    def featurise(self, inputs: Union[LabelledCandidates, List[Candidate]]) -> Any:
        """Not implemented for mock model."""
        pass

    def train(self, train_data: LabelledCandidates, val_data: LabelledCandidates) -> None:
        """Not implemented for mock model."""
        pass

    def sample(self, *args: Any, **kwargs: Any) -> List[Candidate]:
        """Not implemented for mock model."""
        raise NotImplementedError()


@pytest.fixture
def mock_alf_model_with_variances():
    """Create an ALF model that provides variances.

    Returns:
        MockALFModelWithVariances with default parameters (offset=5.0, variance=0.2).

    Example:
        >>> def test_something(mock_alf_model_with_variances):
        ...     predictions = mock_alf_model_with_variances.predict(candidates)
        ...     assert predictions.variances is not None
    """
    return MockALFModelWithVariances(mean_offset=5.0, variance_value=0.2)


@pytest.fixture
def mock_alf_model_without_variances():
    """Create an ALF model that does NOT provide variances.

    Returns:
        MockALFModelWithoutVariances instance.

    Example:
        >>> def test_error_handling(mock_alf_model_without_variances):
        ...     predictions = mock_alf_model_without_variances.predict(candidates)
        ...     assert predictions.variances is None
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

    Example:
        >>> def test_model(simple_train_data):
        ...     train_X, train_Y = simple_train_data
        ...     model = SingleTaskGP(train_X, train_Y)
    """
    torch.manual_seed(42)
    train_X = torch.rand(10, 2, dtype=torch.float32)
    train_Y = torch.sin(train_X[:, 0] * 3.14159) + torch.cos(train_X[:, 1] * 3.14159)
    train_Y = train_Y.unsqueeze(-1)  # Shape (10, 1)
    return train_X, train_Y


@pytest.fixture
def botorch_gp_model(simple_train_data):
    """Create a native BoTorch SingleTaskGP model.

    Args:
        simple_train_data: Fixture providing training data.

    Returns:
        Trained SingleTaskGP model.

    Example:
        >>> def test_botorch_integration(botorch_gp_model):
        ...     posterior = botorch_gp_model.posterior(test_X)
        ...     assert posterior.mean.shape[0] == len(test_X)
    """
    train_X, train_Y = simple_train_data
    model = SingleTaskGP(train_X, train_Y)
    return model


# =============================================================================
# Dataset Fixtures
# =============================================================================


@pytest.fixture
def base_dataset_config():
    """Create a base dataset configuration for testing.

    Returns:
        BaseDatasetConfig with standard test parameters.

    Example:
        >>> def test_dataset(base_dataset_config):
        ...     config = base_dataset_config
        ...     assert config.seed == 42
    """
    return BaseDatasetConfig(
        name="test_dataset",
        modality=Modality.TABULAR,
        seed=42,
        train_ratio=0.1,
        validation_frac=0.2,
        test_ratio=0.1,
        split_type="random",
    )


@pytest.fixture
def branin_dataset():
    """Create a Branin synthetic dataset for testing.

    Returns:
        BoTorchSyntheticDataset configured for Branin function with 500 samples.

    Example:
        >>> def test_with_branin(branin_dataset):
        ...     train_data = branin_dataset.train_dataset
        ...     assert len(train_data.candidates) > 0
    """
    config = BaseDatasetConfig(
        name="test_branin",
        modality=Modality.TABULAR,
        seed=42,
        train_ratio=0.05,
        validation_frac=0.2,
        test_ratio=0.1,
        split_type="random",
    )

    dataset = BoTorchSyntheticDataset(
        config=config,
        function_name="branin",
        noise_std=0.0,
        n_initial_samples=500,
    )
    dataset.setup()
    return dataset


@pytest.fixture
def branin_dataset_factory():
    """Factory for creating Branin datasets with custom configurations.

    Returns:
        Function that creates BoTorchSyntheticDataset instances.

    Example:
        >>> def test_custom_branin(branin_dataset_factory):
        ...     dataset = branin_dataset_factory(n_samples=100, noise_std=0.1)
        ...     assert dataset.noise_std == 0.1
    """

    def _create_branin_dataset(
        n_samples: int = 500,
        noise_std: float = 0.0,
        train_ratio: float = 0.05,
        seed: int = 42,
    ) -> BoTorchSyntheticDataset:
        """Create a Branin dataset with specified parameters.

        Args:
            n_samples: Number of initial samples.
            noise_std: Standard deviation of observation noise.
            train_ratio: Fraction of data for training.
            seed: Random seed for reproducibility.

        Returns:
            Configured and setup BoTorchSyntheticDataset.
        """
        config = BaseDatasetConfig(
            name="test_branin",
            modality=Modality.TABULAR,
            seed=seed,
            train_ratio=train_ratio,
            validation_frac=0.2,
            test_ratio=0.1,
            split_type="random",
        )

        dataset = BoTorchSyntheticDataset(
            config=config,
            function_name="branin",
            noise_std=noise_std,
            n_initial_samples=n_samples,
        )
        dataset.setup()
        return dataset

    return _create_branin_dataset


# =============================================================================
# Surrogate and Task State Fixtures
# =============================================================================


@pytest.fixture
def trained_surrogate(branin_dataset):
    """Create and train a GP surrogate on the Branin dataset.

    Args:
        branin_dataset: Fixture providing a Branin dataset.

    Returns:
        Surrogate with trained BoTorchGPModel.

    Example:
        >>> def test_surrogate(trained_surrogate):
        ...     predictions = trained_surrogate.predict(test_candidates)
        ...     assert predictions.means is not None
    """
    gp_model = BoTorchGPModel(num_iterations=50, learning_rate=0.1)
    surrogate = Surrogate(model=gp_model)
    surrogate.fit(branin_dataset.train_dataset, branin_dataset.validation_dataset)
    return surrogate


@pytest.fixture
def task_state(branin_dataset, trained_surrogate):
    """Create a task state for testing.

    Args:
        branin_dataset: Fixture providing a Branin dataset.
        trained_surrogate: Fixture providing a trained surrogate.

    Returns:
        TaskState with dataset and surrogate.

    Example:
        >>> def test_task_state(task_state):
        ...     predictions = task_state.surrogate.predict(candidates)
        ...     best_label = np.max(task_state.dataset.train_dataset.labels)
    """
    return TaskState(dataset=branin_dataset, surrogate=trained_surrogate)


# =============================================================================
# Common Test Data Fixtures
# =============================================================================


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
def test_tensor_3d():
    """Create 3D test tensor (batch format) for BoTorch operations.

    Returns:
        Torch tensor of shape (2, 3, 2) representing 2 batches of 3 points each.

    Example:
        >>> def test_batched_posterior(test_tensor_3d):
        ...     # Shape: (batch_size=2, q=3, d=2)
        ...     posterior = model.posterior(test_tensor_3d)
    """
    # Shape: (2, 3, 2) - 2 batches, 3 points each, 2 dimensions
    return torch.tensor(
        [
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            [[0.7, 0.8], [0.9, 0.1], [0.2, 0.3]],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def test_bounds_2d():
    """Create 2D bounds for optimization problems.

    Returns:
        List of tuples representing bounds: [(0.0, 1.0), (0.0, 1.0)].

    Example:
        >>> def test_optimization(test_bounds_2d):
        ...     acq_fn = BoTorchAcquisition(bounds=test_bounds_2d, ...)
    """
    return [(0.0, 1.0), (0.0, 1.0)]


# =============================================================================
# Utility Fixtures
# =============================================================================


@pytest.fixture
def random_seed():
    """Set a fixed random seed for reproducibility.

    Yields:
        The seed value (42), and resets numpy/torch random state after test.

    Example:
        >>> def test_reproducible(random_seed):
        ...     # All random operations will be reproducible
        ...     data = np.random.randn(100)
    """
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    yield seed
    # Cleanup - reset to random state
    np.random.seed(None)
