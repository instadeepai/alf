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

"""Shared pytest fixtures for experiment tests."""

from typing import Any, List, Union

import numpy as np
import pytest
from alf_core.dataclasses import Candidate, LabeledCandidates, Predictions, TaskState
from alf_core.dataset.base_dataset import BaseDataset
from alf_core.model.base_model import BaseModel
from alf_core.optimizer.acquisition_function import AcquisitionFunction
from alf_core.optimizer.optimizer import Optimizer
from alf_core.optimizer.search import DatasetSearch
from alf_core.oracle.oracle import Oracle
from alf_core.surrogate.surrogate import Surrogate


@pytest.fixture
def dummy_surrogate():
    """Fixture to create a dummy surrogate model for testing.

    Returns:
        A dummy surrogate model for testing.
    """
    return Surrogate(model=DummyModel(seed=42))


@pytest.fixture
def dummy_dataset():
    """Fixture to create a dummy dataset for testing.

    Returns:
        A dummy dataset for testing.
    """
    return DummyDataset(seed=42, num_samples=1000)


@pytest.fixture
def dummy_dataset_factory():
    """Fixture factory to create dummy datasets with custom configurations.

    Returns:
        A factory function that creates DummyDataset instances.

    Example:
        def test_something(dummy_dataset_factory):
            split_config = {"split_ratio": {"train": 0.5, "test": 0.5}, "split_type": "random"}
            dataset = dummy_dataset_factory(split_config=split_config, num_samples=100)
    """

    def _create_dummy_dataset(
        split_config: dict[str, Any] | None = None,
        seed: int = 42,
        num_samples: int = 1000,
        name: str = "dummy",
        modality: str = "sequence",
    ) -> DummyDataset:
        """Create a DummyDataset with the specified configuration.

        Args:
            split_config: Split configuration dictionary
            seed: Random seed for reproducibility
            num_samples: Number of dummy samples to generate
            name: Name of the dataset
            modality: Modality of the data

        Returns:
            A dummy dataset with the specified configuration.
        """
        return DummyDataset(
            name=name,
            modality=modality,
            seed=seed,
            split_config=split_config,
            num_samples=num_samples,
        )

    return _create_dummy_dataset


@pytest.fixture
def oracle(dummy_dataset):
    """Fixture to create an oracle for testing.

    Args:
        dummy_dataset: A dummy dataset for testing.

    Returns:
        An oracle for testing.
    """
    return Oracle(scorer=dummy_dataset)


@pytest.fixture
def dummy_optimizer():
    """Fixture to create a dummy optimizer for testing.

    Returns:
        A dummy optimizer for testing.
    """
    return Optimizer(acquisition_fn=DummyAcquisitionFunction(seed=42), search_fn=DatasetSearch())


class DummyModel(BaseModel):
    """Dummy model that generates random predictions for testing."""

    def __init__(self, name: str = "dummy_model", seed: int = 42):
        """Initialize a dummy model.

        Args:
            name: Name of the model
            seed: Random seed for reproducibility
        """
        self.name = name
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def predict(self, candidate_points: List[Candidate]) -> Predictions:
        """Generate random predictions for the candidate points.

        Args:
            candidate_points: List of candidate points to predict.

        Returns:
            Predictions for the candidate points.
        """
        labels = self.rng.randn(len(candidate_points))
        return Predictions(means=labels)

    def featurise(self, inputs: Union[LabeledCandidates, List[Candidate]]) -> Any:
        """Dummy model does not perform featurisation."""
        pass

    def train(
        self,
        train_data: LabeledCandidates,
        val_data: LabeledCandidates,
    ) -> None:
        """Dummy model does not perform any actual training but updates the random seed."""
        self.rng = np.random.RandomState(self.seed + 1)

    def sample(self, *args: Any, **kwargs: Any) -> List[Candidate]:
        """Dummy model does not perform sampling."""
        raise NotImplementedError("Sampling is not implemented for this model.")


class DummyDataset(BaseDataset):
    """Dummy dataset that generates random data for testing."""

    def __init__(
        self,
        name: str = "dummy",
        modality: str = "sequence",
        seed: int = 42,
        split_config: dict[str, Any] | None = None,
        num_samples: int = 1000,
    ):
        """Initialize a dummy dataset.

        Args:
            name: Name of the dataset
            modality: Modality of the data (e.g., "sequence")
            seed: Random seed for reproducibility
            split_config: Split configuration dictionary
            num_samples: Number of dummy samples to generate
        """
        # Default split config if none provided
        if split_config is None:
            split_config = {
                "split_ratio": {"train": 0.6, "validation_frac": 0.2, "test": 0.2},
                "split_type": "random",
            }
        super().__init__(name, modality, seed, split_config)
        self.num_samples = num_samples
        self.setup()

    def load_dataset(self) -> LabeledCandidates:
        """Generate dummy dataset with random sequences and labels.

        Returns:
            LabeledCandidates with dummy data
        """
        # Generate dummy sequences (e.g., random strings)
        candidates = []
        labels = []

        for i in range(self.num_samples):
            # Generate a dummy sequence (e.g., random string of length 10)
            dummy_sequence = "".join(self.rng.choice(list("ACGT"), size=10))

            # Generate a dummy label (random float between 0 and 10)
            dummy_label = float(self.rng.uniform(0, 10))

            candidates.append(Candidate(data=dummy_sequence, modality=self.modality))
            labels.append(dummy_label)

        return LabeledCandidates(candidates=candidates, labels=np.array(labels))


class DummyAcquisitionFunction(AcquisitionFunction):
    """Dummy acquisition function that generates random acquisition values for testing."""

    def __init__(self, seed: int = 42):
        """Initialize the dummy acquisition function."""
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def _get_acquisition_values(self, predictions: Predictions, state: TaskState) -> np.ndarray:
        """Generate random acquisition values for the predictions.

        Args:
            predictions: Predictions from the surrogate model.
            state: Task state.

        Returns:
            Acquisition values.
        """
        return self.rng.randn(len(predictions))
