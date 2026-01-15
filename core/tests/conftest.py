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

from typing import Any, List, Literal, Union

import numpy as np
import pytest
from alf_core.dataclasses import Candidate, LabeledCandidates, Predictions, TaskState
from alf_core.dataset.base_dataset import BaseDataset, BaseDatasetConfig
from alf_core.model.base_model import BaseModel
from alf_core.optimizer.acquisition_function import AcquisitionFunction
from alf_core.optimizer.optimizer import Optimizer
from alf_core.optimizer.search import DatasetSearch
from alf_core.oracle.oracle import Oracle
from alf_core.surrogate.surrogate import Surrogate


class DummyDatasetConfig(BaseDatasetConfig):
    """Configuration for DummyDataset.

    Attributes:
        num_samples: Number of dummy samples to generate.
    """

    num_samples: int = 1000


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
    config = DummyDatasetConfig(
        name="dummy",
        modality="sequence",
        seed=42,
        train_ratio=0.6,
        validation_frac=0.2,
        test_ratio=0.2,
        split_type="random",
        num_samples=1000,
    )
    return DummyDataset(config)


@pytest.fixture
def dummy_dataset_factory():
    """Fixture factory to create dummy datasets with custom configurations.

    Returns:
        A factory function that creates DummyDataset instances.

    Example:
        def test_something(dummy_dataset_factory):
            config = DummyDatasetConfig(
                name="test", modality="sequence", seed=42,
                train_ratio=0.5, validation_frac=0.2, test_ratio=0.3,
                split_type="random", num_samples=100
            )
            dataset = dummy_dataset_factory(config=config)
    """

    def _create_dummy_dataset(
        config: DummyDatasetConfig | None = None,
        # Legacy parameters for backwards compatibility with existing tests
        seed: int = 42,
        num_samples: int = 1000,
        name: str = "dummy",
        modality: str = "sequence",
        train_ratio: float = 0.6,
        validation_frac: float = 0.2,
        test_ratio: float = 0.2,
        split_type: Literal["random", "low_vs_high"] = "random",
        max_candidate_pool: int | None = None,
    ) -> DummyDataset:
        """Create a DummyDataset with the specified configuration.

        Args:
            config: Full configuration object (preferred).
            seed: Random seed for reproducibility.
            num_samples: Number of dummy samples to generate.
            name: Name of the dataset.
            modality: Modality of the data.
            train_ratio: Fraction of data for training.
            validation_frac: Fraction of training data for validation.
            test_ratio: Fraction of data for testing.
            split_type: Type of split.
            max_candidate_pool: Maximum candidate pool size.

        Returns:
            A dummy dataset with the specified configuration.
        """
        if config is None:
            config = DummyDatasetConfig(
                name=name,
                modality=modality,
                seed=seed,
                train_ratio=train_ratio,
                validation_frac=validation_frac,
                test_ratio=test_ratio,
                split_type=split_type,
                max_candidate_pool=max_candidate_pool,
                num_samples=num_samples,
            )
        return DummyDataset(config)

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

    def __init__(self, config: DummyDatasetConfig):
        """Initialize a dummy dataset.

        Args:
            config: Configuration for the dummy dataset.
        """
        super().__init__(config)
        self.setup()

    def load_dataset(self) -> LabeledCandidates:
        """Generate dummy dataset with random sequences and labels.

        Returns:
            LabeledCandidates with dummy data
        """
        candidates = []
        labels = []

        for _ in range(self.config.num_samples):
            # Generate a dummy sequence (random string of length 10)
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

    def _get_features(self, candidates: list[Candidate], state: TaskState) -> Any:
        """Get surrogate predictions of the candidates.

        Args:
            candidates: List of Candidate objects.
            state: Current task state.

        Returns:
            Predictions of the candidates.
        """
        return state.surrogate.predict(candidates)
