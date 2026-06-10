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

"""Shared fixtures: synthetic AL components and a registry wired to them.

These dummies let the runner integration test exercise the full sweep without
heavy dependencies, training, or network access.
"""

from typing import Any

import numpy as np
import pytest
from alf_benchmark.registry import Registry, default_registry
from alf_core.dataclasses import Candidate, LabelledCandidates, Predictions, State
from alf_core.dataset.base_dataset import BaseDataset, BaseDatasetConfig
from alf_core.model.base_model import BaseModel
from alf_core.optimizer.acquisition_function import AcquisitionFunction
from alf_core.optimizer.search import DatasetSearch


class DummyDatasetConfig(BaseDatasetConfig):
    """Dataset config adding a sample count.

    Attributes:
        num_samples: Number of synthetic samples to generate.
    """

    num_samples: int = 300


class DummyDataset(BaseDataset):
    """Synthetic dataset of random sequences and labels."""

    def __init__(self, config: DummyDatasetConfig) -> None:
        """Initialise and load the dummy dataset.

        Args:
            config: Configuration for the dummy dataset.
        """
        super().__init__(config)
        self.setup()

    def load_dataset(self) -> LabelledCandidates:
        """Generate random sequences with random float labels.

        Returns:
            The generated labelled candidates.
        """
        candidates = []
        labels = []
        for _ in range(self.config.num_samples):
            sequence = "".join(self.rng.choice(list("ACGT"), size=10))
            candidates.append(Candidate(data=sequence, modality=self.modality))
            labels.append(float(self.rng.uniform(0, 10)))
        return LabelledCandidates(candidates=candidates, labels=np.array(labels))


class DummyModel(BaseModel):
    """Model returning seeded random predictions; performs no real training."""

    def __init__(self, name: str = "dummy_model", seed: int = 42) -> None:
        """Initialise the dummy model.

        Args:
            name: Model name.
            seed: Seed for the prediction RNG.
        """
        self.name = name
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Return random predictions for the candidates.

        Args:
            candidate_points: Candidates to score.

        Returns:
            Predictions with random means.
        """
        return Predictions(means=self.rng.randn(len(candidate_points)))

    def featurise(self, inputs: Any) -> Any:
        """No-op featurisation.

        Args:
            inputs: Ignored.

        Returns:
            ``None``.
        """
        return None

    def train(self, train_data: LabelledCandidates, val_data: LabelledCandidates) -> None:
        """Reset the RNG to mimic a deterministic retrain.

        Args:
            train_data: Ignored training data.
            val_data: Ignored validation data.
        """
        self.rng = np.random.RandomState(self.seed + 1)

    def sample(self, *args: Any, **kwargs: Any) -> list[Candidate]:
        """Sampling is unsupported.

        Args:
            *args: Ignored.
            **kwargs: Ignored.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError("Sampling is not implemented for DummyModel.")


class DummyAcquisitionFunction(AcquisitionFunction):
    """Acquisition function assigning seeded random scores."""

    def __init__(self, seed: int = 42) -> None:
        """Initialise the dummy acquisition function.

        Args:
            seed: Seed for the scoring RNG.
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, search_candidates: list[Candidate], state: State) -> LabelledCandidates:
        """Score candidates with random acquisition values.

        Args:
            search_candidates: Candidates to score.
            state: Current task state.

        Returns:
            Candidates paired with random acquisition values.
        """
        values = self.rng.randn(len(search_candidates))
        return LabelledCandidates(candidates=search_candidates, labels=values)


@pytest.fixture
def registry() -> Registry:
    """Provide a registry pre-populated with the synthetic components.

    Returns:
        A registry with dummy dataset, model, acquisition, and search registered.
    """
    reg = Registry()
    reg.register("alf.datasets", "dummy", DummyDataset)
    reg.register("alf.models", "dummy", DummyModel)
    reg.register("alf.acquisition_functions", "dummy_acq", DummyAcquisitionFunction)
    reg.register("alf.searches", "dataset_search", DatasetSearch)
    return reg


@pytest.fixture
def dummies_in_default_registry():
    """Register the synthetic components in the shared registry, then clean up.

    Yields:
        The shared registry, so CLI paths that use it can resolve the dummies.
    """
    reg = default_registry()
    entries = {
        ("alf.datasets", "dummy"): DummyDataset,
        ("alf.models", "dummy"): DummyModel,
        ("alf.acquisition_functions", "dummy_acq"): DummyAcquisitionFunction,
    }
    for (group, name), component in entries.items():
        reg.register(group, name, component)
    yield reg
    for group, name in entries:
        reg.unregister(group, name)
