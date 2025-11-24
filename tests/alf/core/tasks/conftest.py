"""Shared pytest fixtures for experiment tests."""

import pytest

from alf.core.optimizer.optimizer import Optimizer
from alf.core.optimizer.search import DatasetSearch
from alf.core.oracle.oracle import Oracle
from alf.core.surrogate.surrogate import Surrogate

from .dummy_modules.acquisition_function import DummyAcquisitionFunction
from .dummy_modules.dataset import DummyDataset
from .dummy_modules.model import DummyModel


@pytest.fixture
def dummy_surrogate():
    """Fixture to create a dummy surrogate model for testing."""
    return Surrogate(model=DummyModel(seed=42))


@pytest.fixture
def dummy_dataset():
    """Fixture to create a dummy dataset for testing."""
    return DummyDataset(seed=42, num_samples=1000)


@pytest.fixture
def oracle(dummy_dataset):
    """Fixture to create an oracle for testing."""
    return Oracle(scorer=dummy_dataset)


@pytest.fixture
def dummy_optimizer():
    """Fixture to create a dummy optimizer for testing."""
    return Optimizer(acquisition_fn=DummyAcquisitionFunction(seed=42), search_fn=DatasetSearch())
