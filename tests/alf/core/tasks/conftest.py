"""Shared pytest fixtures for experiment tests."""
import pytest

from alf.core.surrogate.surrogate import Surrogate
from alf.core.oracle.oracle import Oracle
from tests.alf.core.tasks.dummy_components import DummyModel, DummyDataset


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
    return Oracle(module=dummy_dataset)
