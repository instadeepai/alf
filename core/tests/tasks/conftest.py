"""Shared pytest fixtures for experiment tests."""

import pytest
from alf_core.optimizer.optimizer import Optimizer
from alf_core.optimizer.search import DatasetSearch
from alf_core.oracle.oracle import Oracle
from alf_core.surrogate.surrogate import Surrogate

from core.tests.dummy_modules import DummyAcquisitionFunction, DummyDataset, DummyModel


@pytest.fixture
def dummy_surrogate():
    """Fixture to create a dummy surrogate model for testing.

    Returns:
        Surrogate: A dummy surrogate model for testing.
    """
    return Surrogate(model=DummyModel(seed=42))


@pytest.fixture
def dummy_dataset():
    """Fixture to create a dummy dataset for testing.

    Returns:
        DummyDataset: A dummy dataset for testing.
    """
    return DummyDataset(seed=42, num_samples=1000)


@pytest.fixture
def oracle(dummy_dataset):
    """Fixture to create an oracle for testing.

    Args:
        dummy_dataset: A dummy dataset for testing.

    Returns:
        Oracle: An oracle for testing.
    """
    return Oracle(scorer=dummy_dataset)


@pytest.fixture
def dummy_optimizer():
    """Fixture to create a dummy optimizer for testing.

    Returns:
        Optimizer: A dummy optimizer for testing.
    """
    return Optimizer(acquisition_fn=DummyAcquisitionFunction(seed=42), search_fn=DatasetSearch())
