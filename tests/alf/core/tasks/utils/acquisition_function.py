import numpy as np
from alf.core.dataclasses import Predictions, TaskState
from alf.core.optimizer.acquisition_function import AcquisitionFunction


class DummyAcquisitionFunction(AcquisitionFunction):
    """Dummy acquisition function that generates random acquisition values for testing."""

    def __init__(self, seed: int = 42):
        """Initialize the dummy acquisition function."""
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def _get_acquisition_values(self, predictions: Predictions, state: TaskState) -> np.ndarray:
        """Generate random acquisition values for the predictions."""
        return self.rng.randn(len(predictions))
