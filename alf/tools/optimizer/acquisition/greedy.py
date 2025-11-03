from core.optimizer.acquisition import AcquisitionFunction
from core.dataclasses.task_state import TaskState
from core.dataclasses.predictions import Predictions
import numpy as np

class Greedy(AcquisitionFunction):
    """Greedy acquisition function."""

    def acquire(self, predictions: Predictions, state: TaskState) -> np.ndarray:
        """Scores the candidates based on the surrogate model's predictions."""
        return predictions.means