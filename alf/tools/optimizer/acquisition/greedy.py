from alf.core.optimizer.acquisition_function import AcquisitionFunction
from alf.core.dataclasses.task_state import TaskState
from alf.core.dataclasses.predictions import Predictions
import numpy as np

class Greedy(AcquisitionFunction):
    """Greedy acquisition function."""

    def _get_acquisition_values(self, predictions: Predictions, state: TaskState) -> np.ndarray:
        """Computes acquisition values for candidates based on surrogate predictions."""
        return predictions.means