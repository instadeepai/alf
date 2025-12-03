import numpy as np
from alf_core import AcquisitionFunction, Predictions, TaskState


class Greedy(AcquisitionFunction):
    """Greedy acquisition function."""

    def _get_acquisition_values(self, predictions: Predictions, state: TaskState) -> np.ndarray:
        """Computes acquisition values for candidates based on surrogate predictions.

        Args:
            predictions: The predictions from the surrogate model.
            state: The task state containing the dataset and surrogate model.

        Returns:
            np.ndarray: The acquisition values for the candidates.
        """
        return predictions.means
