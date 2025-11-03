from alf.core.optimizer.acquisition import AcquisitionFunction
from alf.core.dataclasses.task_state import TaskState
from alf.core.dataclasses.predictions import Predictions
import numpy as np

class UCB(AcquisitionFunction):
    """Upper Confidence Bound acquisition function."""

    def __init__(self, alpha: float):
        """Initialize UCB with exploration parameter alpha."""
        self.alpha = alpha

    def acquire(self, predictions: Predictions, state: TaskState) -> np.ndarray:
        """Scores the candidates based on the surrogate model's predictions."""
        if predictions.variances is not None:
            sigma = np.sqrt(predictions.variances)
            mu = predictions.means
            return mu + self.alpha * sigma
        else:
            raise ValueError(
                "Expected `variances` in predictions, but was not found. Cannot compute UCB."
            )