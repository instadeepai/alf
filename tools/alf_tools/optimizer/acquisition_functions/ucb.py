import numpy as np
from alf_core import AcquisitionFunction, Predictions, TaskState


class UCB(AcquisitionFunction):
    """Upper Confidence Bound acquisition function."""

    def __init__(self, alpha: float):
        """Initialize UCB with exploration parameter alpha.

        Args:
            alpha: The exploration parameter.
        """
        self.alpha = alpha

    def _get_acquisition_values(self, predictions: Predictions, state: TaskState) -> np.ndarray:
        """Computes acquisition values for candidates based on surrogate predictions.

        Args:
            predictions: The predictions from the surrogate model.
            state: The task state containing the dataset and surrogate model.

        Returns:
            np.ndarray: The acquisition values for the candidates.

        Raises:
            ValueError: If `variances` is not found in predictions.
        """
        if predictions.variances is not None:
            sigma = np.sqrt(predictions.variances)
            mu = predictions.means
            return mu + self.alpha * sigma
        else:
            raise ValueError(
                "Expected `variances` in predictions, but was not found. Cannot compute UCB."
            )
