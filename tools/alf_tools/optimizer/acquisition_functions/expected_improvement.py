import numpy as np
from alf_core import AcquisitionFunction, Predictions, TaskState
from scipy.stats import norm


class ExpectedImprovement(AcquisitionFunction):
    """Expected improvement acquisition function."""

    def _get_acquisition_values(self, predictions: Predictions, state: TaskState) -> np.ndarray:
        """Computes acquisition values for candidates based on surrogate predictions.

        Args:
            predictions: The predictions from the surrogate model.
            state: The task state containing the dataset and surrogate model.

        Returns:
            np.ndarray: The acquisition values for the candidates.

        Raises:
            ValueError: If `empirical_dist` or `variances` is not found in predictions.
        """
        best_f = state.dataset.train_dataset.labels.max()
        if predictions.empirical_dist is not None:
            return np.mean(np.maximum(predictions.empirical_dist - best_f, 0), -1)
        elif predictions.variances is not None:
            mu = predictions.means
            sigma = np.sqrt(predictions.variances)

            sigma = np.clip(sigma, 1e-9, None)

            z = (mu - best_f) / sigma
            ei = (mu - best_f) * norm.cdf(z) + sigma * norm.pdf(z)
            return np.maximum(ei, 0.0)
        else:
            raise ValueError(
                "Expected either `empirical_dist` or `variances` in predictions, "
                "but neither was found. Cannot compute expected improvement."
            )
