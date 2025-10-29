from core.optimizer.acquisition import AcquisitionFunction
from core.dataclasses.task_state import TaskState
from core.dataclasses.predictions import Predictions
import numpy as np
from scipy.stats import norm

class ExpectedImprovement(AcquisitionFunction):
    """Expected improvement acquisition function."""

    def acquire(self, predictions: Predictions, state: TaskState) -> np.ndarray:
        """Acquire the candidates."""
        best_f = state.best_f
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


class Greedy(AcquisitionFunction):
    """Greedy acquisition function."""

    def acquire(self, predictions: Predictions, state: TaskState) -> np.ndarray:
        """Acquire the candidates."""
        return predictions.means