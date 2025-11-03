from alf.core.optimizer.acquisition_function import AcquisitionFunction
from alf.core.dataclasses.task_state import TaskState
from alf.core.dataclasses.predictions import Predictions
import numpy as np

class ThompsonSampling(AcquisitionFunction):
    """Thompson Sampling acquisition function.
    
    A generalisation of Thompson Sampling to the case where batch size > num posterior samples.

    For each point, we find its maximum rank under any ensemble member,
    when predictions are sorted in ascending order.
    (Higher predictions correspond to higher ranks)
    We return the maximum rank for each candidate as an acquisition value, so that higher
    is better.
    """

    def acquire(self, predictions: Predictions, state: TaskState) -> np.ndarray:
        """Scores the candidates based on the surrogate model's predictions."""
        if predictions.empirical_dist is not None:
            samples = predictions.empirical_dist
            ranks = samples.argsort(axis=0).argsort(axis=0) + 1
            return ranks.max(-1)
        elif predictions.variances is not None:
            # NOTE: This needs to be implemented for GP
            raise NotImplementedError
        else:
            raise ValueError(
                "Expected either `empirical_dist` or `variances` in predictions, "
                "but neither was found. Cannot perform Thomson Sampling."
            )