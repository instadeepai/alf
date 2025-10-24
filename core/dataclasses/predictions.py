from dataclasses import dataclass
import numpy as np
from core.dataclasses.candidate import Candidate
from typing import List


@dataclass
class Predictions:
    """A data class for storing and managing predictions from a model.

    Attributes:
        means: A numpy array of mean predictions across all candidates.
        variances: An optional numpy array of prediction variances, representing
            the model's uncertainty for each prediction.
        empirical_dist: An optional 2D numpy array containing predictions
            from individual models in an ensemble. Typically has the shape
            (num_candidates, num_ensemble_models).
    """


    means: np.ndarray
    variances: np.ndarray | None = None
    empirical_dist: np.ndarray | None = None

    def save_predictions(
        self,
        output_dir: str,
        candidates: List[Candidate],
        targets: np.ndarray,
        filename: str,
    ) -> None:
        """
        Save the predictions to disk.

        First create a dataframe of {(sequence_i, mean_i target_i)} and then save to disk.
        Additionally, if they exist then save variances, and empirical dists.
        """
        # TODO: Implement this
        pass