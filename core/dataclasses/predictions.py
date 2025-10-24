from dataclasses import dataclass
import numpy as np
from core.dataclasses.candidate import Candidate
from typing import List
import pandas as pd
import os
from core.utils.io import input_handler


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

    def __post_init__(self) -> None:
        assert len(self.means) > 0, "Means must have at least one prediction"
        if self.variances is not None:
            assert len(self.variances) == len(self.means), "Variances must have the same length as means"
        if self.empirical_dist is not None:
            assert len(self.empirical_dist) == len(self.means), "Empirical dist must have the same length as means"

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
        predictions_list = []
        for i in range(len(self.means)):
            record_i = {
                "sequence": candidates[i].data,
                "mean": self.means[i],
                "variance": self.variances[i] if self.variances is not None else 0,
                "targets": targets[i],
            }

            if self.empirical_dist is not None:
                for j in range(self.empirical_dist.shape[1]):
                    record_i[f"ensemble_pred_{j}"] = self.empirical_dist[i, j]

            predictions_list.append(record_i)

        df = pd.DataFrame.from_records(predictions_list)
        input_handler.save_csv(os.path.join(output_dir, filename), df)

        return