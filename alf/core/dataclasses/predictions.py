# Copyright 2023 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alf.core.dataclasses.candidate import Candidate
from alf.core.utils.io import input_handler


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
        """Validate that predictions arrays have consistent lengths.

        Raises:
            AssertionError: If means is empty, or if variances or empirical_dist
                don't match the length of means.
        """
        assert len(self.means) > 0, "Means must have at least one prediction"
        if self.variances is not None:
            assert len(self.variances) == len(self.means), (
                "Variances must have the same length as means"
            )
        if self.empirical_dist is not None:
            assert len(self.empirical_dist) == len(self.means), (
                "Empirical dist must have the same length as means"
            )

    def __len__(self) -> int:
        """Return the number of predictions.

        Returns:
            int: The number of predictions (length of the means array).
        """
        return len(self.means)

    def save(
        self,
        output_dir: str,
        candidates: list[Candidate],
        targets: np.ndarray,
        filename: str,
    ) -> None:
        """Save the predictions to disk as a CSV file.

        Creates a DataFrame with predictions, targets, and optionally variances
        and ensemble predictions, then saves it to the specified directory.

        Args:
            output_dir: Directory path where the CSV file will be saved.
            candidates: List of Candidate objects corresponding to the predictions.
            targets: Ground truth target values corresponding to each candidate.
            filename: Name of the CSV file to save (e.g., "predictions.csv").
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
