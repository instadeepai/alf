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


from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from alf_core.dataclasses.candidate import Candidate
from alf_core.enums import ProblemType


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
        """Validate prediction arrays have consistent lengths.

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
            The number of predictions (length of the means array).
        """
        return len(self.means)

    def to_dataframe(
        self,
        candidates: list[Candidate],
        targets: np.ndarray,
        problem_type: Optional[ProblemType] = None
    ) -> pd.DataFrame:
        """Convert predictions to a DataFrame.

        Creates a DataFrame with predictions, targets, and optionally variances
        and ensemble predictions.

        Args:
            candidates: List of Candidate objects corresponding to the predictions.
            targets: Ground truth target values corresponding to each candidate.

        Returns:
            A DataFrame with predictions, targets, and optionally variances
            and ensemble predictions.
        """
        predictions_list = []
        if problem_type is None:
            is_classification = self.means.ndim == 2
        else:
            is_classification = problem_type in [ProblemType.BINARY, ProblemType.MULTICLASS]

        for i in range(len(self.means)):
            record_i: dict[str, Any] = {"sequence": candidates[i].data, "targets": targets[i]}

            if is_classification:
                for cls_idx in range(self.means.shape[1]):
                    record_i[f"prob_class_{cls_idx}"] = self.means[i, cls_idx]
            else:
                record_i["mean"] = self.means[i]
                record_i["variance"] = self.variances[i] if self.variances is not None else 0

            if self.empirical_dist is not None:
                for j in range(self.empirical_dist.shape[1]):
                    record_i[f"ensemble_pred_{j}"] = self.empirical_dist[i, j]

            predictions_list.append(record_i)

        df = pd.DataFrame.from_records(predictions_list)
        return df
