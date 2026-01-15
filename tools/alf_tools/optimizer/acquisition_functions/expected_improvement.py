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

from typing import Any

import numpy as np
from alf_core import AcquisitionFunction, Candidate, TaskState
from scipy.stats import norm


class ExpectedImprovement(AcquisitionFunction):
    """Expected improvement acquisition function."""

    def _get_acquisition_values(self, features: Any, state: TaskState) -> np.ndarray:
        """Computes acquisition values for candidates based on their features.

        Args:
            features: The features of the candidates which are the surrogate predictions.
            state: The task state containing the dataset and surrogate model.

        Returns:
            The acquisition values for the candidates.

        Raises:
            ValueError: If `empirical_dist` or `variances` is not found in predictions.
        """
        predictions = features
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

    def _get_features(self, candidates: list[Candidate], state: TaskState) -> Any:
        """Get surrogatepredictions of the candidates.

        Args:
            candidates: List of Candidate objects.
            state: Current task state.

        Returns:
            Predictions of the candidates.
        """
        return state.surrogate.predict(candidates)
