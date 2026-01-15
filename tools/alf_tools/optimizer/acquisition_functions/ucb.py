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


class UCB(AcquisitionFunction):
    """Upper Confidence Bound acquisition function."""

    def __init__(self, alpha: float):
        """Initialize UCB with exploration parameter alpha.

        Args:
            alpha: The exploration parameter.
        """
        self.alpha = alpha

    def _get_acquisition_values(self, features: Any, state: TaskState) -> np.ndarray:
        """Computes acquisition values for candidates based on surrogate predictions.

        Args:
            features: The features of the candidates which are the surrogate predictions.
            state: The task state containing the dataset and surrogate model.

        Returns:
            The acquisition values for the candidates.

        Raises:
            ValueError: If `variances` is not found in predictions.
        """
        predictions = features
        if predictions.variances is not None:
            sigma = np.sqrt(predictions.variances)
            mu = predictions.means
            return mu + self.alpha * sigma
        else:
            raise ValueError(
                "Expected `variances` in predictions, but was not found. Cannot compute UCB."
            )

    def _get_features(self, candidates: list[Candidate], state: TaskState) -> Any:
        """Get surrogate predictions of the candidates.

        Args:
            candidates: List of Candidate objects.
            state: Current task state.

        Returns:
            Predictions of the candidates.
        """
        return state.surrogate.predict(candidates)
