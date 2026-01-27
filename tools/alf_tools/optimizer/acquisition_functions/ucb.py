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


import numpy as np
from alf_core import AcquisitionFunction, Candidate, LabelledCandidates, TaskState


class UCB(AcquisitionFunction):
    """Upper Confidence Bound acquisition function.

    The UCB acquisition value is given by: UCB = μ + ασ, where μ is the mean prediction,
    σ is the standard deviation, and α is the exploration parameter.
    Higher α values lead to more exploration, while lower α values lead to more exploitation.
    This is a maximising acquisition function.
    """

    def __init__(self, alpha: float):
        """Initialize UCB with exploration parameter alpha.

        Args:
            alpha: The exploration parameter.
        """
        self.alpha = alpha

    def __call__(self, search_candidates: list[Candidate], state: TaskState) -> LabelledCandidates:
        """Compute Upper Confidence Bound (UCB) acquisition values for unlabelled candidates.

        Args:
            search_candidates: List of unlabelled candidates to score.
            state: The task state containing the current datasets and surrogate model.

        Raises:
            ValueError: If `variances` is not found in predictions.

        Returns:
            LabelledCandidates with UCB acquisition values.
        """
        predictions = state.surrogate.predict(search_candidates)
        if predictions.variances is not None:
            sigma = np.sqrt(predictions.variances)
            mu = predictions.means
            acquisition_values = mu + self.alpha * sigma
        else:
            raise ValueError(
                "Expected `variances` in predictions, but was not found. Cannot compute UCB."
            )
        return LabelledCandidates(candidates=search_candidates, labels=acquisition_values)
