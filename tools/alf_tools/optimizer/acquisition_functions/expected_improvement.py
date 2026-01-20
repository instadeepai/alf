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
from alf_core import AcquisitionFunction, Candidate, LabeledCandidates, TaskState
from scipy.stats import norm


class ExpectedImprovement(AcquisitionFunction):
    """Expected improvement acquisition function.
    
    The Expected Improvement acquisition value is given by:
    EI = (μ - best_f) * Φ(z) + σ * φ(z), where μ is the mean prediction, 
    σ is the standard deviation, best_f is the best observed value, 
    and Φ and φ are the CDF and PDF of the standard normal distribution.
    Higher EI values indicate more improvement over the best observed value.
    """

    def __call__(self, search_candidates: list[Candidate], state: TaskState) -> LabeledCandidates:
        """Compute Expected Improvement acquisition values for unlabelled candidates.

        Args:
            search_candidates: List of unlabelled candidates to score.
            state: The task state containing the current datasets and surrogate model.

        Raises:
            ValueError: If `empirical_dist` or `variances` is not found in predictions.

        Returns:
            LabeledCandidates with Expected Improvement acquisition values.
        """
        predictions = state.surrogate.predict(search_candidates)
        best_f = state.dataset.train_dataset.labels.max()
        if predictions.empirical_dist is not None:
            acquisition_values = np.mean(np.maximum(predictions.empirical_dist - best_f, 0), -1)
        elif predictions.variances is not None:
            mu = predictions.means
            sigma = np.sqrt(predictions.variances)

            sigma = np.clip(sigma, 1e-9, None)

            z = (mu - best_f) / sigma
            ei = (mu - best_f) * norm.cdf(z) + sigma * norm.pdf(z)
            acquisition_values = np.maximum(ei, 0.0)
        else:
            raise ValueError(
                "Expected either `empirical_dist` or `variances` in predictions, "
                "but neither was found. Cannot compute expected improvement."
            )
        return LabeledCandidates(candidates=search_candidates, labels=acquisition_values)
