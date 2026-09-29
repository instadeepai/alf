# Copyright 2026 InstaDeep Ltd. All rights reserved.
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
from alf_core import AcquisitionFunction, Candidate, LabelledCandidates, State


class UncertaintySampling(AcquisitionFunction):
    """Uncertainty sampling acquisition function.

    Scores each candidate by the surrogate's predictive uncertainty alone,
    ignoring the predicted mean: `acquisition = σ`. This is the pure-exploration
    counterpart to Greedy (pure exploitation), and reproduces UCB's ranking in the
    limit of a large exploration parameter. It is the natural baseline when
    the goal is to *improve the surrogate* (model-quality metrics such as test
    RMSE) rather than to find high-scoring candidates.

    Uncertainty is read from `variances` when the surrogate reports it. For an
    ensemble surrogate that only reports per-member predictions
    (`empirical_dist`), the disagreement between members — their standard
    deviation across the ensemble axis — is used instead.

    This is a maximising acquisition function.
    """

    def __call__(self, search_candidates: list[Candidate], state: State) -> LabelledCandidates:
        """Compute uncertainty-sampling acquisition values for unlabelled candidates.

        Args:
            search_candidates: List of unlabelled candidates to score.
            state: The task state containing the current datasets and surrogate model.

        Raises:
            ValueError: If neither `variances` nor `empirical_dist` is found in predictions.

        Returns:
            LabelledCandidates with predictive standard deviation as acquisition values.
        """
        predictions = state.surrogate.predict(search_candidates)
        # EI and Thompson Sampling check `empirical_dist` first as they need the full
        # sample distribution; a scalar spread is enough here, so prefer `variances`.
        if predictions.variances is not None:
            # Clip: tiny negative variances from numerical error would sqrt to NaN.
            acquisition_values = np.sqrt(np.maximum(predictions.variances, 0.0))
        elif predictions.empirical_dist is not None:
            acquisition_values = predictions.empirical_dist.std(axis=-1)
        else:
            raise ValueError(
                "Expected either `variances` or `empirical_dist` in predictions, "
                "but neither was found. Cannot perform uncertainty sampling."
            )
        return LabelledCandidates(candidates=search_candidates, labels=acquisition_values)
