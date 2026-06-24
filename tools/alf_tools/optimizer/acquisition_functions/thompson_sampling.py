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


class ThompsonSampling(AcquisitionFunction):
    """Thompson Sampling acquisition function.

    A generalisation of Thompson Sampling to the case where batch size > num posterior samples.

    For an ensemble surrogate (`empirical_dist`), each point is scored by its maximum rank
    under any ensemble member, when predictions are sorted in ascending order.
    (Higher predictions correspond to higher ranks.)
    We return the maximum rank for each candidate as an acquisition value, so that higher
    is better.

    For a Gaussian-process surrogate (per-candidate `means`/`variances` but no
    `empirical_dist`), we draw one posterior sample per candidate from
    `N(mean, std)` and use the sample as the acquisition value (higher is better).
    The sampling RNG is seeded by combining the `seed` argument with the round,
    so seed replications decorrelate while staying reproducible.

    This is a maximising acquisition function.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialise Thompson Sampling.

        Args:
            seed: Base random seed for the Gaussian-process sampling path. It is
                combined with the acquisition round so that draws decorrelate
                across rounds while staying reproducible; pass a distinct seed per
                experiment replication so the replications themselves decorrelate.
                None (the default) draws from fresh OS entropy on every round
                (non-reproducible). Unused by the ensemble (`empirical_dist`) path.
        """
        self.seed = seed

    def __call__(self, search_candidates: list[Candidate], state: State) -> LabelledCandidates:
        """Compute Thompson Sampling acquisition values for unlabelled candidates.

        Args:
            search_candidates: List of unlabelled candidates to score.
            state: The task state containing the current datasets and surrogate model.

        Raises:
            ValueError: If neither `empirical_dist` nor `variances` is found in predictions.

        Returns:
            LabelledCandidates with Thompson Sampling acquisition values.
        """
        predictions = state.surrogate.predict(search_candidates)
        if predictions.empirical_dist is not None:
            samples = predictions.empirical_dist
            ranks = samples.argsort(axis=0).argsort(axis=0) + 1
            acquisition_values = ranks.max(-1)
        elif predictions.variances is not None:
            # GP path: draw one posterior sample per candidate from N(mean, std).
            # Seed on (seed, round) so each replication draws an independent
            # standard-normal vector instead of sharing one across runs; an unset
            # seed falls back to fresh OS entropy.
            entropy = (self.seed, state.round_metrics.round) if self.seed is not None else None
            rng = np.random.default_rng(entropy)
            acquisition_values = rng.normal(
                loc=predictions.means, scale=np.sqrt(np.maximum(predictions.variances, 0.0))
            )
        else:
            raise ValueError(
                "Expected either `empirical_dist` or `variances` in predictions, "
                "but neither was found. Cannot perform Thomson Sampling."
            )
        return LabelledCandidates(candidates=search_candidates, labels=acquisition_values)
