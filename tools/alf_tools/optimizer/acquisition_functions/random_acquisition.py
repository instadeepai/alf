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


class RandomAcquisition(AcquisitionFunction):
    """Random acquisition function — the mandatory "floor" baseline.

    Assigns each candidate a uniform random score, completely ignoring the
    surrogate's predictions (mean or uncertainty). Any acquisition strategy
    that does not clearly beat Random is not adding value over blind
    selection, which is why every comparison should include this floor.

    A fresh RNG is derived per round from ``(seed, state.round_metrics.round)``
    so that draws decorrelate across rounds (no two rounds pick candidates in
    the same relative order) while remaining fully reproducible for a given
    seed.
    """

    def __init__(self, seed: int):
        """Initialize RandomAcquisition with a base seed.

        Args:
            seed: Base random seed. Combined with the current round number to
                derive a per-round RNG.
        """
        self.seed = seed

    def __call__(self, search_candidates: list[Candidate], state: State) -> LabelledCandidates:
        """Assign uniform random acquisition scores to unlabelled candidates.

        Args:
            search_candidates: List of unlabelled candidates to score.
            state: The task state, used only to read the current round number
                (the surrogate is never consulted).

        Returns:
            LabelledCandidates with uniform random acquisition values in [0, 1).
        """
        rng = np.random.default_rng((self.seed, state.round_metrics.round))
        acquisition_values = rng.random(len(search_candidates))
        return LabelledCandidates(candidates=search_candidates, labels=acquisition_values)
