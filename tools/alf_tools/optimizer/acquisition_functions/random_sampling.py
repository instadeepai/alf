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


class RandomSampling(AcquisitionFunction):
    """Random-sampling acquisition baseline.

    Scores every candidate uniformly at random, ignoring the surrogate entirely. It is the
    standard control for demonstrating that an informed acquisition function actually helps.
    The RNG is seeded by combining `seed` with the acquisition round so that successive rounds
    draw different scores while staying reproducible.
    """

    def __init__(self, seed: int = 0) -> None:
        """Initialise the random-sampling baseline.

        Args:
            seed: Non-negative base random seed (default 0). It is combined with the
                acquisition round so successive rounds draw different scores while staying
                reproducible; pass a distinct seed per experiment replication so the
                replications themselves decorrelate.

        Raises:
            ValueError: If `seed` is negative.
        """
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}.")
        self.seed = seed

    def __call__(self, search_candidates: list[Candidate], state: State) -> LabelledCandidates:
        """Assign each unlabelled candidate a uniform random acquisition value.

        Args:
            search_candidates: List of unlabelled candidates to score.
            state: The task state, used for the per-round RNG seed.

        Returns:
            LabelledCandidates with uniform random acquisition values in [0, 1).
        """
        rng = np.random.default_rng((self.seed, state.round_metrics.round))
        acquisition_values = rng.random(len(search_candidates))
        return LabelledCandidates(candidates=search_candidates, labels=acquisition_values)
