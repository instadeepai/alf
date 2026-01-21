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


from alf_core import AcquisitionFunction, Candidate, LabeledCandidates, TaskState


class ThompsonSampling(AcquisitionFunction):
    """Thompson Sampling acquisition function.

    A generalisation of Thompson Sampling to the case where batch size > num posterior samples.

    For each point, we find its maximum rank under any ensemble member,
    when predictions are sorted in ascending order.
    (Higher predictions correspond to higher ranks)
    We return the maximum rank for each candidate as an acquisition value, so that higher
    is better.
    This is a maximising acquisition function.
    """

    def __call__(self, search_candidates: list[Candidate], state: TaskState) -> LabeledCandidates:
        """Compute Thompson Sampling acquisition values for unlabelled candidates.

        Args:
            search_candidates: List of unlabelled candidates to score.
            state: The task state containing the current datasets and surrogate model.

        Raises:
            ValueError: If `empirical_dist` or `variances` is not found in predictions.

        Returns:
            LabeledCandidates with Thompson Sampling acquisition values.
        """
        predictions = state.surrogate.predict(search_candidates)
        if predictions.empirical_dist is not None:
            samples = predictions.empirical_dist
            ranks = samples.argsort(axis=0).argsort(axis=0) + 1
            acquisition_values = ranks.max(-1)
        elif predictions.variances is not None:
            # NOTE: This needs to be implemented for GP
            raise NotImplementedError
        else:
            raise ValueError(
                "Expected either `empirical_dist` or `variances` in predictions, "
                "but neither was found. Cannot perform Thomson Sampling."
            )
        return LabeledCandidates(candidates=search_candidates, labels=acquisition_values)
