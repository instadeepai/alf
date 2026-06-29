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


from alf_core import AcquisitionFunction, Candidate, LabelledCandidates, State


class UncertaintySampling(AcquisitionFunction):
    """Uncertainty-sampling (maximum-variance) acquisition function.

    Scores each candidate by its predictive variance, ignoring the mean: the candidates the
    surrogate is least certain about score highest. This is the natural acquisition for
    *reducing model error* (query-by-committee / uncertainty sampling), as opposed to the
    maximising acquisitions (`Greedy`, `UCB`, `ExpectedImprovement`, ...) that chase a high
    predicted target value. It requires an uncertainty-aware surrogate, for example an
    ensemble (`EnsembleWrapper`) or a Gaussian process, whose `predict` populates `variances`.
    """

    def __call__(self, search_candidates: list[Candidate], state: State) -> LabelledCandidates:
        """Compute uncertainty-sampling acquisition values for unlabelled candidates.

        Args:
            search_candidates: List of unlabelled candidates to score.
            state: The task state containing the current datasets and surrogate model.

        Raises:
            ValueError: If `variances` is not found in predictions.

        Returns:
            LabelledCandidates whose labels are the per-candidate predictive variances
            (higher variance = more uncertain = more informative to label).
        """
        predictions = state.surrogate.predict(search_candidates)
        if predictions.variances is None:
            raise ValueError(
                "Expected `variances` in predictions, but none were found. UncertaintySampling "
                "requires an uncertainty-aware surrogate (e.g. an ensemble or a GP)."
            )
        return LabelledCandidates(candidates=search_candidates, labels=predictions.variances)
