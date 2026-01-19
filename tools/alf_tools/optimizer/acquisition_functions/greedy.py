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


class Greedy(AcquisitionFunction):
    """Greedy acquisition function."""

    def __call__(self, search_candidates: list[Candidate], state: TaskState) -> LabeledCandidates:
        """Generate greedy acquisition values for the unlabeled candidates.

        Args:
            search_candidates: List of unlabeled candidates to score.
            state: Current task state containing the dataset and surrogate model.

        Returns:
            Predictions of the candidates.
        """
        predictions = state.surrogate.predict(search_candidates)
        acquisition_values = predictions.means
        return LabeledCandidates(candidates=search_candidates, labels=acquisition_values)
