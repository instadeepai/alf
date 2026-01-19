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


import abc

from alf_core.dataclasses import Candidate, LabeledCandidates, TaskState


class AcquisitionFunction(abc.ABC):
    """Abstract base class for acquisition functions."""

    @abc.abstractmethod
    def __call__(
        self,
        search_candidates: list[Candidate],
        state: TaskState,
    ) -> LabeledCandidates:
        """Compute acquisition values for candidates and return them as LabeledCandidates.

        Args:
            search_candidates: List of unlabeled candidates to score.
            state: Current task state containing the dataset and surrogate model.

        Returns:
            Candidates paired with their acquisition values.
        """
        pass
