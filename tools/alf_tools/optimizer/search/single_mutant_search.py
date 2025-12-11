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

from typing import List

from alf_core import Candidate, SearchProtocol, TaskState
from alf_tools.utils.constants import PROTEIN_ALPHABET


class SingleMutantSearch(SearchProtocol):
    """Search protocol for single mutant search."""

    def __init__(self, alphabet: str = PROTEIN_ALPHABET):
        """Initialize the single mutant search protocol with the alphabet."""
        self.alphabet = alphabet

    def __call__(self, task_state: TaskState) -> List[Candidate]:
        """Apply the search protocol to return a pool of candidates.

        Args:
            task_state: The task state containing the dataset and surrogate model.

        Returns:
            A list of candidates.
        """
        # TODO: Add features to the candidates
        train_dataset = task_state.dataset.train_dataset
        best_id = train_dataset.labels.argmax()
        best_sequence = train_dataset.candidates[best_id].data
        single_mutant_pool = []
        for i in range(len(best_sequence)):
            for j in range(len(self.alphabet)):
                if best_sequence[i] == self.alphabet[j]:
                    continue
                single_mutant_pool.append(
                    Candidate(
                        data=best_sequence[:i] + self.alphabet[j] + best_sequence[i + 1 :],
                        modality="sequence",
                    )
                )
        return single_mutant_pool
