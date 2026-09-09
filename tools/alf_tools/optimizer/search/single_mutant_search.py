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

from typing import List

import numpy as np
from alf_core import Candidate, LabelledCandidates, Modality, SearchProtocol, State
from alf_tools.utils.constants import PROTEIN_ALPHABET


class SingleMutantSearch(SearchProtocol):
    """Search protocol that enumerates single-point mutants of the top-K training sequences.

    For each of the ``top_k`` highest-labelled training sequences, every single-position
    substitution over ``alphabet`` is enumerated. Seed neighbourhoods overlap, so the
    union is deduplicated by sequence in generation order (not via set iteration) to keep
    the pool deterministic. ``top_k=1``, the default, is a pure hill-climb on a single
    neighbourhood; higher values keep several local optima under exploration at once.
    """

    def __init__(self, alphabet: str = PROTEIN_ALPHABET, top_k: int = 1):
        """Initialize the single mutant search protocol with the alphabet.

        Args:
            alphabet: Characters substituted in at each position of each seed sequence.
            top_k: Number of best-labelled training sequences to mutate from. 1
                reproduces single-best hill-climbing; values above the training set
                size are clamped to it.

        Raises:
            ValueError: If ``top_k`` is less than 1.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}.")
        self.alphabet = alphabet
        self.top_k = top_k

    def __call__(self, state: State) -> List[Candidate]:
        """Apply the search protocol to return a pool of candidates.

        Seeds are ranked by ``LabelledCandidates.get_top_k``, which breaks label ties by
        training-set position, so ``top_k=1`` selects the same seed as ``labels.argmax()``.

        Args:
            state: The task state containing the dataset and surrogate model.

        Returns:
            A deduplicated list of candidates, ordered by seed rank then by mutation
            position and alphabet order.

        Raises:
            ValueError: If the training set is empty, or if the training labels have more
                than one meaningful dimension.
        """
        train_dataset = state.dataset.train_dataset
        if len(train_dataset.candidates) == 0:
            raise ValueError(
                "SingleMutantSearch requires at least one training candidate to mutate, "
                "but state.dataset.train_dataset is empty."
            )

        labels = np.asarray(train_dataset.labels)
        # Shape (n, 1) is a column vector of scalar labels, so squeeze it before ranking.
        # Genuinely multi-output labels can't be ranked without a scalarisation, and
        # get_top_k would sort along the wrong axis and mis-select seeds, so fail instead.
        if labels.ndim > 1:
            squeezable = [axis for axis in range(1, labels.ndim) if labels.shape[axis] == 1]
            if not squeezable:
                raise ValueError(
                    "SingleMutantSearch ranks training candidates by a single scalar label "
                    f"per candidate, but got labels with shape {labels.shape}. Reduce "
                    "multi-output labels to one objective (e.g. by scalarising them) before "
                    "using this search protocol."
                )
            train_dataset = LabelledCandidates(
                candidates=train_dataset.candidates,
                labels=labels.reshape(labels.shape[0], -1).squeeze(axis=1),
            )

        seeds = train_dataset.get_top_k(self.top_k).candidates

        single_mutant_pool: List[Candidate] = []
        seen: set = set()
        for seed in seeds:
            seed_sequence = seed.data
            for i in range(len(seed_sequence)):
                for character in self.alphabet:
                    if seed_sequence[i] == character:
                        continue
                    mutant = seed_sequence[:i] + character + seed_sequence[i + 1 :]
                    if mutant in seen:
                        continue
                    seen.add(mutant)
                    single_mutant_pool.append(Candidate(data=mutant, modality=Modality.SEQUENCE))
        return single_mutant_pool
