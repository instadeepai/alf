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
from alf_core import Candidate, Modality, SearchProtocol, State
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

        Label ties are broken by training-set position, matching ``labels.argmax()``.

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
        num_seeds = min(self.top_k, len(train_dataset.candidates))
        if num_seeds == 0:
            raise ValueError(
                "SingleMutantSearch requires at least one training candidate to mutate, "
                "but state.dataset.train_dataset is empty."
            )

        labels = np.asarray(train_dataset.labels)
        # Shape (n, 1) is a column vector of scalar labels, so squeeze it. Genuinely
        # multi-output labels can't be ranked without a scalarisation, and argsort would
        # return indices that don't line up with `candidates` — so fail instead.
        if labels.ndim > 1:
            squeezable = [axis for axis in range(1, labels.ndim) if labels.shape[axis] == 1]
            labels = labels.reshape(labels.shape[0], -1).squeeze(axis=1) if squeezable else labels
        if labels.ndim > 1:
            raise ValueError(
                f"SingleMutantSearch ranks training candidates by a single scalar label per "
                f"candidate, but got labels with shape {np.asarray(train_dataset.labels).shape}. "
                "Reduce multi-output labels to one objective (e.g. by scalarising them) before "
                "using this search protocol."
            )

        # Stable sort on negated labels keeps tied candidates in training-set order, so
        # the top-1 seed is exactly `labels.argmax()`. Do not "simplify" to
        # `argsort(labels)[::-1]`: it reverses tied runs, silently changing which seed
        # `top_k=1` picks when the best label is duplicated.
        ranked_indices = np.argsort(-labels, kind="stable")[:num_seeds]

        single_mutant_pool: List[Candidate] = []
        seen: set = set()
        for index in ranked_indices:
            seed_sequence = train_dataset.candidates[index].data
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
