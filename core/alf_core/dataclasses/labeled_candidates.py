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


from dataclasses import dataclass
from typing import Any, Union

import numpy as np
import pandas as pd
from alf_core.dataclasses.candidate import Candidate


@dataclass
class LabeledCandidates:
    """A collection of candidates paired with their labels.

    Attributes:
        candidates: A list of Candidate objects.
        labels: A numpy array of labels corresponding to each candidate.
    """

    candidates: list[Candidate]
    labels: np.ndarray

    def __post_init__(self) -> None:
        """Validate that candidates and labels have the same length.

        Raises:
            AssertionError: If the length of candidates and labels don't match.
        """
        assert len(self.candidates) == len(self.labels), (
            "Candidates and labels must have the same length"
        )

    def __len__(self) -> int:
        """Return the number of candidates in the collection.

        Returns:
            The number of candidates (and labels) in the collection.
        """
        return len(self.candidates)

    def __getitem__(
        self, index: Union[int, slice, np.ndarray]
    ) -> tuple[list[Candidate], np.ndarray]:
        """Make LabeledCandidates subscriptable.

        Args:
            index: Integer index or slice to select candidates and labels.

        Returns:
            The candidates and labels at the specified index or slice.
        """
        if isinstance(index, int):
            return ([self.candidates[index]], [self.labels[index]])
        elif isinstance(index, np.ndarray):
            return ([self.candidates[i] for i in index], self.labels[index])
        else:
            return (self.candidates[index], self.labels[index])

    @property
    def data(self) -> list[Any]:
        """Return the raw data of each candidate.

        Returns:
            A list containing the raw data (sequence, graph, image, etc.)
            of each candidate in the collection.
        """
        return [cand.data for cand in self.candidates]

    def append(
        self,
        candidates: Union[list[Candidate], "LabeledCandidates"],
        labels: np.ndarray | None = None,
    ) -> None:
        """Append candidates and labels to this collection.

        Args:
            candidates: Either a list of Candidate objects or another LabeledCandidates
                object. If a list is provided, labels must also be provided.
            labels: Optional numpy array of labels. Required if candidates is a list,
                ignored if candidates is a LabeledCandidates object.

        Raises:
            AssertionError: If candidates is a list and labels is None, or if the
                length of candidates and labels don't match.
        """
        if isinstance(candidates, LabeledCandidates):
            self.candidates.extend(candidates.candidates)
            self.labels = np.concatenate((self.labels, candidates.labels), axis=0)
        else:
            assert labels is not None and len(candidates) == len(labels), (
                "Candidates and labels must have the same length"
            )
            self.candidates.extend(candidates)
            self.labels = np.concatenate((self.labels, labels), axis=0)

    def shuffle(self, seed: int) -> "LabeledCandidates":
        """Create a new LabeledCandidates object with shuffled candidates and labels.

        Args:
            seed: Random seed for reproducibility of the shuffle.

        Returns:
            A new LabeledCandidates object with the same candidates and labels,
            but in a randomly shuffled order.
        """
        shuffled_indices = np.random.RandomState(seed).permutation(len(self.candidates))
        return LabeledCandidates(
            candidates=[self.candidates[i] for i in shuffled_indices],
            labels=self.labels[shuffled_indices],
        )

    def sort(self, ascending: bool = True) -> "LabeledCandidates":
        """Sort the candidates and labels by label values.

        Args:
            ascending: If True, sort in ascending order (lowest to highest).
                If False, sort in descending order (highest to lowest).
                Defaults to True.

        Returns:
            A new LabeledCandidates object with candidates and labels sorted
            by label values.
        """
        sorted_indices = np.argsort(self.labels)
        if not ascending:
            sorted_indices = sorted_indices[::-1]
        return LabeledCandidates(
            candidates=[self.candidates[i] for i in sorted_indices],
            labels=self.labels[sorted_indices],
        )

    def remove(self, candidates: Union[list[Candidate], "LabeledCandidates"]) -> None:
        """Remove specified candidates and their corresponding labels from this collection.

        Candidates not present in the collection are silently ignored.

        Args:
            candidates: Either a list of Candidate objects or a LabeledCandidates
                object containing the candidates to remove.
        """
        if isinstance(candidates, LabeledCandidates):
            candidates = candidates.candidates

        # Filter to only remove candidates that are actually present in the collection
        candidates_to_remove = [c for c in candidates if c in self.candidates]
        if not candidates_to_remove:
            return

        indices_to_remove = [self.candidates.index(cand) for cand in candidates_to_remove]
        self.labels = np.delete(self.labels, indices_to_remove)
        self.candidates = [cand for cand in self.candidates if cand not in candidates_to_remove]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the labeled candidates to a pandas DataFrame.

        Returns:
            A DataFrame with columns:
            - "data": The stringified data of each candidate
            - "label": The label value for each candidate
            - Additional columns for any features present in the candidates
        """
        rows = []
        for cand, label in zip(self.candidates, self.labels):
            d = {"data": cand.stringify(), "label": label}
            if cand.features is not None and isinstance(cand.features, dict):
                for k, v in cand.features.items():
                    d[k] = v
            rows.append(d)

        return pd.DataFrame.from_records(rows)

    def get_top_k(self, k: int) -> "LabeledCandidates":
        """Return the top k candidates based on their label values.

        Args:
            k: Number of top candidates to select.

        Returns:
            New LabeledCandidates object containing the top k candidates sorted
            by label values (highest first).
        """
        top_k_indices = self.labels.argsort()[::-1][:k]
        top_k_candidates = [self.candidates[i] for i in top_k_indices]
        top_k_labels = self.labels[top_k_indices]
        return LabeledCandidates(candidates=top_k_candidates, labels=top_k_labels)

    def __iter__(self):
        """Iterate over candidates and labels, yielding (candidate, label) tuples.

        Yields:
            Tuple of (Candidate, float): A candidate and its corresponding label
        """
        for candidate, label in zip(self.candidates, self.labels):
            yield candidate, label
