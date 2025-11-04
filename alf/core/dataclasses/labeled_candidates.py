from dataclasses import dataclass
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from alf.core.dataclasses.candidate import Candidate


@dataclass
class LabeledCandidates:
    """A collection of candidates paired with their labels."""

    candidates: List["Candidate"]
    labels: np.ndarray

    def __post_init__(self) -> None:
        assert len(self.candidates) == len(self.labels), (
            "Candidates and labels must have the same length"
        )

    def __len__(self) -> int:
        """Return the number of candidates in the collection."""
        return len(self.candidates)

    def __getitem__(self, index: Union[int, slice]) -> "LabeledCandidates":
        """Make LabeledCandidates subscriptable.

        Args:
            index: Integer index or slice

        Returns:
            For integer index: new LabeledCandidates object with the candidate and label at the index
            For slice: new LabeledCandidates object with sliced data
        """
        if isinstance(index, int):
            return LabeledCandidates(
                candidates=[self.candidates[index]],
                labels=np.array([self.labels[index]]),
            )
        elif isinstance(index, slice):
            return LabeledCandidates(
                candidates=self.candidates[index], labels=self.labels[index]
            )
        else:
            raise TypeError(
                f"Indices must be integers or slices, not {type(index).__name__}"
            )

    @property
    def data(self) -> List[Any]:
        """Return the raw data (sequence, graph, image, etc.) of each candidate."""
        return [cand.data for cand in self.candidates]

    def validate_candidates(self, candidates: List["Candidate"]) -> bool:
        """Validate candidates from this collection."""
        return all(candidate in self.candidates for candidate in candidates)

    def append(
        self,
        candidates: Union[List["Candidate"], "LabeledCandidates"],
        labels: Optional[np.ndarray] = None,
    ) -> None:
        """Append candidates and labels to this collection.

        Args:
            candidates: Either a list of Candidate objects or another LabeledCandidates
            labels: Optional labels array (required if candidates is a list)

        Raises:
            ValueError: If candidates and labels don't match in length
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
        """Create a new LabeledCandidates object with shuffled candidates and labels."""
        shuffled_indices = np.random.RandomState(seed).permutation(len(self.candidates))
        return LabeledCandidates(
            candidates=[self.candidates[i] for i in shuffled_indices],
            labels=self.labels[shuffled_indices],
        )

    def sort(self, ascending: bool = True) -> "LabeledCandidates":
        """Sort the candidates and labels by the labels."""
        sorted_indices = np.argsort(self.labels)
        if not ascending:
            sorted_indices = sorted_indices[::-1]
        return LabeledCandidates(
            candidates=[self.candidates[i] for i in sorted_indices],
            labels=self.labels[sorted_indices],
        )

    def remove(self, candidates: Union[List["Candidate"], "LabeledCandidates"]) -> None:
        """Remove candidates from this collection."""
        if isinstance(candidates, LabeledCandidates):
            candidates = candidates.candidates

        assert self.validate_candidates(candidates), (
            "Candidates must be in this collection"
        )
        self.labels = np.delete(
            self.labels, [self.candidates.index(cand) for cand in candidates]
        )
        self.candidates = [cand for cand in self.candidates if cand not in candidates]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the labeled candidates to a pandas dataframe."""
        rows = []
        for cand, label in zip(self.candidates, self.labels):
            d = {"data": cand.stringify(), "label": label}
            if cand.features is not None and isinstance(cand.features, dict):
                for k, v in cand.features.items():
                    d[k] = v
            rows.append(d)

        return pd.DataFrame.from_records(rows)
