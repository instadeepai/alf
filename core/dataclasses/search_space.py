from dataclasses import dataclass, field
from typing import List, Optional, Union
import numpy as np


@dataclass
class SearchSpace:
    """Defines the search space for active learning.
    
    Args:
        candidate_pool: The pool of candidates to search from (offline optimization), 
            can be None if the search space is not defined by a fixed dataset (online optimization).
    """

    candidate_pool: Optional[List["Candidate"]] = None

    def __len__(self) -> int:
        if self.candidate_pool is not None:
            return len(self.candidate_pool)
        return 0

    def __iter__(self):
        if self.candidate_pool is not None:
            return iter(self.candidate_pool)
        return iter([])

    @property
    def candidate_data(self) -> Optional[List[Union[str, object]]]:
        """Return the raw data for each candidate (sequence, graph, image, etc.)."""
        if self.candidate_pool is not None:
            return [cand.data for cand in self.candidate_pool]
        return None

    def contains(self, candidate: "Candidate") -> bool:
        """Check if a candidate is in the dataset."""
        return candidate.data in set(self.candidate_data or [])

    def remove(self, candidates: List["Candidate"]) -> None:
        """Remove candidates from the dataset."""
        if self.candidate_pool is None:
            return

        for cand in candidates:
            if self.contains(cand):
                self.candidate_pool.remove(cand)


    