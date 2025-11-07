from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Candidate:
    """A candidate is a data point with a modality and features.

    Attributes:
        data: The raw data of the candidate (e.g., sequence string, graph, image).
        modality: The type/modality of the data (e.g., "sequence", "graph", "image").
        features: Optional dictionary of precomputed features for the candidate.
    """

    data: Any
    modality: str
    features: Optional[dict] = None

    def __post_init__(self):
        if self.features is None:
            self.features = {}

    def __repr__(self) -> str:
        """Return a string representation of the candidate."""
        return f"Candidate(data={self.data}, modality={self.modality}, features={self.features})"

    def stringify(self) -> str:
        """Convert the candidate data to a string."""
        if self.modality == "sequence":
            return self.data
        else:
            # TODO: Implement stringification for other modalities
            # Once implemented, add test cases to test_candidate.py
            raise ValueError(f"Unsupported modality: {self.modality}")
