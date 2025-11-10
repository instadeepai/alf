from dataclasses import dataclass
from typing import Any


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
    features: dict | None = None

    def __post_init__(self) -> None:
        """Initialize features to empty dict if None."""
        if self.features is None:
            self.features = {}

    def __repr__(self) -> str:
        """Return a string representation of the candidate.

        Returns:
            str: A string representation showing the candidate's data, modality, and features.
        """
        return f"Candidate(data={self.data}, modality={self.modality}, features={self.features})"

    def stringify(self) -> str:
        """Convert the candidate data to a string representation.

        Returns:
            str: The string representation of the candidate's data.

        Raises:
            ValueError: If the modality is not "sequence" (other modalities not yet supported).
        """
        if self.modality == "sequence":
            return self.data
        else:
            # TODO: Implement stringification for other modalities
            # Once implemented, add test cases to test_candidate.py
            raise ValueError(f"Unsupported modality: {self.modality}")
