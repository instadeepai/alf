from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class Candidate:
    """A candidate is a data point with a modality and features."""
    data: Any
    modality: str
    features: Optional[dict] = None

    def __post_init__(self):
        if self.features is None:
            self.features = {}

    def __repr__(self) -> str:
        return f"Candidate(data={self.data}, modality={self.modality}, features={self.features})"
    
    def stringify(self) -> str:
        """Convert the candidate data to a string."""
        if self.modality == "sequence":
            return self.data
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")
    