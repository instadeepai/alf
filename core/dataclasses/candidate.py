from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

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
