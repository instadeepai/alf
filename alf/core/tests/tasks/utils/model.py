from typing import Any, List, Union

import numpy as np

from alf_core.dataclasses import Candidate, LabeledCandidates, Predictions
from alf_core.model.base_model import BaseModel


class DummyModel(BaseModel):
    """Dummy model that generates random predictions for testing."""

    def __init__(self, name: str = "dummy_model", seed: int = 42):
        """Initialize a dummy model.

        Args:
            name: Name of the model
            seed: Random seed for reproducibility
        """
        self.name = name
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def predict(self, candidate_points: List[Candidate]) -> Predictions:
        """Generate random predictions for the candidate points."""
        labels = self.rng.randn(len(candidate_points))
        return Predictions(means=labels)

    def featurise(self, inputs: Union[LabeledCandidates, List[Candidate]]) -> Any:
        """Dummy model does not perform featurisation."""
        pass

    def train(
        self,
        train_data: LabeledCandidates,
        val_data: LabeledCandidates,
    ) -> None:
        """Dummy model does not perform any actual training but updates the random seed."""
        self.rng = np.random.RandomState(self.seed + 1)

    def sample(self, *args: Any, **kwargs: Any) -> List[Candidate]:
        """Dummy model does not perform sampling."""
        raise NotImplementedError("Sampling is not implemented for this model.")
