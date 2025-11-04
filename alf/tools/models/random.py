from typing import Any, List, Optional, Union

import numpy as np

from alf.core.dataclasses import Candidate, LabeledCandidates, Predictions
from alf.core.model.base_model import BaseModel
from alf.core.utils.logger import Logger


class RandomModel(BaseModel):
    def __init__(self, name: str = "random_model", seed: int = 42):
        self.name = name
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def featurise(self, inputs: Union[LabeledCandidates, List[Candidate]]) -> Any:
        pass

    def cleanup(self) -> None:
        pass

    def train(
        self,
        train_data: LabeledCandidates,
        val_data: LabeledCandidates,
        logger: Optional[Logger] = None,
    ) -> None:
        pass

    def predict(self, candidate_points: List[Candidate]) -> Predictions:
        labels = self.rng.randn(len(candidate_points))
        return Predictions(means=labels)

    def sample(self, *args: Any, **kwargs: Any) -> List[Candidate]:
        raise NotImplementedError("Sampling is not implemented for this model.")

    def get_training_summary_metrics(self) -> dict[str, Union[float, int, np.number]]:
        return {}
