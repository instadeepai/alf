from typing import List, Union
from core.dataclasses import Candidate, LabeledCandidates, Predictions
from core.model.base_model import BaseModel
from core.utils.logger import Logger
import numpy as np


class Surrogate:
    """Surrogate model is fine-tuned during the active learning process on the acquired candidates.
    Any BaseModel child class can be used as a surrogate model.
    """

    def __init__(self, model: BaseModel):
        """Initialize the surrogate model."""
        self.model: BaseModel = model
    
    def fit(self, train_data: LabeledCandidates, val_data: LabeledCandidates, logger: Logger) -> None:
        """Fit the surrogate model on the batch of candidates."""
        self.model.train(train_data, val_data, logger)
    
    def predict(self, candidates: List[Candidate]) -> Predictions:
        """Predict the scores for the candidates."""
        return self.model.predict(candidates)

    def get_training_summary_metrics(self) -> dict[str, Union[float, int, np.number]]:
        """Get the training summary metrics."""
        return self.model.get_training_summary_metrics()