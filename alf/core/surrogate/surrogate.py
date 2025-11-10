from typing import Union

import numpy as np

from alf.core.dataclasses import Candidate, LabeledCandidates, Predictions
from alf.core.model.base_model import BaseModel
from alf.core.utils.logger import Logger


class Surrogate:
    """Surrogate model is fine-tuned during the active learning process on the acquired candidates.
    Any BaseModel child class can be used as a surrogate model.
    """

    def __init__(self, model: BaseModel) -> None:
        """Initialize the surrogate model.

        Args:
            model: A BaseModel instance to use as the surrogate model.
        """
        self.model: BaseModel = model

    def fit(
        self,
        train_data: LabeledCandidates,
        val_data: LabeledCandidates,
        logger: Logger | None = None,
    ) -> None:
        """Fit the surrogate model on training and validation data.

        Args:
            train_data: Labeled candidates for training.
            val_data: Labeled candidates for validation.
            logger: Optional logger for recording training metrics.
        """
        self.model.train(train_data, val_data, logger)

    def predict(self, candidates: list[Candidate]) -> Predictions:
        """Predict scores for the given candidates.

        Args:
            candidates: List of Candidate objects to make predictions for.

        Returns:
            Predictions: Predictions object containing means and optionally variances
                and empirical distributions.
        """
        return self.model.predict(candidates)

    def get_training_summary_metrics(self) -> dict[str, Union[float, int, np.number]]:
        """Get summary metrics from the most recent training run.

        Returns:
            dict[str, Union[float, int, np.number]]: Dictionary of metric names to values
                from the underlying model's training.
        """
        return self.model.get_training_summary_metrics()
