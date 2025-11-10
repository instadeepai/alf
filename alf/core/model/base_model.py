import abc
from typing import Any, Union

import numpy as np

from alf.core.dataclasses import Candidate, LabeledCandidates, Predictions
from alf.core.utils.logger import Logger


class BaseModel(abc.ABC):
    """Base class for all models.
    Several components of the framework can be treated as models, such as the surrogate, oracle,
    and the generator defined as model based search.
    """

    @abc.abstractmethod
    def featurise(self, inputs: Union[LabeledCandidates, list[Candidate]]) -> Any:
        """Featurise the inputs."""
        pass

    @abc.abstractmethod
    def train(
        self,
        train_data: LabeledCandidates,
        val_data: LabeledCandidates,
        logger: Logger | None = None,
    ) -> None:
        """Train the model."""
        pass

    @abc.abstractmethod
    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Predict the scores for the candidate points."""
        pass

    @abc.abstractmethod
    def sample(self, condition: Any | None = None) -> list[Candidate]:
        """Sample candidate points from the model."""
        pass

    def get_training_summary_metrics(self) -> dict[str, Union[float, int, np.number]]:
        """Get the training summary metrics."""
        return {}

    def cleanup(self) -> None:
        """Delete any temporary files, checkpoints etc that aren't being persisted."""
        pass
