from typing import Callable, List
import numpy as np
from core.dataclasses import LabeledCandidates, Candidate, Predictions
from core.dataclasses.task_state import TaskState
import abc


class AcquisitionFunction(abc.ABC):
    """Abstract base class for acquisition functions."""

    @abc.abstractmethod
    def acquire(self, predictions: Predictions, state: TaskState) -> np.ndarray:
        """Scores the candidates based on the surrogate model's predictions."""
        pass

    def __call__(
        self,
        search_candidates: List[Candidate],
        state: TaskState,
    ) -> LabeledCandidates:
        """Returns the candidates with their acquisition values."""
        predictions: Predictions = state.surrogate.predict(search_candidates)
        acquisition_values = self.acquire(predictions, state)
        return LabeledCandidates(search_candidates, acquisition_values)