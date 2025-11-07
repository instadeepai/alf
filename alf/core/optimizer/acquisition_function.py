import abc
from typing import List

import numpy as np

from alf.core.dataclasses import Candidate, LabeledCandidates, Predictions
from alf.core.dataclasses.task_state import TaskState


class AcquisitionFunction(abc.ABC):
    """Abstract base class for acquisition functions."""

    @abc.abstractmethod
    def _get_acquisition_values(
        self, predictions: Predictions, state: TaskState
    ) -> np.ndarray:
        """Computes acquisition values for candidates based on surrogate predictions."""
        pass

    def __call__(
        self,
        search_candidates: List[Candidate],
        state: TaskState,
    ) -> LabeledCandidates:
        """Returns the candidates with their acquisition values."""
        predictions: Predictions = state.surrogate.predict(search_candidates)
        acquisition_values = self._get_acquisition_values(predictions, state)
        return LabeledCandidates(search_candidates, acquisition_values)
