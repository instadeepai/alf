import abc

import numpy as np

from alf.core.dataclasses import Candidate, LabeledCandidates, Predictions, TaskState


class AcquisitionFunction(abc.ABC):
    """Abstract base class for acquisition functions."""

    @abc.abstractmethod
    def _get_acquisition_values(
        self, predictions: Predictions, state: TaskState
    ) -> np.ndarray:
        """Compute acquisition values for candidates based on surrogate predictions.

        Args:
            predictions: Predictions from the surrogate model.
            state: Current task state.

        Returns:
            np.ndarray: Array of acquisition values, one per candidate.
        """
        pass

    def __call__(
        self,
        search_candidates: list[Candidate],
        state: TaskState,
    ) -> LabeledCandidates:
        """Compute acquisition values for candidates and return them as LabeledCandidates.

        Args:
            search_candidates: List of Candidate objects to score.
            state: Current task state containing the dataset and surrogate model.

        Returns:
            LabeledCandidates: Candidates paired with their acquisition values.
        """
        predictions: Predictions = state.surrogate.predict(search_candidates)
        acquisition_values = self._get_acquisition_values(predictions, state)
        return LabeledCandidates(search_candidates, acquisition_values)
