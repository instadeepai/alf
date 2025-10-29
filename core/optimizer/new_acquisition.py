from typing import Callable, List
import numpy as np
from core.dataclasses import LabeledCandidates, Candidate, Predictions
from core.dataclasses.task_state import TaskState
import abc


class AcquisitionFunction(abc.ABC):
    """Wraps an acquisition function and a surrogate model to select the next candidates to evaluate."""

    # def __init__(self, name: str, surrogate: Surrogate) -> None:
    #     """Initialize the acquisition function.
        
    #     Args:
    #         name: The name of the acquisition function.
    #         surrogate: The surrogate model.
    #     """
    #     assert name in ACQ_DICT, f"Acquisition function {name} not found"
    #     self.acq_fn: Callable = ACQ_DICT[name]
    #     self.surrogate = surrogate

    @abc.abstractmethod
    def acquire(self, predictions: Predictions, state: TaskState) -> np.ndarray:
        """Acquire the candidates."""
        pass


    def __call__(
        self,
        search_candidates: List[Candidate],
        # best_f: np.ndarray = EMPTY_ARRAY,
        state,
    ) -> LabeledCandidates:
        """Returns the candidates with their acquisition values."""
        predictions: Predictions = state.surrogate.predict(search_candidates)
        acquisition_values = self.acquire(predictions, state)
        return LabeledCandidates(search_candidates, acquisition_values)