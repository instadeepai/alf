import time
from typing import Tuple, Union

import numpy as np

from alf.core.dataclasses import Candidate, LabeledCandidates, TaskState
from alf.core.dataset.base_dataset import BaseDataset
from alf.core.model.base_model import BaseModel


class Oracle:
    """Oracle model is used to evaluate the candidates.
    For offline optimization tasks, the oracle is the dataset.
    For online optimization tasks, the oracle is a model.
    """

    def __init__(self, module: BaseModel | BaseDataset) -> None:
        """Initialize the oracle with a model or dataset.

        Args:
            module: Either a BaseModel (for online evaluation) or BaseDataset
                (for offline evaluation from a dataset).
        """
        self.module: BaseModel | BaseDataset = module

    def evaluate(
        self, candidates: list[Candidate], state: TaskState
    ) -> Tuple[LabeledCandidates, TaskState]:
        """Evaluate candidates and return their labels.

        Args:
            candidates: List of Candidate objects to evaluate.
            state: Current task state (updated with evaluation time).

        Returns:
            Tuple[LabeledCandidates, TaskState]: A tuple containing:
                - LabeledCandidates: Candidates paired with their evaluated labels
                - TaskState: Updated state with oracle_time metric
        """
        t0 = time.perf_counter()
        if isinstance(self.module, BaseDataset):
            evaluated_candidates = self.module.query(candidates)
        else:
            evaluated_candidates = LabeledCandidates(
                candidates=candidates, labels=self.module.predict(candidates).means
            )
        t1 = time.perf_counter()
        state.round_metrics.update({"oracle_time": t1 - t0})
        return evaluated_candidates, state

    def get_metrics(self) -> dict[str, Union[float, int, np.number]]:
        """Get metrics from the underlying module if available.

        Returns:
            dict[str, Union[float, int, np.number]]: Dictionary of metric names to values.
                Returns empty dict if the module doesn't provide metrics.
        """
        if hasattr(self.module, "get_metrics"):
            return self.module.get_metrics()
        return {}
