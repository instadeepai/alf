import time
from typing import List, Tuple, Union

import numpy as np

from alf.core.dataclasses import Candidate, LabeledCandidates, TaskState
from alf.core.model.base_model import BaseModel
from alf.core.dataset.base_dataset import BaseDataset


class Oracle:
    """Oracle model is used to evaluate the candidates.
    For offline optimization tasks, the oracle is the dataset.
    For online optimization tasks, the oracle is a model.
    """

    def __init__(self, module: BaseModel | BaseDataset):
        self.module: BaseModel | BaseDataset = module

    def evaluate(
        self, candidates: List[Candidate], state: TaskState
    ) -> Tuple[LabeledCandidates, TaskState]:
        """Evaluate the candidates."""
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
        """Get the metrics for the oracle."""
        if hasattr(self.module, "get_metrics"):
            return self.module.get_metrics()
        return {}
