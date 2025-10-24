from typing import List, Union, Tuple
from core.dataclasses import Candidate, LabeledCandidates
from core.models.base_model import BaseModel
from core.datasets.base_dataset import BaseDataset
import time
import numpy as np
from core.dataclasses.task_state import TaskState


class Oracle:
    """Oracle model is used to evaluate the candidates.
    For offline optimization tasks, the oracle is the ground truth dataset.
    For online optimization tasks, the oracle is a model.
    """

    def __init__(self, module: BaseModel | BaseDataset):
        self.module: BaseModel | BaseDataset = module
            
    def evaluate(self, candidates: List[Candidate], state: TaskState) -> Tuple[LabeledCandidates, TaskState]:
        """Evaluate the candidates."""
        t0 = time.perf_counter()
        if isinstance(self.module, BaseDataset):
            labeled_candidates = self.module.query(candidates)
        else:
            labeled_candidates = self.module.predict(candidates)
        t1 = time.perf_counter()
        state.round_metrics.update({"oracle_time": t1 - t0})
        return labeled_candidates, state

    def get_metrics(self) -> dict[str, Union[float, int, np.number]]:
        """Get the metrics for the oracle."""
        if hasattr(self.module, "get_metrics"):
            return self.module.get_metrics()
        return {}
