from typing import List, Dict
from core.dataclasses import Candidate, TaskState
from core.model.base_model import BaseModel
from core.dataset.base_dataset import BaseDataset
import abc
import numpy as np
from core.optimizer.metrics import compute_recall, compute_regret

EMPTY_ARRAY: np.ndarray = np.array([])  


class BaseSearch(abc.ABC):
    """Base class for all types of search methods."""

    @abc.abstractmethod
    def __call__(self, task_state: TaskState, **kwargs) -> List[Candidate]:
        """Returns the search candidates."""
        pass

    def get_metrics(self, task_state: TaskState) -> Dict[str, float]:
        """Get the metrics for the search function."""
        return {}


class SearchProtocol:
    """Search protocol is used to define the search process."""

    @abc.abstractmethod
    def __call__(self, task_state: TaskState) -> List[Candidate]:
        """Apply the search protocol to return a pool of candidates."""
        pass


class GeneratorSearch(BaseSearch):
    """Search method based on a model generating samples."""
    
    def __init__(self, model: BaseModel):
        self.model: BaseModel = model
		
    def __call__(self, task_state: TaskState, **kwargs) -> List[Candidate]:
        """Sample candidates from the model to define the search pool."""
        return self.model.sample()
    
			
class DatasetSearch(BaseSearch):
    """Offline search method based on a dataset defining the search pool."""

    def __call__(self, task_state: TaskState, **kwargs) -> List[Candidate]:
        """Get the candidate pool from the dataset."""
        candidate_pool = task_state.dataset.candidate_pool
        return candidate_pool.candidates
    
    def get_metrics(self, task_state: TaskState) -> Dict[str, float]:
        """Return recall and regret metrics for the dataset search method."""
        init_candidate_pool = task_state.dataset.init_candidate_pool
        acquired_candidates = task_state.dataset.train_dataset
        recall_metrics = compute_recall(init_candidate_pool, acquired_candidates)
        regret_metrics = compute_regret(init_candidate_pool, acquired_candidates)
        return {**recall_metrics, **regret_metrics}
            

class ProtocolSearch(BaseSearch):
    """Search method based on a function/procedure to define the search pool."""

    def __init__(self, protocol: SearchProtocol):
        self.protocol: SearchProtocol = protocol
    
    def __call__(self, task_state: TaskState, **kwargs) -> List[Candidate]:
        """Apply the search protocol to return a pool of candidates."""
        return self.protocol(task_state, **kwargs)


class ModelProtocolSearch(BaseSearch):
    """Search method based on a model and a protocol to define the search pool."""

    def __init__(self, model: BaseModel, protocol: SearchProtocol):
        self.model: BaseModel = model
        self.protocol: SearchProtocol = protocol
    
    @abc.abstractmethod
    def __call__(self, task_state: TaskState, **kwargs) -> List[Candidate]:
        """Apply the model and the search protocol to return a pool of candidates."""
        pass