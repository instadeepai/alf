"""Dummy components for testing purposes, including a dummy dataset, model, and acquisition function."""
from typing import Union, List, Any, Optional
import numpy as np

from alf.core.model.base_model import BaseModel
from alf.core.dataset.base_dataset import BaseDataset
from alf.core.optimizer.acquisition_function import AcquisitionFunction
from alf.core.dataclasses import Candidate, LabeledCandidates, Predictions
from alf.core.utils.logger import Logger
from alf.core.dataclasses import TaskState

class DummyDataset(BaseDataset):
    """Dummy dataset that generates random data for testing."""
    
    def __init__(
        self,
        name: str = "dummy",
        modality: str = "sequence",
        seed: int = 42,
        split_config: dict[str, Any] = None,
        num_samples: int = 1000,
    ):
        """
        Initialize a dummy dataset.
        
        Args:
            name: Name of the dataset
            modality: Modality of the data (e.g., "sequence")
            seed: Random seed for reproducibility
            split_config: Split configuration dictionary
            num_samples: Number of dummy samples to generate
        """
        # Default split config if none provided
        if split_config is None:
            split_config = {
                "split_ratio": {"train": 0.48, "validation": 0.12, "test": 0.2, "candidate_pool": 0.2},
                "split_type": "random"
            }
        super().__init__(name, modality, seed, split_config)
        self.num_samples = num_samples
        self.setup()
    
    def load_dataset(self) -> LabeledCandidates:
        """
        Generate dummy dataset with random sequences and labels.
        
        Returns:
            LabeledCandidates with dummy data
        """
        # Generate dummy sequences (e.g., random strings)
        candidates = []
        labels = []
        
        for i in range(self.num_samples):
            # Generate a dummy sequence (e.g., random string of length 10)
            dummy_sequence = "".join(self.rng.choice(list("ACGT"), size=10))
            
            # Generate a dummy label (random float between 0 and 10)
            dummy_label = float(self.rng.uniform(0, 10))
            
            candidates.append(Candidate(data=dummy_sequence, modality=self.modality))
            labels.append(dummy_label)
        
        return LabeledCandidates(
            candidates=candidates,
            labels=np.array(labels)
        )


class DummyModel(BaseModel):
    """Dummy model that generates random predictions for testing."""
    
    def __init__(self, name: str = "dummy_model", seed: int = 42):
        """
        Initialize a dummy model.
        
        Args:
            name: Name of the model
            seed: Random seed for reproducibility
        """
        self.name = name
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def predict(self, candidate_points: List[Candidate]) -> Predictions:
        """Generate random predictions for the candidate points."""
        labels = self.rng.randn(len(candidate_points))
        return Predictions(means=labels)

    def featurise(self, inputs: Union[LabeledCandidates, List[Candidate]]) -> Any:
        """Dummy model does not perform featurisation."""
        pass

    def train(self, train_data: LabeledCandidates, val_data: LabeledCandidates, logger: Optional[Logger] = None) -> None:
        """Dummy model does not perform any actual training but updates the random seed."""
        self.rng = np.random.RandomState(self.seed + 1)
   
    def sample(self, *args: Any, **kwargs: Any) -> List[Candidate]:
        """Dummy model does not perform sampling."""
        pass

class DummyAcquisitionFunction(AcquisitionFunction):
    """Dummy acquisition function that generates random acquisition values for testing."""
    
    def __init__(self, seed: int = 42):
        """Initialize the dummy acquisition function."""
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def _get_acquisition_values(self, predictions: Predictions, state: TaskState) -> np.ndarray:
        """Generate random acquisition values for the predictions."""
        return self.rng.randn(len(predictions))
