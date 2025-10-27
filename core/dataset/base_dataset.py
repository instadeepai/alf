from typing import Any, Dict, Tuple, List, Union
import abc
from core.dataclasses.labeled_candidates import Candidate, LabeledCandidates
import numpy as np
import copy


class BaseDataset(abc.ABC):
    """Base class for all datasets."""

    def __init__(self, name: str, modality: str, seed: int, split_config: dict[str, Any]):
        self.name = name
        self.modality = modality
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.validate_split_config(split_config)
        self.split_ratio = split_config["split_ratio"]
        self.split_type = split_config["split_type"]
        self.metadata: dict | None = None
        self._raw_dataset: LabeledCandidates | None = None
        self.splits: dict[str, LabeledCandidates] = {}

    @abc.abstractmethod
    def load_dataset(self) -> LabeledCandidates:
        """Load the dataset into HF datasets format."""
        pass

    def set_metadata(self) -> None:
        """Set the metadata for the dataset."""
        pass

    @property
    def train_dataset(self) -> LabeledCandidates:
        """Get the train dataset."""
        assert "train" in self.splits, "Dataset must be split before accessing train dataset"
        return self.splits["train"]
    
    @property
    def test_dataset(self) -> LabeledCandidates:
        """Get the test dataset."""
        assert "test" in self.splits, "Dataset must be split before accessing test dataset"
        return self.splits["test"]
    
    @property
    def validation_dataset(self) -> LabeledCandidates:
        """Get the validation dataset."""
        assert "validation" in self.splits, "Dataset must be split before accessing validation dataset"
        return self.splits["validation"]

    @property
    def candidate_pool(self) -> LabeledCandidates:
        """Get the candidate pool."""
        assert "candidate_pool" in self.splits, "Dataset must be split before accessing candidate pool"
        return self.splits["candidate_pool"]

    def __repr__(self) -> str:
        return f"Dataset(name={self.name}, modality={self.modality}, seed={self.seed}, train_size={len(self.train_dataset)}, validation_size={len(self.validation_dataset)}, test_size={len(self.test_dataset)}, candidate_pool_size={len(self.candidate_pool)})"
    
    def validate_split_config(self, split_config: dict[str, Any]) -> None:
        """Validate the split config for the dataset."""
        assert "split_ratio" in split_config, "Split ratio must be set"
        assert "split_type" in split_config, "Split type must be set"
        assert "train" and "test" and "validation" in split_config["split_ratio"], "Train, test, and validation splits ratio must be set"

    def _split_dataset(self) -> Tuple[Dict[str, LabeledCandidates], LabeledCandidates]:
        """Split dataset into train, test, validation splits."""
        assert self._raw_dataset is not None, "Dataset must be loaded before splitting"

        # Calculate split sizes
        train_plus_validation_size = int(len(self._raw_dataset) * self.split_ratio["train"])
        validation_size = int(train_plus_validation_size * self.split_ratio["validation"])
        train_size = train_plus_validation_size - validation_size
        test_size = int(len(self._raw_dataset) * self.split_ratio["test"])

        # Perform split based on type
        if self.split_type == "random":
            datasets_dict = self._split_random(train_size, validation_size, test_size)
        elif self.split_type == "low_vs_high":
            datasets_dict = self._split_low_vs_high(train_size, validation_size, test_size)
        else:
            raise ValueError(f"Invalid split type: {self.split_type}")

        self.init_candidate_pool = copy.deepcopy(datasets_dict["candidate_pool"])
        return datasets_dict

    def _split_random(self, train_size: int, validation_size: int, test_size: int) -> dict[str, LabeledCandidates]:
        """Split dataset randomly into train, validation, test, and candidate pool."""
        shuffled_candidates = self._raw_dataset.shuffle(seed=self.seed)
        
        start_idx = 0
        train = shuffled_candidates[start_idx:start_idx + train_size]
        start_idx += train_size
        
        validation = shuffled_candidates[start_idx:start_idx + validation_size]
        start_idx += validation_size
        
        test = shuffled_candidates[start_idx:start_idx + test_size]
        start_idx += test_size
        
        candidate_pool = shuffled_candidates[start_idx:]
        
        return {
            "train": train,
            "validation": validation,
            "test": test,
            "candidate_pool": candidate_pool
        }

    def _split_low_vs_high(self, train_size: int, validation_size: int, test_size: int) -> dict[str, LabeledCandidates]:
        """Split dataset so train/validation contain low-scoring candidates, test/pool contain high-scoring."""
        # Sort indices by label (highest to lowest)
        sorted_indices = self._raw_dataset.labels.argsort()[::-1]
        
        # Low-scoring candidates go to train/validation, high-scoring to test/pool
        train_plus_validation_size = train_size + validation_size
        low_scoring_indices = sorted_indices[:train_plus_validation_size]
        high_scoring_indices = sorted_indices[train_plus_validation_size:]
        
        # Randomly shuffle within each group
        shuffled_low = self._raw_dataset[self.rng.permutation(low_scoring_indices)]
        shuffled_high = self._raw_dataset[self.rng.permutation(high_scoring_indices)]
            
        return {
            "train": shuffled_low[:train_size],
            "validation": shuffled_low[train_size:],
            "test": shuffled_high[:test_size],
            "candidate_pool": shuffled_high[test_size:]
        }

    def setup(self) -> None:
        """Setup the dataset."""
        self._raw_dataset = self.load_dataset()
        self.splits = self._split_dataset()
        self.set_metadata()            

    def update_splits(self, acquired_candidates: LabeledCandidates) -> None:
        """Update the splits with the acquired candidates."""
        
        # Remove acquired candidates from candidate pool if they are in it
        self.splits["candidate_pool"].remove(acquired_candidates.candidates)
        
        # Split the acquired candidates into train and validation splits based on the split ratio
        num_val = int(len(acquired_candidates) * self.split_ratio["validation"])
        shuffled_acquired_candidates = acquired_candidates.shuffle(self.seed)
        self.splits["train"].append(shuffled_acquired_candidates[:-num_val])
        self.splits["validation"].append(shuffled_acquired_candidates[-num_val:])

    def save_splits(self, save_path: str, _verbose: bool = False) -> None:
        """Save the splits to a file."""
        pass

    def summarize(self) -> dict[str, Union[float, int, np.number]]:
        """Summarize the dataset."""
        summary = {}
        for key, split in self.splits.items():
            summary[f"num_{key}"] = len(split)
            summary[f"{key}_mean"] = np.mean(split.labels)
        return summary
    
    def query(self, candidates: List[Candidate]) -> LabeledCandidates:
        """Return the labels for the candidates."""
        indices = [self._raw_dataset.data.index(cand.data) for cand in candidates]
        labels = self._raw_dataset.labels[indices]
        return LabeledCandidates(candidates=candidates, labels=labels)

    def get_metrics(self) -> dict[str, Union[float, int, np.number]]:
        """Get the metrics for the dataset."""
        return {}
