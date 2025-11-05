from typing import Any, Dict, Tuple, List, Union
import abc
from alf.core.dataclasses.labeled_candidates import Candidate, LabeledCandidates
from alf.core.dataset.splitting_utils import split_dataset
import numpy as np
import copy
import logging
from alf.core.utils.io import input_handler
import os

logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]")
log = logging.getLogger("rich")


class BaseDataset(abc.ABC):
    """Base class for all datasets."""

    def __init__(self, name: str, modality: str, seed: int, split_config: dict[str, Any]):
        self.name = name
        self.modality = modality
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._validate_split_config(split_config)
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
        """Return a string representation of the dataset."""
        return f"Dataset(name={self.name}, modality={self.modality}, seed={self.seed}, train_size={len(self.train_dataset)}, validation_size={len(self.validation_dataset)}, test_size={len(self.test_dataset)}, candidate_pool_size={len(self.candidate_pool)})"
    
    def _validate_split_config(self, split_config: dict[str, Any]) -> None:
        """Validate the split config for the dataset."""
        assert "split_ratio" in split_config, "Split ratio must be set"
        assert "split_type" in split_config, "Split type must be set"
        assert "train" and "test" and "validation" in split_config["split_ratio"], "Train, test, and validation splits ratio must be set"
        self.split_type = split_config["split_type"]
        self.split_ratio = split_config["split_ratio"]

        train_split_ratio = self.split_ratio["train"]
        validation_split_ratio = self.split_ratio["validation"]
        test_split_ratio = self.split_ratio["test"]
        if "candidate_pool" in self.split_ratio:
            candidate_pool_split_ratio = self.split_ratio["candidate_pool"]
            assert train_split_ratio + validation_split_ratio + test_split_ratio + candidate_pool_split_ratio <= 1, "Split ratios must sum to less than or equal to 1"
        else:
            assert train_split_ratio + validation_split_ratio + test_split_ratio <= 1, "Split ratios must sum to less than or equal to 1"

    def _split_dataset(self) -> Tuple[Dict[str, LabeledCandidates], LabeledCandidates]:
        """Split dataset into train, test, validation splits."""
        assert self._raw_dataset is not None, "Dataset must be loaded before splitting"

        # Calculate split sizes
        train_size = int(len(self._raw_dataset) * self.split_ratio["train"])
        validation_size = int(len(self._raw_dataset) * self.split_ratio["validation"])
        test_size = int(len(self._raw_dataset) * self.split_ratio["test"])
        if "candidate_pool" in self.split_ratio:
            candidate_pool_size = int(len(self._raw_dataset) * self.split_ratio["candidate_pool"])
        else:
            log.warning("Candidate pool ratio not set, using remaining dataset size")
            candidate_pool_size = len(self._raw_dataset) - train_size - validation_size - test_size

        # Perform split based on type
        datasets_dict = split_dataset(self.split_type, self._raw_dataset, train_size, validation_size, test_size, candidate_pool_size, self.seed)
        self.init_candidate_pool = copy.deepcopy(datasets_dict["candidate_pool"])
        return datasets_dict

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

    def save_splits(self, output_dir: str, _verbose: bool = False) -> None:
        """Save the datasplits to the output directory."""
        for key, split in self.splits.items():
            if key not in ["train", "validation", "test"]:
                continue
            if _verbose:
                log.info(f"Saving {key} split to {output_dir}")
            input_handler.save_csv(
                os.path.join(output_dir, f"{key}.csv"),
                split.to_dataframe(),
            )
    
    def query(self, candidates: List[Candidate]) -> LabeledCandidates:
        """Return the labels for the candidates."""
        indices = [self._raw_dataset.data.index(cand.data) for cand in candidates]
        labels = self._raw_dataset.labels[indices]
        return LabeledCandidates(candidates=candidates, labels=labels)

    def get_metrics(self) -> dict[str, Union[float, int, np.number]]:
        """Get the metrics for the dataset."""
        metrics = {}
        for key, split in self.splits.items():
            metrics[f"num_{key}"] = len(split)
            metrics[f"{key}_mean"] = np.mean(split.labels)
        return metrics
