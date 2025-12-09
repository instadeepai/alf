# Copyright 2023 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import abc
import copy
import logging
import os
from math import floor
from pathlib import Path
from typing import Any, Union

import numpy as np
from alf_core.dataclasses.candidate import Modality
from alf_core.dataclasses.labeled_candidates import Candidate, LabeledCandidates
from alf_core.dataset.splitting_utils import split_dataset

logger = logging.getLogger("alf-core")


class BaseDataset(abc.ABC):
    """Base class for all datasets."""

    def __init__(self, name: str, modality: str, seed: int, split_config: dict[str, Any]) -> None:
        """Initialize the base dataset.

        Args:
            name: Name identifier for the dataset.
            modality: Data modality (e.g., "sequence", "graph", "image").
            seed: Random seed for reproducibility.
            split_config: Dictionary containing:
                - "split_ratio": dict with "train", "validation", "test" ratios
                - "split_type": Type of split ("random" or "low_vs_high")
                - "max_candidate_pool": optional, maximum size of the candidate pool (int)
        """
        self.name = name
        # Validate and convert modality string to Modality enum
        valid_modalities = [m.value for m in Modality]
        assert modality in valid_modalities, (
            f"Invalid modality: {modality}. Must be one of {valid_modalities}"
        )
        self.modality = Modality(modality)
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self._validate_and_process_split_config(split_config)
        self.metadata: dict | None = None
        self._raw_dataset: LabeledCandidates | None = None
        self.splits: dict[str, LabeledCandidates] = {}

    @abc.abstractmethod
    def load_dataset(self) -> LabeledCandidates:
        """Load the raw dataset.

        This method must be implemented by subclasses to load data from their
        specific source.

        Returns:
            The loaded dataset with candidates and labels.
        """
        pass

    def set_metadata(self) -> None:
        """Set metadata for the dataset.

        Subclasses can override this method to compute and store dataset-specific
        metadata. Called automatically during setup().
        """
        pass

    @property
    def train_dataset(self) -> LabeledCandidates:
        """Get the training dataset split.

        Returns:
            Training dataset.

        Raises:
            AssertionError: If dataset hasn't been split yet.
        """
        assert "train" in self.splits, "Dataset must be split before accessing train dataset"
        return self.splits["train"]

    @property
    def test_dataset(self) -> LabeledCandidates:
        """Get the test dataset split.

        Returns:
            Test dataset.

        Raises:
            AssertionError: If dataset hasn't been split yet.
        """
        assert "test" in self.splits, "Dataset must be split before accessing test dataset"
        return self.splits["test"]

    @property
    def validation_dataset(self) -> LabeledCandidates:
        """Get the validation dataset split.

        Returns:
            Validation dataset.

        Raises:
            AssertionError: If dataset hasn't been split yet.
        """
        assert "validation" in self.splits, (
            "Dataset must be split before accessing validation dataset"
        )
        return self.splits["validation"]

    @property
    def candidate_pool(self) -> LabeledCandidates:
        """Get the candidate pool split.

        Returns:
            Candidate pool available for acquisition.

        Raises:
            AssertionError: If dataset hasn't been split yet.
        """
        assert "candidate_pool" in self.splits, (
            "Dataset must be split before accessing candidate pool"
        )
        return self.splits["candidate_pool"]

    def __repr__(self) -> str:
        """Return a string representation of the dataset.

        Returns:
            String showing dataset name, modality, seed, and split sizes.
        """
        return (
            f"Dataset(name={self.name}, modality={self.modality}, seed={self.seed}, "
            f"train_size={len(self.train_dataset)}, "
            f"validation_size={len(self.validation_dataset)}, "
            f"test_size={len(self.test_dataset)}, "
            f"candidate_pool_size={len(self.candidate_pool)})"
        )

    def _validate_and_process_split_config(self, split_config: dict[str, Any]) -> None:
        """Validate and process the split config for the dataset.

        This method validates the split configuration and sets default values
        where appropriate.
        The split_ratio keys must be "train", "test", "validation_frac".
        validation_frac is the fraction of the train set that is held out in the validation set.
        Optionally, the split_config can contain "max_candidate_pool" (int),
        the maximum size of the candidate pool.

        Args:
            split_config: Dictionary containing the split configuration.

        Raises:
            AssertionError: If the split configuration is invalid.
        """
        # Check required top-level keys
        assert "split_ratio" in split_config, "split_config must contain 'split_ratio'"
        assert "split_type" in split_config, "split_config must contain 'split_type'"

        self.split_type = split_config["split_type"]
        self.split_ratio = split_config["split_ratio"]

        # Check required split_ratio keys present and between 0 and 1
        required_keys = ["train", "test", "validation_frac"]
        for key in required_keys:
            assert key in self.split_ratio, f"split_ratio must contain '{key}'"
            assert 0 <= self.split_ratio[key] <= 1, f"{key} ratio must be between 0 and 1"

        assert self.split_ratio["train"] + self.split_ratio["test"] <= 1, (
            "train + test ratios must be <= 1"
        )
        self.max_candidate_pool = split_config.get("max_candidate_pool", None)

    def _split_dataset(self) -> dict[str, LabeledCandidates]:
        """Split the raw dataset into train, validation, test, and candidate pool.

        Note, that the candidate pool size is everything that is not in the train, validation,
        or test sets and is capped at max_candidate_pool (int). The validation_frac is the
        fraction of the train set that is held out in the validation set.

        Returns:
            Dictionary with keys "train", "validation", "test", and "candidate_pool",
            each containing a LabeledCandidates object.

        Raises:
            AssertionError: If dataset hasn't been loaded yet.
        """
        assert self._raw_dataset is not None, "Dataset must be loaded before splitting"

        # Calculate split sizes
        dataset_size = len(self._raw_dataset)
        train_plus_validation_size = floor(dataset_size * self.split_ratio["train"])
        validation_size = floor(train_plus_validation_size * self.split_ratio["validation_frac"])
        train_size = train_plus_validation_size - validation_size
        test_size = floor(dataset_size * self.split_ratio["test"])
        candidate_pool_size = dataset_size - train_plus_validation_size - test_size
        if self.max_candidate_pool is not None:
            candidate_pool_size = min(self.max_candidate_pool, candidate_pool_size)

        # Perform split based on type
        datasets_dict = split_dataset(
            self.split_type,
            self._raw_dataset,
            train_size,
            validation_size,
            test_size,
            candidate_pool_size,
            self.seed,
        )
        self.init_candidate_pool = copy.deepcopy(datasets_dict["candidate_pool"])
        return datasets_dict

    def setup(self) -> None:
        """Setup the dataset by loading and splitting it.

        Loads the raw dataset, splits it according to the split configuration,
        and sets metadata. This must be called before accessing dataset splits.
        """
        self._raw_dataset = self.load_dataset()
        self.splits = self._split_dataset()
        self.set_metadata()

    def update_splits(self, acquired_candidates: LabeledCandidates) -> None:
        """Update dataset splits with newly acquired candidates.

        Removes acquired candidates from the candidate pool and distributes them
        between train and validation splits according to the split ratio.

        Args:
            acquired_candidates: Newly acquired candidates with labels.
        """
        # Remove acquired candidates from candidate pool if they are in it
        self.splits["candidate_pool"].remove(acquired_candidates.candidates)

        # Split the acquired candidates into train and validation splits based on the split ratio
        num_val = floor(len(acquired_candidates) * self.split_ratio["validation_frac"])
        shuffled_acquired_candidates = acquired_candidates.shuffle(self.seed)
        # Add num_val candidates to validation split and the rest to train split
        self.splits["train"].append(LabeledCandidates(*shuffled_acquired_candidates[:-num_val]))
        self.splits["validation"].append(
            LabeledCandidates(*shuffled_acquired_candidates[-num_val:])
        )

    def query(self, candidates: list[Candidate]) -> LabeledCandidates:
        """Query labels for the given candidates from the raw dataset.

        Args:
            candidates: List of Candidate objects to query.

        Returns:
            Candidates paired with their labels from the dataset.

        Raises:
            AssertionError: If dataset hasn't been loaded yet.
            ValueError: If any candidate's data is not found in the dataset.
        """
        assert self._raw_dataset is not None, "Dataset must be loaded before querying"
        indices = [self._raw_dataset.data.index(cand.data) for cand in candidates]
        labels = self._raw_dataset.labels[indices]
        return LabeledCandidates(candidates=candidates, labels=labels)

    def get_metrics(self) -> dict[str, Union[float, int, np.number]]:
        """Get summary metrics for all dataset splits.

        Returns:
            Dictionary containing:
            - "num_{split}": Number of samples in each split
            - "{split}_mean": Mean label value for each split
        """
        metrics: dict[str, Union[float, int, np.number]] = {}
        for key, split in self.splits.items():
            metrics[f"num_{key}"] = len(split)
            metrics[f"{key}_mean"] = np.mean(split.labels)
        return metrics

    def save_splits(self, output_path: str | os.PathLike) -> None:
        """Save the dataset splits to a file.

        Args:
            output_path: Path to the directory to save the dataset splits to
        """
        data_splits_path = Path(output_path) / "data_splits"
        data_splits_path.mkdir(parents=True, exist_ok=True)
        for key, data_split in self.splits.items():
            data_split.to_dataframe().to_csv(data_splits_path / f"{key}.csv", index=False)
