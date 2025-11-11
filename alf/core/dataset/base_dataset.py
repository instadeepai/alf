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
from typing import Any, Union

import numpy as np

from alf.core.dataclasses.labeled_candidates import Candidate, LabeledCandidates
from alf.core.dataset.splitting_utils import split_dataset
from alf.core.utils.io import input_handler

log = logging.getLogger("alf-core")


class BaseDataset(abc.ABC):
    """Base class for all datasets."""

    def __init__(self, name: str, modality: str, seed: int, split_config: dict[str, Any]) -> None:
        """Initialize the base dataset.

        Args:
            name: Name identifier for the dataset.
            modality: Data modality (e.g., "sequence", "graph", "image").
            seed: Random seed for reproducibility.
            split_config: Dictionary containing:
                - "split_ratio": dict with "train", "validation", "test", and
                  (optionally) "candidate_pool" ratios
                - "split_type": Type of split ("random" or "low_vs_high")
        """
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
        """Load the raw dataset.

        This method must be implemented by subclasses to load data from their
        specific source.

        Returns:
            LabeledCandidates: The loaded dataset with candidates and labels.
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
            LabeledCandidates: Training dataset.

        Raises:
            AssertionError: If dataset hasn't been split yet.
        """
        assert "train" in self.splits, "Dataset must be split before accessing train dataset"
        return self.splits["train"]

    @property
    def test_dataset(self) -> LabeledCandidates:
        """Get the test dataset split.

        Returns:
            LabeledCandidates: Test dataset.

        Raises:
            AssertionError: If dataset hasn't been split yet.
        """
        assert "test" in self.splits, "Dataset must be split before accessing test dataset"
        return self.splits["test"]

    @property
    def validation_dataset(self) -> LabeledCandidates:
        """Get the validation dataset split.

        Returns:
            LabeledCandidates: Validation dataset.

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
            LabeledCandidates: Candidate pool available for acquisition.

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
            str: String showing dataset name, modality, seed, and split sizes.
        """
        return (
            f"Dataset(name={self.name}, modality={self.modality}, seed={self.seed}, "
            f"train_size={len(self.train_dataset)}, "
            f"validation_size={len(self.validation_dataset)}, "
            f"test_size={len(self.test_dataset)}, "
            f"candidate_pool_size={len(self.candidate_pool)})"
        )

    def validate_split_config(self, split_config: dict[str, Any]) -> None:
        """Validate the split config for the dataset.

        Args:
            split_config: Split config of type dict to validate.

        Raises:
            AssertionError: If split_ratio or split_type are missing, or if
                required split ratios are not present.
        """
        assert "split_ratio" in split_config, "Split ratio must be set"
        assert "split_type" in split_config, "Split type must be set"
        assert "train" and "test" and "validation" in split_config["split_ratio"], (
            "Train, test, and validation splits ratio must be set"
        )

    def _split_dataset(self) -> dict[str, LabeledCandidates]:
        """Split the raw dataset into train, validation, test, and candidate pool.

        Returns:
            dict[str, LabeledCandidates]: Dictionary with keys "train", "validation",
                "test", and "candidate_pool", each containing a LabeledCandidates object.

        Raises:
            AssertionError: If dataset hasn't been loaded yet.
        """
        assert self._raw_dataset is not None, "Dataset must be loaded before splitting"

        # Calculate split sizes
        train_plus_validation_size = int(len(self._raw_dataset) * self.split_ratio["train"])
        validation_size = int(train_plus_validation_size * self.split_ratio["validation"])
        train_size = train_plus_validation_size - validation_size
        test_size = int(len(self._raw_dataset) * self.split_ratio["test"])

        # Perform split based on type
        datasets_dict = split_dataset(
            self.split_type,
            self._raw_dataset,
            train_size,
            validation_size,
            test_size,
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
        num_val = int(len(acquired_candidates) * self.split_ratio["validation"])
        shuffled_acquired_candidates = acquired_candidates.shuffle(self.seed)
        self.splits["train"].append(shuffled_acquired_candidates[:-num_val])
        self.splits["validation"].append(shuffled_acquired_candidates[-num_val:])

    def save_splits(self, output_dir: str, _verbose: bool = False) -> None:
        """Save dataset splits to CSV files in the output directory.

        Saves train, validation, and test splits (but not candidate_pool) as
        separate CSV files.

        Args:
            output_dir: Directory path where CSV files will be saved.
            _verbose: If True, log messages when saving each split.
        """
        for key, split in self.splits.items():
            if key not in ["train", "validation", "test"]:
                continue
            if _verbose:
                log.info(f"Saving {key} split to {output_dir}")
            input_handler.save_csv(
                os.path.join(output_dir, f"{key}.csv"),
                split.to_dataframe(),
            )

    def query(self, candidates: list[Candidate]) -> LabeledCandidates:
        """Query labels for the given candidates from the raw dataset.

        Args:
            candidates: List of Candidate objects to query.

        Returns:
            LabeledCandidates: Candidates paired with their labels from the dataset.

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
            dict[str, Union[float, int, np.number]]: Dictionary containing:
                - "num_{split}": Number of samples in each split
                - "{split}_mean": Mean label value for each split
        """
        metrics: dict[str, Union[float, int, np.number]] = {}
        for key, split in self.splits.items():
            metrics[f"num_{key}"] = len(split)
            metrics[f"{key}_mean"] = np.mean(split.labels)
        return metrics
