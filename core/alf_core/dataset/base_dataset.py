# Copyright 2026 InstaDeep Ltd. All rights reserved.
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
from typing import Annotated, Self, Union

import numpy as np
from alf_core.dataclasses.candidate import Modality
from alf_core.dataclasses.labelled_candidates import Candidate, LabelledCandidates
from alf_core.dataset.splitting_utils import SplitType, split_dataset
from alf_core.utils.enums import ProblemType
from pydantic import BaseModel, Field, model_validator

FloatBetweenZeroAndOne = Annotated[float, Field(ge=0, le=1)]

logger = logging.getLogger("alf-core")


class BaseDatasetConfig(BaseModel):
    """Configuration for BaseDataset.

    Attributes:
        name: Name identifier for the dataset.
        modality: Data modality (validated against Modality enum).
        seed: Random seed for reproducibility.
        train_ratio: Fraction of data for training (0-1).
        validation_frac: Fraction of training data held out for validation (0-1).
        test_ratio: Fraction of data for testing (0-1).
        split_type: Type of split.
        max_candidate_pool: Optional maximum size of the candidate pool.
    """

    name: str
    modality: Modality
    seed: int
    train_ratio: FloatBetweenZeroAndOne
    validation_frac: FloatBetweenZeroAndOne
    test_ratio: FloatBetweenZeroAndOne
    split_type: SplitType = "random"
    problem_type: ProblemType
    max_candidate_pool: int | None = None

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """Validate dataset configuration.

        Returns:
            The validated configuration instance.

        Raises:
            ValueError: If train_ratio + test_ratio exceeds 1.
            ValueError: If split_type is "stratified" with a REGRESSION problem_type.
        """
        if self.train_ratio + self.test_ratio > 1:
            raise ValueError("train_ratio + test_ratio must be <= 1")
        if self.split_type == "stratified" and self.problem_type == ProblemType.REGRESSION:
            raise ValueError(
                "split_type='stratified' requires a classification problem_type "
                "(BINARY or MULTICLASS), not REGRESSION."
            )
        return self


class BaseDataset(abc.ABC):
    """Base class for all datasets."""

    def __init__(self, config: BaseDatasetConfig) -> None:
        """Initialize the base dataset.

        Args:
            config: Configuration for the dataset.
        """
        self.config = config
        self.modality = config.modality
        self.rng = np.random.RandomState(config.seed)
        self.split_ratio = {
            "train": config.train_ratio,
            "validation_frac": config.validation_frac,
            "test": config.test_ratio,
        }
        self.metadata: dict | None = None
        self._raw_dataset: LabelledCandidates | None = None
        self.splits: dict[str, LabelledCandidates] = {}

    @abc.abstractmethod
    def load_dataset(self) -> LabelledCandidates:
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
    def train_dataset(self) -> LabelledCandidates:
        """Get the training dataset split.

        Returns:
            Training dataset.

        Raises:
            AssertionError: If dataset hasn't been split yet.
        """
        assert "train" in self.splits, (
            "Dataset must be split before accessing train dataset; "
            "train split has not been created yet — "
            "call dataset.setup() before accessing train_dataset"
        )
        return self.splits["train"]

    @property
    def test_dataset(self) -> LabelledCandidates:
        """Get the test dataset split.

        Returns:
            Test dataset.

        Raises:
            AssertionError: If dataset hasn't been split yet.
        """
        assert "test" in self.splits, (
            "Dataset must be split before accessing test dataset; "
            "test split has not been created yet — "
            "call dataset.setup() before accessing test_dataset"
        )
        return self.splits["test"]

    @property
    def validation_dataset(self) -> LabelledCandidates:
        """Get the validation dataset split.

        Returns:
            Validation dataset.

        Raises:
            AssertionError: If dataset hasn't been split yet.
        """
        assert "validation" in self.splits, (
            "Dataset must be split before accessing validation dataset; "
            "validation split has not been created yet — "
            "call dataset.setup() before accessing validation_dataset"
        )
        return self.splits["validation"]

    @property
    def candidate_pool(self) -> LabelledCandidates:
        """Get the candidate pool split.

        Returns:
            Candidate pool available for acquisition.

        Raises:
            AssertionError: If dataset hasn't been split yet.
        """
        assert "candidate_pool" in self.splits, (
            "Dataset must be split before accessing candidate pool; "
            "candidate_pool split has not been created yet — "
            "call dataset.setup() before accessing candidate_pool"
        )
        return self.splits["candidate_pool"]

    @property
    def raw_dataset(self) -> LabelledCandidates:
        """Get the full labelled dataset before splitting.

        Returns:
            The complete labelled dataset loaded by `load_dataset`.

        Raises:
            AssertionError: If the dataset has not been set up yet.
        """
        assert self._raw_dataset is not None, (
            "_raw_dataset is None — call dataset.setup() before accessing raw_dataset"
        )
        return self._raw_dataset

    def __repr__(self) -> str:
        """Return a string representation of the dataset.

        Returns:
            String showing dataset name, modality, seed, and split sizes.
        """
        return (
            f"Dataset(name={self.config.name}, modality={self.modality}, "
            f"seed={self.config.seed}, "
            f"train_size={len(self.train_dataset)}, "
            f"validation_size={len(self.validation_dataset)}, "
            f"test_size={len(self.test_dataset)}, "
            f"candidate_pool_size={len(self.candidate_pool)})"
        )

    def _split_dataset(self) -> dict[str, LabelledCandidates]:
        """Split the raw dataset into train, validation, test, and candidate pool.

        Note, that the candidate pool size is everything that is not in the train, validation,
        or test sets and is capped at max_candidate_pool (int). The validation_frac is the
        fraction of the train set that is held out in the validation set.

        Returns:
            Dictionary with keys "train", "validation", "test", and "candidate_pool",
            each containing a LabelledCandidates object.

        Raises:
            AssertionError: If dataset hasn't been loaded yet.
        """
        assert self._raw_dataset is not None, (
            "Dataset must be loaded before splitting; "
            "_raw_dataset is None — load_dataset() must be called and return a "
            "LabelledCandidates before _split_dataset() is called"
        )

        # Calculate split sizes
        dataset_size = len(self._raw_dataset)
        train_plus_validation_size = floor(dataset_size * self.split_ratio["train"])
        validation_size = floor(train_plus_validation_size * self.split_ratio["validation_frac"])
        train_size = train_plus_validation_size - validation_size
        test_size = floor(dataset_size * self.split_ratio["test"])
        candidate_pool_size = dataset_size - train_plus_validation_size - test_size
        if self.config.max_candidate_pool is not None:
            candidate_pool_size = min(self.config.max_candidate_pool, candidate_pool_size)

        # Perform split based on type
        datasets_dict = split_dataset(
            self.config.split_type,
            self._raw_dataset,
            train_size,
            validation_size,
            test_size,
            candidate_pool_size,
            self.config.seed,
        )
        self.init_candidate_pool = copy.deepcopy(datasets_dict["candidate_pool"])
        return datasets_dict

    def setup(self) -> None:
        """Setup the dataset by loading and splitting it.

        Loads the raw dataset, splits it according to the split configuration,
        and sets metadata. This must be called before accessing dataset splits.
        """
        self._raw_dataset = self.load_dataset()
        self.num_classes = self.determine_num_classes()
        self.splits = self._split_dataset()
        self.set_metadata()

    def update_splits(self, acquired_candidates: LabelledCandidates) -> None:
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
        num_train = len(acquired_candidates) - num_val
        shuffled_acquired_candidates = acquired_candidates.shuffle(self.config.seed)
        # Add num_val candidates to validation split and the rest to train split
        self.splits["train"].append(LabelledCandidates(*shuffled_acquired_candidates[:num_train]))
        self.splits["validation"].append(
            LabelledCandidates(*shuffled_acquired_candidates[num_train:])
        )

    def query(self, candidates: list[Candidate]) -> LabelledCandidates:
        """Query labels for the given candidates from the raw dataset.

        Args:
            candidates: List of Candidate objects to query.

        Returns:
            Candidates paired with their labels from the dataset.

        Raises:
            AssertionError: If dataset hasn't been loaded yet.
            ValueError: If any candidate's data is not found in the dataset.
        """
        assert self._raw_dataset is not None, (
            "Dataset must be loaded before querying; "
            "_raw_dataset is None — call dataset.setup() before querying"
        )
        indices = [self._raw_dataset.candidates.index(cand) for cand in candidates]
        labels = self._raw_dataset.labels[indices]
        return LabelledCandidates(candidates=candidates, labels=labels)

    def determine_num_classes(self) -> int:
        """Determine the number of output neurons based on problem type and labels.

        Raises:
            RuntimeError: If the raw dataset is not loaded.
            ValueError: If the dataset is empty (no labels found).

        Returns:
            int: Number of output neurons. For REGRESSION and BINARY, returns 1 and 2 respectively.
        """
        if self._raw_dataset is None:
            raise RuntimeError(
                "Dataset must be loaded before querying; "
                "_raw_dataset is None — call dataset.setup() before querying"
            )
        problem_type = self.config.problem_type
        if problem_type == ProblemType.REGRESSION:
            return 1
        labels = self._raw_dataset.labels
        unique_labels = np.unique(labels)
        if len(unique_labels) == 0:
            raise ValueError("Dataset is empty — no labels found.")
        if problem_type == ProblemType.BINARY and len(unique_labels) != 2:
            raise ValueError(
                f"problem_type=BINARY requires exactly 2 unique classes, "
                f"got {len(unique_labels)}: {unique_labels.tolist()}"
            )
        if problem_type == ProblemType.MULTICLASS and len(unique_labels) < 3:
            raise ValueError(
                f"problem_type=MULTICLASS requires at least 3 unique classes, "
                f"got {len(unique_labels)}: {unique_labels.tolist()}"
            )
        if problem_type == ProblemType.BINARY:
            return 2
        if problem_type == ProblemType.MULTICLASS:
            return len(unique_labels)
        raise NotImplementedError(
            f"determine_num_classes not implemented for ProblemType {problem_type!r}"
        )

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
            # np.mean on an empty split emits RuntimeWarnings; the value is NaN either way.
            metrics[f"{key}_mean"] = np.mean(split.labels) if len(split) > 0 else float("nan")
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
