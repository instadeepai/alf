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

import copy
import io
import logging
import zipfile
from math import floor
from pathlib import Path
from typing import Literal, Self

import numpy as np
import pandas as pd
import requests
from alf_core import BaseDataset, BaseDatasetConfig, Candidate, LabelledCandidates, ProblemType
from pydantic import model_validator

logger = logging.getLogger("alf-tools")

DATAPATH = Path(__file__).parent / "data"
FLIP_GITHUB_BASE = "https://github.com/J-SNACKKB/FLIP/raw/main/splits"

FLIP_DATASETS = Literal[
    "aav",
    "gb1",
    "meltome",
    "scl",
    "sav",
]

# Active (green) splits per dataset — orange/red splits and FASTA-only datasets excluded
FLIP_SPLITS: dict[str, list[str]] = {
    "aav": ["des_mut", "mut_des", "one_vs_many", "two_vs_many", "seven_vs_many", "low_vs_high"],
    "gb1": ["one_vs_rest", "two_vs_rest", "three_vs_rest", "low_vs_high"],
    "meltome": ["mixed", "human", "human_cell"],
    "scl": ["mixed_soft", "mixed_hard", "human_soft", "human_hard", "balanced", "mixed_vs_human_2"],
    "sav": ["mixed", "human", "only_savs"],
}

# A few FLIP split names don't match their CSV file stem exactly
_SPLIT_FILE_MAP: dict[tuple[str, str], str] = {
    ("meltome", "mixed"): "mixed_split",
}

_EXPECTED_COLUMNS = {"sequence", "target", "set", "validation"}


class FLIPConfig(BaseDatasetConfig):
    """Configuration for FLIP benchmark datasets.

    `train_ratio` controls what fraction of the FLIP train sequences form the
    initial labelled training set. `validation_frac` is applied to that portion
    to carve out a validation set (same as `BaseDataset`). The remaining FLIP
    train sequences become the candidate pool (capped at `max_candidate_pool`).
    `test_ratio` controls what fraction of the FLIP test sequences form the
    evaluation set. `split_type` defaults to `"random"` and need not be set.

    Attributes:
        flip_dataset: Name of the FLIP dataset (e.g. `"gb1"`, `"aav"`).
        flip_split: Name of the split within the dataset (e.g. `"one_vs_rest"`).
            Must be an active (green-status) split. See `FLIP_SPLITS` for valid options.

    Example::

        config = FLIPConfig(
            name="gb1_experiment",
            modality="sequence",
            seed=42,
            flip_dataset="gb1",
            flip_split="one_vs_rest",
            train_ratio=0.1,
            validation_frac=0.1,
            test_ratio=0.8,
            max_candidate_pool=5000,
        )
    """

    flip_dataset: FLIP_DATASETS
    flip_split: str
    problem_type: ProblemType = ProblemType.REGRESSION

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """Override base class validator.

        train_ratio and test_ratio apply to separate FLIP pools (FLIP train and FLIP test),
        so their sum is allowed to exceed 1. Also validates that flip_split is a valid
        active split for the chosen flip_dataset.

        Returns:
            The validated configuration instance.

        Raises:
            ValueError: If `flip_split` is not a valid active split for `flip_dataset`.
        """
        if self.flip_split not in FLIP_SPLITS.get(self.flip_dataset, []):
            raise ValueError(
                f"'{self.flip_split}' is not a valid active split for dataset "
                f"'{self.flip_dataset}'. Valid splits: {FLIP_SPLITS[self.flip_dataset]}"
            )
        return self


class FLIP(BaseDataset):
    """FLIP benchmark dataset class.

    FLIP provides pre-defined train/test splits for protein fitness landscapes.
    The CSV columns are:
        - sequence: amino acid string (may contain special characters)
        - target: float fitness score
        - set: "train" or "test"
        - validation: bool, FLIP-recommended early-stopping flag (stored as a
          candidate feature but not used for splitting)

    The splits map to alf as follows:
        - train_ratio × FLIP train                        -> alf train
        - validation_frac × (train_ratio × FLIP train)   -> alf validation
        - remaining FLIP train (up to max_candidate_pool) -> alf candidate_pool
        - test_ratio × FLIP test                          -> alf test
    """

    config: FLIPConfig  # narrows the inherited BaseDatasetConfig type

    def __init__(self, config: FLIPConfig):
        """Initialise FLIP dataset."""
        super().__init__(config)
        self.setup()

    def __repr__(self) -> str:
        """Return a string representation identifying dataset and split."""
        return (
            f"FLIP(name={self.config.name}, modality={self.modality}, "
            f"seed={self.config.seed}, "
            f"flip_dataset={self.config.flip_dataset}, "
            f"flip_split={self.config.flip_split})"
        )

    def load_dataset(self) -> LabelledCandidates:
        """Load FLIP split from local cache or download from GitHub.

        Downloads splits.zip for the configured dataset if not cached locally,
        then extracts and parses the configured split CSV.

        Returns:
            LabelledCandidates with sequence data, fitness labels, and features
            containing 'set' ("train"/"test") and 'validation' (bool).

        Raises:
            FileNotFoundError: If the split CSV is not found inside splits.zip.
            requests.HTTPError: If the download from GitHub fails.
        """
        df = self._load_split_dataframe()

        candidates = []
        labels = []
        for _, row in df.iterrows():
            candidates.append(
                Candidate(
                    data=row["sequence"],
                    modality=self.modality,
                    features={
                        "set": row["set"],
                        "validation": bool(row["validation"]),
                    },
                )
            )
            labels.append(row["target"])

        return LabelledCandidates(
            candidates=candidates,
            labels=np.array(labels),
        )

    def _split_dataset(self) -> dict[str, LabelledCandidates]:
        """Split the raw dataset into train, validation, candidate_pool, and test.

        The FLIP train sequences are shuffled, then partitioned using train_ratio
        and validation_frac. The remainder becomes the candidate pool (capped at
        max_candidate_pool). The FLIP test sequences are sampled using test_ratio.

        Returns:
            Dictionary with keys "train", "validation", "test", "candidate_pool".

        Raises:
            RuntimeError: If dataset has not been loaded yet.
        """
        if self._raw_dataset is None:
            raise RuntimeError(
                "Dataset must be loaded before splitting; "
                "_raw_dataset is None — call dataset.setup() (or load_dataset()) before splitting"
            )

        # Separate FLIP's pre-defined train and test pools
        flip_train = LabelledCandidates(candidates=[], labels=np.array([]))
        flip_test = LabelledCandidates(candidates=[], labels=np.array([]))
        for candidate, label in self._raw_dataset:
            if candidate.features and candidate.features["set"] == "train":
                flip_train.append([candidate], np.array([label]))
            else:
                flip_test.append([candidate], np.array([label]))

        logger.debug("FLIP pools — train: %d, test: %d", len(flip_train), len(flip_test))

        # Shuffle FLIP train for reproducible random splitting
        flip_train = flip_train.shuffle(self.config.seed)

        # Split FLIP train into train, validation, and candidate_pool
        train_plus_val_size = floor(len(flip_train) * self.split_ratio["train"])
        validation_size = floor(train_plus_val_size * self.split_ratio["validation_frac"])
        train_size = train_plus_val_size - validation_size

        train = LabelledCandidates(
            candidates=flip_train.candidates[:train_size],
            labels=flip_train.labels[:train_size],
        )
        validation = LabelledCandidates(
            candidates=flip_train.candidates[train_size:train_plus_val_size],
            labels=flip_train.labels[train_size:train_plus_val_size],
        )

        pool_candidates = flip_train.candidates[train_plus_val_size:]
        pool_labels = flip_train.labels[train_plus_val_size:]
        if self.config.max_candidate_pool is not None:
            pool_candidates = pool_candidates[: self.config.max_candidate_pool]
            pool_labels = pool_labels[: self.config.max_candidate_pool]
        candidate_pool = LabelledCandidates(candidates=pool_candidates, labels=pool_labels)

        # Sample FLIP test using test_ratio
        test_size = floor(len(flip_test) * self.split_ratio["test"])
        test = LabelledCandidates(
            candidates=flip_test.candidates[:test_size],
            labels=flip_test.labels[:test_size],
        )

        logger.debug(
            "Split sizes — train: %d, validation: %d, candidate_pool: %d, test: %d",
            len(train),
            len(validation),
            len(candidate_pool),
            len(test),
        )

        self.init_candidate_pool = copy.deepcopy(candidate_pool)

        return {
            "train": train,
            "validation": validation,
            "test": test,
            "candidate_pool": candidate_pool,
        }

    def _load_split_dataframe(self) -> pd.DataFrame:
        """Load the split CSV as a DataFrame, downloading splits.zip if needed.

        Returns:
            DataFrame with columns: sequence, target, set, validation.

        Raises:
            FileNotFoundError: If split CSV not found inside the zip archive.
            ValueError: If the loaded CSV is missing expected columns.
            requests.HTTPError: If the GitHub download fails.
        """
        zip_path = DATAPATH / "FLIP" / self.config.flip_dataset / "splits.zip"

        if not zip_path.exists():
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            url = f"{FLIP_GITHUB_BASE}/{self.config.flip_dataset}/splits.zip"
            logger.info(f"Downloading FLIP splits for '{self.config.flip_dataset}' from {url}")
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info("Download complete.")
        else:
            logger.debug("Using cached splits.zip at %s", zip_path)

        file_stem = _SPLIT_FILE_MAP.get(
            (self.config.flip_dataset, self.config.flip_split),
            self.config.flip_split,
        )
        csv_name = f"{file_stem}.csv"
        with zipfile.ZipFile(zip_path, "r") as zf:
            available = zf.namelist()
            match = next((n for n in available if n.endswith(csv_name)), None)
            if match is None:
                raise FileNotFoundError(
                    f"'{csv_name}' not found in splits.zip for dataset "
                    f"'{self.config.flip_dataset}'. Available files: {available}"
                )
            with zf.open(match) as csv_file:
                df = pd.read_csv(io.BytesIO(csv_file.read()))

        missing = _EXPECTED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"FLIP CSV '{csv_name}' is missing expected columns: {missing}. "
                f"Found: {set(df.columns)}"
            )

        return df
