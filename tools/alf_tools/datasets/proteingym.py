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

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from alf_core import BaseDataset, Candidate, LabeledCandidates
from alf_tools.utils.constants import HF_DATASETS_REPOSITORY_NAME
from huggingface_hub import hf_hub_download

logger = logging.getLogger("alf-tools")

DATAPATH = Path(__file__).parent / "data"


class ProteinGym(BaseDataset):
    """ProteinGym dataset class."""

    def __init__(
        self,
        name: str,
        modality: str,
        seed: int,
        split_config: dict[str, Any],
        dataset_config: dict[str, Any],
    ):
        """Initialize ProteinGym dataset.

        Args:
            name: Name of the dataset.
            modality: Modality of the data.
            seed: Random seed for reproducibility.
            split_config: Split configuration.
            dataset_config: Dataset configuration.
        """
        super().__init__(name, modality, seed, split_config)
        self.dataset_config = dataset_config
        self.setup()

    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        return (
            f"ProteinGym(name={self.name}, modality={self.modality}, seed={self.seed}, "
            f"split_ratio={self.split_ratio}, dataset_config={self.dataset_config})"
        )

    def load_dataset(self) -> LabeledCandidates:
        """Load ProteinGym dataset from local file or download from HF if not present.
        Process dataset and return as labeled candidates.

        Returns:
            LabeledCandidates: Labeled candidates with ProteinGym data.

        Raises:
            ValueError: If HF token is not set as environment variable or config has missing fields.
        """
        # Check HF token is set as environment variable and config has required fields
        if os.environ.get("HF_TOKEN") is None:
            raise ValueError("HF token must be set as environment variable")
        dms_name = self.dataset_config.get("dms_name", None)
        if dms_name is None:
            raise ValueError("DMS name must be set")
        dms_type = self.dataset_config.get("dms_type", None)
        if dms_type is None or dms_type not in ["singles", "multiples"]:
            raise ValueError("DMS type must be set and must be one of singles or multiples")

        filename = f"ProteinGym/ProteinGym_Cross_Validation/{dms_type}/{dms_name}.csv"
        filepath = DATAPATH / filename
        if not filepath.exists():
            DATAPATH.mkdir(parents=True, exist_ok=True)
            logger.info("No local copy of file, so downloading from hub")
            hf_hub_download(
                repo_id=HF_DATASETS_REPOSITORY_NAME,
                filename=filename,
                repo_type="dataset",
                local_dir=DATAPATH,
            )
            logger.info("ProteinGym dataset downloaded successfully.")

        df = pd.read_csv(filepath)
        dataset = LabeledCandidates(candidates=[], labels=np.array([]))
        for _, row in df.iterrows():
            data = row["mutated_sequence"]
            label = row["DMS_score"]
            features = {"mutant_code": row["mutant"]}
            if dms_type == "singles":
                features.update({
                    "random_fold_id": row["fold_random_5"],
                    "modulo_fold_id": row["fold_modulo_5"],
                    "fold_contiguous_id": row["fold_contiguous_5"],
                })
            else:
                features.update({
                    "random_fold_id": row["fold_rand_multiples"],
                })
            dataset.append(
                [Candidate(data=data, modality=self.modality, features=features)], np.array([label])
            )

        return dataset

    def _split_dataset(self) -> dict[str, LabeledCandidates]:
        """Split dataset into train, validation, test and candidate pool splits.

        Returns:
            dict[str, LabeledCandidates]: A dictionary of the splits.

        Raises:
            ValueError: If cross-validation type or fold is not set or invalid.
        """
        if self.dataset_config.get("cross_validation", False):
            cross_validation_type = self.dataset_config.get("cross_validation_type", None)
            if cross_validation_type is None and cross_validation_type not in [
                "random",
                "modulo",
                "contiguous",
            ]:
                raise ValueError(
                    "Cross-validation type must be set and be one of random, modulo, or contiguous"
                )

            cross_validation_fold = self.dataset_config.get("cross_validation_fold", None)
            if (
                cross_validation_fold is None
                or cross_validation_fold < 0
                or cross_validation_fold > 4
            ):
                raise ValueError(
                    "Cross-validation fold must be set and must be between 0 and 4 inclusive"
                )

            return self._split_cross_validation()
        else:
            return super()._split_dataset()

    def _split_cross_validation(self) -> dict[str, LabeledCandidates]:
        """Split dataset into cross-validation folds.

        Returns:
            dict[str, LabeledCandidates]: A dictionary of the splits.

        Raises:
            ValueError: If dataset is not loaded before splitting.
        """
        if self._raw_dataset is None:
            raise ValueError("Dataset must be loaded before splitting")
        # Calculate split sizes
        dataset_size = len(self._raw_dataset)
        train_plus_validation_size = round(dataset_size * self.split_ratio["train"])
        validation_size = round(train_plus_validation_size * self.split_ratio["validation_frac"])
        train_size = train_plus_validation_size - validation_size
        test_size = round(dataset_size * self.split_ratio["test"])
        candidate_pool_size = dataset_size - train_plus_validation_size - test_size
        if self.max_candidate_pool is not None:
            candidate_pool_size = min(candidate_pool_size, self.max_candidate_pool)

        # Shuffle dataset
        shuffled_dataset = self._raw_dataset.shuffle(self.seed)
        train_and_validation_dataset = LabeledCandidates(candidates=[], labels=[])
        test_and_candidate_pool_dataset = LabeledCandidates(candidates=[], labels=[])

        # Split dataset into train/test sets depending on cross-validation fold
        cv_type = f"{self.dataset_config['cross_validation_type']}_fold_id"
        cv_fold = self.dataset_config["cross_validation_fold"]
        for candidate, label in shuffled_dataset:
            if cv_fold == candidate.features[cv_type]:
                test_and_candidate_pool_dataset.append([candidate], [label])
            else:
                train_and_validation_dataset.append([candidate], [label])

        # Split train/validation and test/candidate pool sets
        train_dataset = LabeledCandidates(*train_and_validation_dataset[:train_size])
        validation_dataset = LabeledCandidates(*train_and_validation_dataset[train_size:])
        test_dataset = LabeledCandidates(*test_and_candidate_pool_dataset[:test_size])
        candidate_pool_dataset = LabeledCandidates(
            *test_and_candidate_pool_dataset[test_size : test_size + candidate_pool_size]
        )

        return {
            "train": train_dataset,
            "validation": validation_dataset,
            "test": test_dataset,
            "candidate_pool": candidate_pool_dataset,
        }
