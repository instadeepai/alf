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
from typing import Literal

import numpy as np
import pandas as pd
from alf_core import BaseDataset, Candidate, LabeledCandidates
from alf_core.dataset.base_dataset import BaseDatasetConfig
from alf_tools.utils.constants import HF_DATASETS_REPOSITORY_NAME
from huggingface_hub import hf_hub_download

logger = logging.getLogger("alf-tools")

DATAPATH = Path(__file__).parent / "data"


class ProteinGymConfig(BaseDatasetConfig):
    """Configuration for ProteinGym dataset.

    Attributes:
        dms_name: Name of the DMS assay (e.g., "IF1_ECOLI_Kelsic_2016").
        dms_type: Type of DMS data ("singles" or "multiples").
        cross_validation: Whether to use cross-validation splits.
        cross_validation_type: Type of CV split ("random", "modulo", or "contiguous").
        cross_validation_fold: Which CV fold to use (0-4).
    """

    dms_name: str
    dms_type: Literal["singles", "multiples"]
    cross_validation: bool = False
    cross_validation_type: Literal["random", "modulo", "contiguous"] | None = None
    cross_validation_fold: Literal[0, 1, 2, 3, 4] | None = None


class ProteinGym(BaseDataset):
    """ProteinGym dataset class."""

    def __init__(self, config: ProteinGymConfig):
        """Initialize ProteinGym dataset.

        Args:
            config: Configuration for the ProteinGym dataset.
        """
        super().__init__(config)
        self.setup()

    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        return (
            f"ProteinGym(name={self.config.name}, modality={self.modality}, "
            f"seed={self.config.seed}, split_ratio={self.split_ratio}, "
            f"dms_name={self.config.dms_name})"
        )

    def load_dataset(self) -> LabeledCandidates:
        """Load ProteinGym dataset from local file or download from HF if not present.
        Process dataset and return as labeled candidates.

        Returns:
            Labeled candidates with ProteinGym data.

        Raises:
            ValueError: If HF token is not set as environment variable.
        """
        if os.environ.get("HF_TOKEN") is None:
            raise ValueError("HF token must be set as environment variable")

        dms_name = self.config.dms_name
        dms_type = self.config.dms_type

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
            A dictionary of the splits.
        """
        if self.config.cross_validation:
            return self._split_cross_validation()
        else:
            return super()._split_dataset()

    def _split_cross_validation(self) -> dict[str, LabeledCandidates]:
        """Split dataset into cross-validation folds.

        Returns:
            A dictionary of the splits.

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
        if self.config.max_candidate_pool is not None:
            candidate_pool_size = min(candidate_pool_size, self.config.max_candidate_pool)

        # Shuffle dataset
        shuffled_dataset = self._raw_dataset.shuffle(self.config.seed)
        train_and_validation_dataset = LabeledCandidates(candidates=[], labels=[])
        test_and_candidate_pool_dataset = LabeledCandidates(candidates=[], labels=[])

        # Split dataset into train/test sets depending on cross-validation fold
        cv_type = f"{self.config.cross_validation_type}_fold_id"
        cv_fold = self.config.cross_validation_fold
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
