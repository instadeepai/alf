import logging
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
from alf_core import BaseDataset, Candidate, LabeledCandidates
from alf_tools.utils.constants import HF_DATASETS_REPOSITORY_NAME
from huggingface_hub import hf_hub_download

logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]")
log = logging.getLogger("rich")

DATAPATH = "alf_tools/datasets/data/"


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
        """
        dms_name = self.dataset_config.get("dms_name", None)
        assert dms_name is not None, "DMS name must be set"
        dms_type = self.dataset_config.get("dms_type", None)
        assert dms_type is not None and dms_type in ["singles", "multiples"], (
            "DMS type must be set and must be one of singles or multiples"
        )

        assert os.environ.get("HF_TOKEN") is not None, "HF token must be set"

        filename = f"ProteinGym/ProteinGym_Cross_Validation/{dms_type}/{dms_name}.csv"
        if not os.path.exists(f"{DATAPATH}/{filename}"):
            os.makedirs(DATAPATH, exist_ok=True)
            log.info("No local copy of file, so downloading from hub")
            hf_hub_download(
                repo_id=HF_DATASETS_REPOSITORY_NAME,
                filename=filename,
                repo_type="dataset",
                local_dir=DATAPATH,
            )
            log.info("ProteinGym dataset downloaded successfully.")

        df = pd.read_csv(f"{DATAPATH}/{filename}")
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

    def _split_dataset(self) -> Dict[str, LabeledCandidates]:
        """Split dataset into train, validation, test and candidate pool splits.

        Returns:
            Dict[str, LabeledCandidates]: A dictionary of the splits.
        """
        if self.dataset_config.get("cross_validation", False):
            assert self.dataset_config.get("cross_validation_type", None) is not None, (
                "Cross-validation type must be set"
            )
            assert self.dataset_config.get("cross_validation_fold", None) is not None, (
                "Cross-validation fold must be set"
            )
            assert self.dataset_config["cross_validation_type"] in [
                "random",
                "modulo",
                "contiguous",
            ], "Cross-validation type must be one of random, modulo, or contiguous"
            assert 5 > self.dataset_config["cross_validation_fold"] >= 0, (
                "Cross-validation fold must be between 0 and 4 inclusive"
            )
            return self._split_cross_validation()
        else:
            return super()._split_dataset()

    def _split_cross_validation(self) -> Dict[str, LabeledCandidates]:
        """Split dataset into cross-validation folds.

        Returns:
            Dict[str, LabeledCandidates]: A dictionary of the splits.
        """
        assert self._raw_dataset is not None, "Dataset must be loaded before splitting"
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
