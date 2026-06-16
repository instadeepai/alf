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
from pathlib import Path
from typing import Literal, Self

import numpy as np
import pandas as pd
from alf_core import BaseDataset, Candidate, LabelledCandidates, ProblemType
from alf_core.dataset.base_dataset import BaseDatasetConfig
from huggingface_hub import hf_hub_download
from pydantic import model_validator

logger = logging.getLogger("alf-tools")

DATAPATH = Path(__file__).parent / "data"

# Public ProteinGym dataset — no token required.
_UPSTREAM_REPO = "OATML-Markslab/ProteinGym_v1"
_N_SUBSTITUTION_SHARDS = 5

# Fixed seed for the random CV fold assignment.  This does not reproduce the
# original ProteinGym benchmark random folds; use modulo or contiguous splits
# when exact benchmark parity is required.
_RANDOM_CV_SEED = 0


def _download_dms_dataframe(dms_name: str) -> pd.DataFrame:
    """Download a single DMS assay from ProteinGym_v1 by searching parquet shards.

    Each assay (DMS_id) lives entirely within one shard, so the search stops as
    soon as the matching shard is found.  Shards are cached by huggingface_hub
    so repeat calls are fast.

    Args:
        dms_name: DMS assay identifier (e.g. "IF1_ECOLI_Kelsic_2016").

    Returns:
        DataFrame with columns: mutant, mutated_sequence, DMS_score.

    Raises:
        ValueError: If dms_name is not found in any shard.
    """
    for shard_idx in range(_N_SUBSTITUTION_SHARDS):
        filename = (
            f"DMS_substitutions/train-0000{shard_idx}"
            f"-of-0000{_N_SUBSTITUTION_SHARDS}.parquet"
        )
        local_path = hf_hub_download(
            repo_id=_UPSTREAM_REPO,
            filename=filename,
            repo_type="dataset",
            local_dir=DATAPATH / ".cache" / "proteingym_v1",
        )
        shard_df = pd.read_parquet(local_path)
        result = shard_df[shard_df["DMS_id"] == dms_name]
        if len(result) > 0:
            return result[["mutant", "mutated_sequence", "DMS_score"]].reset_index(drop=True)
    raise ValueError(
        f"DMS assay '{dms_name}' was not found in any shard of {_UPSTREAM_REPO}. "
        "Check that dms_name matches the DMS_id in the ProteinGym_v1 dataset."
    )


def _add_fold_columns_singles(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and attach CV fold columns for a single-substitution assay.

    Three fold strategies are computed, all based on the mutated position:

    - ``fold_modulo_5``:    ``(position - 1) % 5``  (matches ProteinGym benchmark)
    - ``fold_contiguous_5``: positions partitioned into 5 contiguous blocks of
      equal size via ``numpy.array_split``  (matches ProteinGym benchmark)
    - ``fold_random_5``:    position-level random assignment with seed
      ``_RANDOM_CV_SEED``.  **Does not reproduce the original ProteinGym random
      folds**; use modulo/contiguous for benchmark parity.

    Args:
        df: DataFrame with at least a ``mutant`` column (e.g. "A16C").

    Returns:
        The input DataFrame (sorted by mutant, reindexed) with three new columns.
    """
    df = df.sort_values("mutant").reset_index(drop=True)
    df["_pos"] = df["mutant"].str.extract(r"(\d+)").astype(int)
    positions = sorted(df["_pos"].unique())

    df["fold_modulo_5"] = (df["_pos"] - 1) % 5

    groups = np.array_split(positions, 5)
    pos_to_contiguous = {p: fold for fold, grp in enumerate(groups) for p in grp}
    df["fold_contiguous_5"] = df["_pos"].map(pos_to_contiguous)

    pos_folds = np.array([i % 5 for i in range(len(positions))])
    np.random.default_rng(_RANDOM_CV_SEED).shuffle(pos_folds)
    pos_to_random = dict(zip(positions, pos_folds))
    df["fold_random_5"] = df["_pos"].map(pos_to_random)

    return df.drop(columns=["_pos"])


def _add_fold_columns_multiples(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and attach a random CV fold column for a multi-substitution assay.

    Folds are assigned at the variant (row) level rather than position level
    because multi-site mutations span many positions.

    Args:
        df: DataFrame with at least a ``mutant`` column.

    Returns:
        The input DataFrame (sorted by mutant, reindexed) with ``fold_rand_multiples``.
    """
    df = df.sort_values("mutant").reset_index(drop=True)
    n = len(df)
    folds = np.array([i % 5 for i in range(n)])
    np.random.default_rng(_RANDOM_CV_SEED).shuffle(folds)
    df["fold_rand_multiples"] = folds
    return df


class ProteinGymConfig(BaseDatasetConfig):
    """Configuration for ProteinGym dataset.

    Attributes:
        dms_name: Name of the DMS assay (e.g., "IF1_ECOLI_Kelsic_2016").
        dms_type: Type of DMS data ("singles" or "multiples").
        cross_validation: Whether to use cross-validation splits.
        cross_validation_type: Type of CV split ("random", "modulo", or "contiguous").
            Only "random" folds are available for dms_type="multiples".
        cross_validation_fold: Which CV fold to use (0-4).
    """

    dms_name: str
    dms_type: Literal["singles", "multiples"]
    cross_validation: bool = False
    cross_validation_type: Literal["random", "modulo", "contiguous"] | None = None
    cross_validation_fold: Literal[0, 1, 2, 3, 4] | None = None
    problem_type: ProblemType = ProblemType.REGRESSION

    @model_validator(mode="after")
    def _validate_cross_validation(self) -> Self:
        """Validate the cross-validation configuration.

        Returns:
            The validated configuration instance.

        Raises:
            ValueError: If cross_validation is enabled without both
                cross_validation_type and cross_validation_fold set, or if a
                non-"random" fold is requested for dms_type="multiples" (which
                only provides random folds).
        """
        if not self.cross_validation:
            return self
        if self.cross_validation_type is None or self.cross_validation_fold is None:
            raise ValueError(
                "cross_validation=True requires both cross_validation_type and "
                "cross_validation_fold to be set."
            )
        if self.dms_type == "multiples" and self.cross_validation_type != "random":
            raise ValueError(
                "dms_type='multiples' only provides 'random' cross-validation folds; "
                f"cross_validation_type='{self.cross_validation_type}' is not available."
            )
        return self


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

    def load_dataset(self) -> LabelledCandidates:
        """Load ProteinGym dataset, downloading from the public ProteinGym_v1 HF
        repository if a local cache is not present.

        Data is sourced from ``OATML-Markslab/ProteinGym_v1`` — no HuggingFace
        token is required.  CV fold columns are computed deterministically and
        saved alongside the raw data so subsequent loads are instant.

        Returns:
            Labeled candidates with ProteinGym data.
        """
        dms_name = self.config.dms_name
        dms_type = self.config.dms_type

        filepath = DATAPATH / "ProteinGym" / dms_name / "data.csv"
        if not filepath.exists():
            filepath.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading %s from %s", dms_name, _UPSTREAM_REPO)
            df = _download_dms_dataframe(dms_name)
            if dms_type == "singles":
                df = _add_fold_columns_singles(df)
            else:
                df = _add_fold_columns_multiples(df)
            df.to_csv(filepath, index=False)
            logger.info("ProteinGym dataset cached at %s", filepath)

        df = pd.read_csv(filepath)
        dataset = LabelledCandidates(candidates=[], labels=np.array([]))
        for _, row in df.iterrows():
            data = row["mutated_sequence"]
            label = row["DMS_score"]
            features = {"mutant_code": row["mutant"]}
            if dms_type == "singles":
                features.update({
                    "random_fold_id": row["fold_random_5"],
                    "modulo_fold_id": row["fold_modulo_5"],
                    "contiguous_fold_id": row["fold_contiguous_5"],
                })
            else:
                features.update({
                    "random_fold_id": row["fold_rand_multiples"],
                })
            dataset.append(
                [Candidate(data=data, modality=self.modality, features=features)], np.array([label])
            )

        return dataset

    def _split_dataset(self) -> dict[str, LabelledCandidates]:
        """Split dataset into train, validation, test and candidate pool splits.

        Returns:
            A dictionary of the splits.
        """
        if self.config.cross_validation:
            return self._split_cross_validation()
        else:
            return super()._split_dataset()

    def _split_cross_validation(self) -> dict[str, LabelledCandidates]:
        """Split dataset into cross-validation folds.

        Returns:
            A dictionary of the splits.

        Raises:
            RuntimeError: If dataset is not loaded before splitting.
        """
        if self._raw_dataset is None:
            raise RuntimeError(
                "_raw_dataset is None — call dataset.setup() (or load_dataset()) before splitting"
            )
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
        train_and_validation_dataset = LabelledCandidates(candidates=[], labels=np.array([]))
        test_and_candidate_pool_dataset = LabelledCandidates(candidates=[], labels=np.array([]))

        # Split dataset into train/test sets depending on cross-validation fold
        cv_type = f"{self.config.cross_validation_type}_fold_id"
        cv_fold = self.config.cross_validation_fold
        for candidate, label in shuffled_dataset:
            if cv_fold == candidate.features[cv_type]:
                test_and_candidate_pool_dataset.append([candidate], np.array([label]))
            else:
                train_and_validation_dataset.append([candidate], np.array([label]))

        # Split train/validation and test/candidate pool sets
        train_dataset = LabelledCandidates(*train_and_validation_dataset[:train_size])
        validation_dataset = LabelledCandidates(*train_and_validation_dataset[train_size:])
        test_dataset = LabelledCandidates(*test_and_candidate_pool_dataset[:test_size])
        candidate_pool_dataset = LabelledCandidates(
            *test_and_candidate_pool_dataset[test_size : test_size + candidate_pool_size]
        )

        return {
            "train": train_dataset,
            "validation": validation_dataset,
            "test": test_dataset,
            "candidate_pool": candidate_pool_dataset,
        }
