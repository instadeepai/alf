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

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from alf_core import Modality
from alf_tools.datasets.proteingym import (
    ProteinGym,
    ProteinGymConfig,
    _add_fold_columns_multiples,  # noqa: PLC2701
    _add_fold_columns_singles,  # noqa: PLC2701
)
from pydantic import ValidationError


@pytest.fixture
def proteingym_dataset_singles():
    """Create a ProteinGym singles dataset fixture for testing."""
    config = ProteinGymConfig(
        name="proteingym",
        modality="sequence",
        seed=51505,
        train_ratio=0.6,
        validation_frac=0.2,
        test_ratio=0.2,
        split_type="random",
        problem_type="regression",
        dms_name="IF1_ECOLI_Kelsic_2016",
        dms_type="singles",
    )
    return ProteinGym(config)


@pytest.fixture
def proteingym_dataset_multiples():
    """Create a ProteinGym multiples dataset fixture for testing."""
    config = ProteinGymConfig(
        name="proteingym",
        modality="sequence",
        seed=51505,
        train_ratio=0.6,
        validation_frac=0.2,
        test_ratio=0.2,
        split_type="random",
        problem_type="regression",
        dms_name="CAPSD_AAV2S_Sinai_2021",
        dms_type="multiples",
    )
    return ProteinGym(config)


# Modulo fold 0: 284 rows have (position-1) % 5 == 0; the remaining 1083 are train.
# Ratios chosen so round(1367 * ratio) gives exact fold counts.
@pytest.fixture
def proteingym_dataset_cv_singles():
    """Create a ProteinGym cross-validation singles dataset fixture for testing."""
    config = ProteinGymConfig(
        name="proteingym",
        modality="sequence",
        seed=51505,
        train_ratio=0.7921,
        validation_frac=0.0,
        test_ratio=0.2079,
        split_type="random",
        problem_type="regression",
        dms_name="IF1_ECOLI_Kelsic_2016",
        dms_type="singles",
        cross_validation=True,
        cross_validation_type="modulo",
        cross_validation_fold=0,
    )
    return ProteinGym(config)


# Random fold 0 (seed=0, row-level): 8466 rows in fold 0; 33862 in train.
@pytest.fixture
def proteingym_dataset_cv_multiples():
    """Create a ProteinGym cross-validation multiples dataset fixture for testing."""
    config = ProteinGymConfig(
        name="proteingym",
        modality="sequence",
        seed=51505,
        train_ratio=0.79998,
        validation_frac=0.0,
        test_ratio=0.20002,
        split_type="random",
        problem_type="regression",
        dms_name="CAPSD_AAV2S_Sinai_2021",
        dms_type="multiples",
        cross_validation=True,
        cross_validation_type="random",
        cross_validation_fold=0,
    )
    return ProteinGym(config)


@pytest.mark.integration
class TestProteinGymDataset:
    """Integration tests that download the real public ProteinGym_v1 dataset."""

    def test_dataset_initialization_singles(self, proteingym_dataset_singles):
        """Test that singles dataset initializes correctly."""
        assert proteingym_dataset_singles.config.name == "proteingym"
        assert proteingym_dataset_singles.modality == Modality.SEQUENCE
        assert proteingym_dataset_singles.config.seed == 51505

    def test_dataset_initialization_multiples(self, proteingym_dataset_multiples):
        """Test that multiples dataset initializes correctly."""
        assert proteingym_dataset_multiples.config.name == "proteingym"
        assert proteingym_dataset_multiples.modality == Modality.SEQUENCE
        assert proteingym_dataset_multiples.config.seed == 51505

    def test_cross_validation_singles_split_sizes(self, proteingym_dataset_cv_singles):
        """Modulo fold 0 for IF1_ECOLI_Kelsic_2016: 284 test, 1083 train."""
        assert len(proteingym_dataset_cv_singles.train_dataset) == 1083, (
            f"Expected 1083 train samples, got {len(proteingym_dataset_cv_singles.train_dataset)}"
        )
        assert len(proteingym_dataset_cv_singles.test_dataset) == 284, (
            f"Expected 284 test samples, got {len(proteingym_dataset_cv_singles.test_dataset)}"
        )
        assert len(proteingym_dataset_cv_singles.validation_dataset) == 0
        assert len(proteingym_dataset_cv_singles.candidate_pool) == 0

    def test_cross_validation_singles_label_statistics(self, proteingym_dataset_cv_singles):
        """Label statistics for modulo fold 0 of IF1_ECOLI_Kelsic_2016."""
        train_mean = np.mean(proteingym_dataset_cv_singles.train_dataset.labels)
        test_mean = np.mean(proteingym_dataset_cv_singles.test_dataset.labels)
        assert np.isclose(train_mean, 0.806602, atol=1e-4), f"Train mean off: {train_mean}"
        assert np.isclose(test_mean, 0.738843, atol=1e-4), f"Test mean off: {test_mean}"

    def test_cross_validation_multiples_split_sizes(self, proteingym_dataset_cv_multiples):
        """Random fold 0 (seed=0) for CAPSD_AAV2S_Sinai_2021: 8466 test, 33862 train."""
        assert len(proteingym_dataset_cv_multiples.train_dataset) == 33862, (
            f"Expected 33862 train samples, got "
            f"{len(proteingym_dataset_cv_multiples.train_dataset)}"
        )
        assert len(proteingym_dataset_cv_multiples.test_dataset) == 8466, (
            f"Expected 8466 test samples, got {len(proteingym_dataset_cv_multiples.test_dataset)}"
        )
        assert len(proteingym_dataset_cv_multiples.validation_dataset) == 0
        assert len(proteingym_dataset_cv_multiples.candidate_pool) == 0

    def test_cross_validation_multiples_label_statistics(self, proteingym_dataset_cv_multiples):
        """Label statistics for random fold 0 of CAPSD_AAV2S_Sinai_2021."""
        train_mean = np.mean(proteingym_dataset_cv_multiples.train_dataset.labels)
        test_mean = np.mean(proteingym_dataset_cv_multiples.test_dataset.labels)
        assert np.isclose(train_mean, -1.224675, atol=1e-4), f"Train mean off: {train_mean}"
        assert np.isclose(test_mean, -1.230568, atol=1e-4), f"Test mean off: {test_mean}"


class TestProteinGymCrossValidationConfig:
    """Validation of the cross-validation configuration."""

    def _base_kwargs(self, **overrides):
        kwargs = dict(
            name="proteingym",
            modality="sequence",
            seed=0,
            train_ratio=0.5,
            validation_frac=0.0,
            test_ratio=0.5,
            split_type="random",
            problem_type="regression",
            dms_name="X",
            dms_type="singles",
        )
        kwargs.update(overrides)
        return kwargs

    def test_missing_type_raises(self):
        """cross_validation=True without a type raises a clear error."""
        with pytest.raises(ValidationError, match="cross_validation_type"):
            ProteinGymConfig(**self._base_kwargs(cross_validation=True, cross_validation_fold=0))

    def test_missing_fold_raises(self):
        """cross_validation=True without a fold raises a clear error."""
        with pytest.raises(ValidationError, match="cross_validation_fold"):
            ProteinGymConfig(
                **self._base_kwargs(cross_validation=True, cross_validation_type="contiguous")
            )

    @pytest.mark.parametrize("cv_type", ["modulo", "contiguous"])
    def test_multiples_only_supports_random(self, cv_type):
        """dms_type='multiples' rejects non-random folds (only random is available)."""
        with pytest.raises(ValidationError, match="only provides 'random'"):
            ProteinGymConfig(
                **self._base_kwargs(
                    dms_type="multiples",
                    cross_validation=True,
                    cross_validation_type=cv_type,
                    cross_validation_fold=0,
                )
            )

    def test_singles_contiguous_config_is_valid(self):
        """A singles + contiguous CV config validates without error."""
        config = ProteinGymConfig(
            **self._base_kwargs(
                cross_validation=True,
                cross_validation_type="contiguous",
                cross_validation_fold=0,
            )
        )
        assert config.cross_validation_type == "contiguous"


class TestProteinGymFoldComputation:
    """Unit tests for the fold-column helper functions."""

    def _singles_df(self, positions: list[int]) -> pd.DataFrame:
        return pd.DataFrame({
            "mutant": [f"A{p}G" for p in positions],
            "mutated_sequence": ["MAKG" for _ in positions],
            "DMS_score": [0.0] * len(positions),
        })

    def test_modulo_fold_formula(self):
        """fold_modulo_5 == (position - 1) % 5 for all positions."""
        df = self._singles_df(list(range(1, 21)))
        result = _add_fold_columns_singles(df)
        result["_pos"] = result["mutant"].str.extract(r"(\d+)").astype(int)
        result = result.sort_values("_pos")
        expected = [(p - 1) % 5 for p in range(1, 21)]
        assert list(result["fold_modulo_5"]) == expected

    def test_contiguous_fold_five_balanced_groups(self):
        """fold_contiguous_5 partitions 10 positions into 5 groups of 2."""
        df = self._singles_df(list(range(1, 11)))
        result = _add_fold_columns_singles(df)
        fold_counts = result.groupby("fold_contiguous_5").size().to_dict()
        assert all(v == 2 for v in fold_counts.values())

    def test_random_fold_deterministic(self):
        """fold_random_5 is deterministic across calls with the same data."""
        df = self._singles_df(list(range(1, 21)))
        r1 = _add_fold_columns_singles(df.copy())
        r2 = _add_fold_columns_singles(df.copy())
        assert list(r1["fold_random_5"]) == list(r2["fold_random_5"])

    def test_multiples_fold_covers_all_five(self):
        """fold_rand_multiples assigns all 5 fold values for n >= 5 variants."""
        df = pd.DataFrame({
            "mutant": [f"A{i}G:B{i + 1}H" for i in range(1, 21)],
            "mutated_sequence": ["MAKG"] * 20,
            "DMS_score": [0.0] * 20,
        })
        result = _add_fold_columns_multiples(df)
        assert set(result["fold_rand_multiples"].unique()) == {0, 1, 2, 3, 4}


class TestProteinGymContiguousSplit:
    """The contiguous cross-validation split uses the contiguous fold column."""

    def test_contiguous_split_selects_correct_fold(self):
        """Regression test: contiguous CV must hold out exactly the contiguous fold.

        We patch _download_dms_dataframe to return a small synthetic dataframe so
        no network access is needed.
        """
        fold = 1
        config = ProteinGymConfig(
            name="proteingym",
            modality="sequence",
            seed=0,
            train_ratio=0.5,
            validation_frac=0.0,
            test_ratio=0.5,
            split_type="random",
            problem_type="regression",
            dms_name="X",
            dms_type="singles",
            cross_validation=True,
            cross_validation_type="contiguous",
            cross_validation_fold=fold,
        )

        # 20 single-point mutations at positions 1-20 (one per position).
        raw_df = pd.DataFrame({
            "mutant": [f"A{p}G" for p in range(1, 21)],
            "mutated_sequence": ["MAKG" * 5] * 20,
            "DMS_score": [float(i) for i in range(20)],
        })

        with patch("alf_tools.datasets.proteingym._download_dms_dataframe", return_value=raw_df):
            dataset = ProteinGym(config)

        held_out = dataset.test_dataset.candidates + dataset.candidate_pool.candidates
        assert len(held_out) > 0
        assert all(c.features["contiguous_fold_id"] == fold for c in held_out)
        assert all(
            c.features["contiguous_fold_id"] != fold
            for c in dataset.train_dataset.candidates + dataset.validation_dataset.candidates
        )
