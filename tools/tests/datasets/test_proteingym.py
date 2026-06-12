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

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from alf_core import Modality
from alf_tools.datasets.proteingym import ProteinGym, ProteinGymConfig
from pydantic import ValidationError


@pytest.fixture
def proteingym_dataset_singles():
    """Create a ProteinGym singles dataset fixture for testing.

    Returns:
        A ProteinGym singles dataset.
    """
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
    """Create a ProteinGym multiples dataset fixture for testing.

    Returns:
        A ProteinGym multiples dataset.
    """
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


@pytest.fixture
def proteingym_dataset_cv_singles():
    """Create a ProteinGym cross-validation singles dataset fixture for testing.

    Returns:
        A ProteinGym cross-validation singles dataset.
    """
    config = ProteinGymConfig(
        name="proteingym",
        modality="sequence",
        seed=51505,
        train_ratio=0.7827,
        validation_frac=0.0,
        test_ratio=0.2173,
        split_type="random",
        problem_type="regression",
        dms_name="IF1_ECOLI_Kelsic_2016",
        dms_type="singles",
        cross_validation=True,
        cross_validation_type="random",
        cross_validation_fold=0,
    )
    return ProteinGym(config)


@pytest.fixture
def proteingym_dataset_cv_multiples():
    """Create a ProteinGym cross-validation multiples dataset fixture for testing.

    Returns:
        A ProteinGym cross-validation multiples dataset.
    """
    config = ProteinGymConfig(
        name="proteingym",
        modality="sequence",
        seed=51505,
        train_ratio=0.79968,
        validation_frac=0.0,
        test_ratio=0.20032,
        split_type="random",
        problem_type="regression",
        dms_name="CAPSD_AAV2S_Sinai_2021",
        dms_type="multiples",
        cross_validation=True,
        cross_validation_type="random",
        cross_validation_fold=0,
    )
    return ProteinGym(config)


@pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"),
    reason="HF_TOKEN not set; required for gated ProteinGym dataset access",
)
class TestProteinGymDataset:
    """Test class for ProteinGym dataset functionality.

    These tests download the real gated ProteinGym dataset and are skipped when
    `HF_TOKEN` is absent (e.g. local runs or forks without the secret).
    """

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
        """Test that cross-validation singles dataset splits have correct sizes."""
        assert len(proteingym_dataset_cv_singles.train_dataset) == 1070, (
            f"Train dataset should have 1070 samples, got "
            f"{len(proteingym_dataset_cv_singles.train_dataset)}"
        )
        assert len(proteingym_dataset_cv_singles.test_dataset) == 297, (
            f"Test dataset should have 297 samples, got "
            f"{len(proteingym_dataset_cv_singles.test_dataset)}"
        )
        assert len(proteingym_dataset_cv_singles.validation_dataset) == 0, (
            f"Validation dataset should have 0 samples, got "
            f"{len(proteingym_dataset_cv_singles.validation_dataset)}"
        )
        assert len(proteingym_dataset_cv_singles.candidate_pool) == 0, (
            f"Candidate pool should have 0 samples, got "
            f"{len(proteingym_dataset_cv_singles.candidate_pool)}"
        )

    def test_cross_validation_singles_label_statistics(self, proteingym_dataset_cv_singles):
        """Test that cross-validation singles dataset label statistics match expected values."""
        train_mean = np.mean(proteingym_dataset_cv_singles.train_dataset.labels)
        test_mean = np.mean(proteingym_dataset_cv_singles.test_dataset.labels)

        assert np.isclose(train_mean, 0.790617), (
            f"Train dataset mean should be ~0.790617, got {train_mean}"
        )
        assert np.isclose(test_mean, 0.799398), (
            f"Test dataset mean should be ~0.799398, got {test_mean}"
        )

    def test_cross_validation_multiples_split_sizes(self, proteingym_dataset_cv_multiples):
        """Test that cross-validation multiples dataset splits have correct sizes."""
        assert len(proteingym_dataset_cv_multiples.train_dataset) == 33849, (
            f"Train dataset should have 33849 samples, got "
            f"{len(proteingym_dataset_cv_multiples.train_dataset)}"
        )
        assert len(proteingym_dataset_cv_multiples.test_dataset) == 8479, (
            f"Test dataset should have 8479 samples, got "
            f"{len(proteingym_dataset_cv_multiples.test_dataset)}"
        )
        assert len(proteingym_dataset_cv_multiples.validation_dataset) == 0, (
            f"Validation dataset should have 0 samples, got "
            f"{len(proteingym_dataset_cv_multiples.validation_dataset)}"
        )
        assert len(proteingym_dataset_cv_multiples.candidate_pool) == 0, (
            f"Candidate pool should have 0 samples, got "
            f"{len(proteingym_dataset_cv_multiples.candidate_pool)}"
        )

    def test_cross_validation_multiples_label_statistics(self, proteingym_dataset_cv_multiples):
        """Test that cross-validation multiples dataset label statistics match expected values."""
        train_mean = np.mean(proteingym_dataset_cv_multiples.train_dataset.labels)
        test_mean = np.mean(proteingym_dataset_cv_multiples.test_dataset.labels)

        assert np.isclose(train_mean, -1.227048), (
            f"Train dataset mean should be ~-1.227048, got {train_mean}"
        )
        assert np.isclose(test_mean, -1.221083), (
            f"Test dataset mean should be ~-1.221083, got {test_mean}"
        )


def _synthetic_singles_dataframe(n: int = 20) -> pd.DataFrame:
    """Build a synthetic ProteinGym "singles" CSV table with distinct fold columns.

    The contiguous and random fold assignments are deliberately different so a
    test can confirm the contiguous fold column (not another) is the one used.
    """
    return pd.DataFrame({
        "mutated_sequence": ["MKL" + "A" * (i + 1) for i in range(n)],
        "DMS_score": [float(i) for i in range(n)],
        "mutant": [f"A{i}G" for i in range(n)],
        "fold_random_5": [(i + 2) % 5 for i in range(n)],
        "fold_modulo_5": [(i + 1) % 5 for i in range(n)],
        "fold_contiguous_5": [i % 5 for i in range(n)],
    })


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


class TestProteinGymContiguousSplit:
    """The contiguous cross-validation split uses the contiguous fold column."""

    def test_contiguous_split_selects_correct_fold(self, monkeypatch):
        """Regression test: contiguous CV previously raised KeyError because
        load_dataset stored the fold under "fold_contiguous_id" while the split
        looked up "contiguous_fold_id". This drives the real load_dataset (over a
        synthetic CSV) so the held-out split must come from the contiguous fold.
        """
        monkeypatch.setenv("HF_TOKEN", "test-token")
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
        df = _synthetic_singles_dataframe(n=20)

        with (
            patch("alf_tools.datasets.proteingym.hf_hub_download"),
            patch("alf_tools.datasets.proteingym.pd.read_csv", return_value=df),
        ):
            dataset = ProteinGym(config)

        held_out = dataset.test_dataset.candidates + dataset.candidate_pool.candidates
        assert len(held_out) > 0
        assert all(c.features["contiguous_fold_id"] == fold for c in held_out)
        assert all(
            c.features["contiguous_fold_id"] != fold
            for c in dataset.train_dataset.candidates + dataset.validation_dataset.candidates
        )
