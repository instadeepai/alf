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

import numpy as np
import pytest
from alf_core import Modality
from alf_tools.datasets.proteingym import ProteinGym


@pytest.fixture
def proteingym_dataset_singles():
    """Create a ProteinGym singles dataset fixture for testing.

    Returns:
        ProteinGym: A ProteinGym singles dataset.
    """
    return ProteinGym(
        name="proteingym",
        modality="sequence",
        seed=51505,
        split_config={
            "split_ratio": {"train": 0.6, "validation_frac": 0.2, "test": 0.2},
            "split_type": "random",
        },
        dataset_config={"dms_name": "IF1_ECOLI_Kelsic_2016", "dms_type": "singles"},
    )


@pytest.fixture
def proteingym_dataset_multiples():
    """Create a ProteinGym multiples dataset fixture for testing.

    Returns:
        ProteinGym: A ProteinGym multiples dataset.
    """
    return ProteinGym(
        name="proteingym",
        modality="sequence",
        seed=51505,
        split_config={
            "split_ratio": {"train": 0.6, "validation_frac": 0.2, "test": 0.2},
            "split_type": "random",
        },
        dataset_config={"dms_name": "CAPSD_AAV2S_Sinai_2021", "dms_type": "multiples"},
    )


@pytest.fixture
def proteingym_dataset_cv_singles():
    """Create a ProteinGym cross-validation singles dataset fixture for testing.

    Returns:
        ProteinGym: A ProteinGym cross-validation singles dataset.
    """
    return ProteinGym(
        name="proteingym",
        modality="sequence",
        seed=51505,
        split_config={
            "split_ratio": {"train": 0.7827, "validation_frac": 0.0, "test": 0.2173},
            "split_type": "random",
        },
        dataset_config={
            "dms_name": "IF1_ECOLI_Kelsic_2016",
            "dms_type": "singles",
            "cross_validation": True,
            "cross_validation_type": "random",
            "cross_validation_fold": 0,
        },
    )


@pytest.fixture
def proteingym_dataset_cv_multiples():
    """Create a ProteinGym cross-validation multiples dataset fixture for testing.

    Returns:
        ProteinGym: A ProteinGym cross-validation multiples dataset.
    """
    return ProteinGym(
        name="proteingym",
        modality="sequence",
        seed=51505,
        split_config={
            "split_ratio": {"train": 0.79968, "validation_frac": 0.0, "test": 0.20032},
            "split_type": "random",
        },
        dataset_config={
            "dms_name": "CAPSD_AAV2S_Sinai_2021",
            "dms_type": "multiples",
            "cross_validation": True,
            "cross_validation_type": "random",
            "cross_validation_fold": 0,
        },
    )


class TestProteinGymDataset:
    """Test class for ProteinGym dataset functionality."""

    def test_dataset_initialization_singles(self, proteingym_dataset_singles):
        """Test that singles dataset initializes correctly."""
        assert proteingym_dataset_singles.name == "proteingym"
        assert proteingym_dataset_singles.modality == Modality.SEQUENCE
        assert proteingym_dataset_singles.seed == 51505

    def test_dataset_initialization_multiples(self, proteingym_dataset_multiples):
        """Test that multiples dataset initializes correctly."""
        assert proteingym_dataset_multiples.name == "proteingym"
        assert proteingym_dataset_multiples.modality == Modality.SEQUENCE
        assert proteingym_dataset_multiples.seed == 51505

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
