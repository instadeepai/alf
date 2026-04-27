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
import math
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from alf_core import (
    FileStateLogger,
    Surrogate,
    TerminalStateLogger,
    ZeroShotTask,
)
from alf_tools.datasets import GuacaMol, GuacaMolConfig
from alf_tools.datasets.guacamol import FILENAME_TEST, FILENAME_TRAIN, FILENAME_VALID

FIXTURES = Path(__file__).parent.parent / "fixtures" / "guacamol"
LARGE_FIXTURE = FIXTURES / "large.smiles"


@pytest.fixture
def guacamol_data_path(tmp_path):
    """Fixture that prepares a local DATAPATH directory with all three paper-split files.

    Copies the large.smiles fixture into a temp directory using the filenames expected by
    the GuacaMol paper-split mode so that no network download is triggered.

    Returns:
        Path to the prepared data directory.
    """
    data_dir = tmp_path / "guacamol_data"
    data_dir.mkdir()
    for filename in [FILENAME_TRAIN, FILENAME_VALID, FILENAME_TEST]:
        shutil.copy(LARGE_FIXTURE, data_dir / filename)
    return data_dir


class BaseTestZeroShotGuacaMol(abc.ABC):
    """Abstract base class for zero-shot GuacaMol random surrogate E2E tests."""

    @property
    @abc.abstractmethod
    def target_property(self) -> str:
        """The GuacaMol target property used as the label."""
        ...

    @property
    def expected_metrics(self) -> dict:
        """Expected metric values for assertions (filled after first run with seed 42)."""
        return {}

    def test_zeroshot_experiment(self, random_model, guacamol_data_path, tmp_path):
        """Test the complete zero-shot GuacaMol random surrogate experiment pipeline.

        This test verifies that the zero-shot pipeline works correctly with the paper
        split mode and produces expected metrics for the GuacaMol dataset using a
        random surrogate model.
        """
        # Set seed for reproducible results
        np.random.seed(42)

        # Build dataset — patch DATAPATH so the module reads from the local fixture
        config = GuacaMolConfig(
            name="guacamol",
            modality="sequence",
            seed=51505,
            target_property=self.target_property,
            split_mode="paper",
            computed_properties=["MolWt", "MolLogP", "QED"],
            max_molecules=200,
            train_ratio=0.0,
            validation_frac=0.0,
            test_ratio=1.0,
        )
        with patch("alf_tools.datasets.guacamol.DATAPATH", guacamol_data_path):
            dataset = GuacaMol(config)

        surrogate_model = Surrogate(model=random_model)

        # Use pytest's tmp_path for temporary directory
        save_path = tmp_path / f"zeroshot_guacamol_{self.target_property.lower()}_random"
        save_path.mkdir()

        metrics_logger = TerminalStateLogger()
        file_logger = FileStateLogger(output_path=save_path)
        state_loggers = [metrics_logger, file_logger]

        # Create and run the zero-shot task
        task = ZeroShotTask()
        state = task.setup(dataset=dataset, surrogate=surrogate_model)
        task.run(state=state, state_loggers=state_loggers)

        # Load and verify results
        metrics_file = save_path / "metrics.csv"
        assert metrics_file.exists(), "Metrics file should be created"

        metrics = pd.read_csv(metrics_file)

        # Assert paper split boundaries (all three splits non-empty)
        self._assert_paper_split_boundaries(dataset)

        # Assert split feature tags are present on all candidates
        self._assert_split_feature_tags(dataset)

        # Assert surrogate metrics
        self._assert_surrogate_metrics(metrics)

        # Assert dataset metrics
        self._assert_dataset_metrics(metrics)

    def _assert_paper_split_boundaries(self, dataset: GuacaMol):
        """Assert that all three paper splits are non-empty.

        Args:
            dataset: The GuacaMol dataset with paper splits loaded.
        """
        train_candidates = dataset.train_dataset.candidates
        validation_candidates = dataset.validation_dataset.candidates
        test_candidates = dataset.test_dataset.candidates

        assert len(train_candidates) > 0, (
            "Expected non-empty train split from paper file, got 0 candidates"
        )
        assert len(validation_candidates) > 0, (
            "Expected non-empty validation split from paper file, got 0 candidates"
        )
        assert len(test_candidates) > 0, (
            "Expected non-empty test split from paper file, got 0 candidates"
        )

    def _assert_split_feature_tags(self, dataset: GuacaMol):
        """Assert that all candidates have a 'split' key in their features dict.

        Args:
            dataset: The GuacaMol dataset with paper splits loaded.
        """
        assert dataset._raw_dataset is not None, "Dataset must be loaded before checking features"

        valid_tags = {"train", "valid", "test"}
        for i, candidate in enumerate(dataset._raw_dataset.candidates):
            assert "split" in candidate.features, (
                f"Candidate {i} is missing 'split' key in features: {candidate.features}"
            )
            assert candidate.features["split"] in valid_tags, (
                f"Candidate {i} has unexpected split tag '{candidate.features['split']}'; "
                f"expected one of {valid_tags}"
            )

    def _assert_surrogate_metrics(self, metrics: pd.DataFrame):
        """Assert surrogate model performance metrics are finite floats."""
        for metric_name in ["test_mse", "test_spearman", "test_pearson", "test_pairwise_xent"]:
            col = f"surrogate/{metric_name}"
            assert col in metrics.columns, f"Column '{col}' not found in metrics"
            actual_value = metrics[col].iloc[0]
            assert math.isfinite(float(actual_value)), (
                f"Surrogate metric {metric_name} should be finite, got {actual_value}"
            )

        if self.expected_metrics:
            surrogate_expected = self.expected_metrics.get("surrogate", {})
            for metric_name, expected_value in surrogate_expected.items():
                actual_value = metrics[f"surrogate/{metric_name}"].iloc[0]
                assert np.isclose(actual_value, expected_value, atol=1e-4), (
                    f"Surrogate metric {metric_name} mismatch: expected {expected_value}, "
                    f"got {actual_value}"
                )

    def _assert_dataset_metrics(self, metrics: pd.DataFrame):
        """Assert dataset split sizes and label means are valid."""
        # Test split must have candidates (paper split uses test for ZeroShotTask evaluation)
        test_num_col = "dataset/num_test"
        assert test_num_col in metrics.columns, f"Column '{test_num_col}' not found in metrics"
        num_test = metrics[test_num_col].iloc[0]
        assert num_test > 0, f"Expected non-zero test candidates, got {num_test}"

        # Test mean must be finite
        test_mean_col = "dataset/test_mean"
        if test_mean_col in metrics.columns:
            mean_value = metrics[test_mean_col].iloc[0]
            assert math.isfinite(float(mean_value)), (
                f"Dataset metric {test_mean_col} should be finite, got {mean_value}"
            )

        if self.expected_metrics:
            dataset_expected = self.expected_metrics.get("dataset", {})
            for metric_name, expected_value in dataset_expected.items():
                actual_value = metrics[f"dataset/{metric_name}"].iloc[0]
                assert np.isclose(actual_value, expected_value, atol=1e-4), (
                    f"Dataset metric {metric_name} mismatch: expected {expected_value}, "
                    f"got {actual_value}"
                )


class TestZeroShotQED(BaseTestZeroShotGuacaMol):
    """Tests a zero-shot experiment with a random surrogate on the GuacaMol QED property."""

    @property
    def target_property(self) -> str:
        return "QED"

    @property
    def expected_metrics(self) -> dict:
        return {
            "surrogate": {
                "test_mse": 1.2160,
                "test_spearman": -0.1300,
                "test_pearson": -0.1474,
                "test_pairwise_xent": 0.4675,
            },
            "dataset": {
                "num_train": 200,
                "num_validation": 200,
                "num_test": 200,
                "test_mean": 0.5096,
            },
        }


class TestZeroShotMolWt(BaseTestZeroShotGuacaMol):
    """Tests a zero-shot experiment with a random surrogate on the GuacaMol MolWt property."""

    @property
    def target_property(self) -> str:
        return "MolWt"

    @property
    def expected_metrics(self) -> dict:
        return {
            "surrogate": {
                "test_mse": 16293.4287,
                "test_spearman": -0.1116,
                "test_pearson": -0.1135,
                "test_pairwise_xent": 0.4624,
            },
            "dataset": {
                "num_train": 200,
                "num_validation": 200,
                "num_test": 200,
                "test_mean": 111.0645,
            },
        }


class TestZeroShotMolLogP(BaseTestZeroShotGuacaMol):
    """Tests a zero-shot experiment with a random surrogate on the GuacaMol MolLogP property."""

    @property
    def target_property(self) -> str:
        return "MolLogP"

    @property
    def expected_metrics(self) -> dict:
        return {
            "surrogate": {
                "test_mse": 3.7903,
                "test_spearman": -0.0345,
                "test_pearson": -0.0578,
                "test_pairwise_xent": 0.4481,
            },
            "dataset": {
                "num_train": 200,
                "num_validation": 200,
                "num_test": 200,
                "test_mean": 0.9729,
            },
        }
