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
    DatasetSearch,
    DesignTask,
    FileStateLogger,
    Optimizer,
    Oracle,
    Surrogate,
    TerminalStateLogger,
)
from alf_tools.datasets import GuacaMol, GuacaMolConfig
from alf_tools.datasets.guacamol import FILENAME_ALL
from alf_tools.optimizer.acquisition_functions import Greedy

FIXTURES = Path(__file__).parent.parent / "fixtures" / "guacamol"
LARGE_FIXTURE = FIXTURES / "large.smiles"


@pytest.fixture
def guacamol_data_path(tmp_path):
    """Fixture that prepares a local DATAPATH directory containing the GuacaMol corpus.

    Copies the large.smiles fixture into a temp directory using the filename expected by
    the GuacaMol module (combined-corpus mode) so that no network download is triggered.

    Returns:
        Path to the prepared data directory.
    """
    data_dir = tmp_path / "guacamol_data"
    data_dir.mkdir()
    shutil.copy(LARGE_FIXTURE, data_dir / FILENAME_ALL)
    return data_dir


class BaseTestDesignGuacaMol(abc.ABC):
    """Abstract base class for design GuacaMol random surrogate E2E tests."""

    # ------------------------------------------------------------------ #
    # Abstract / overridable interface                                      #
    # ------------------------------------------------------------------ #

    @property
    @abc.abstractmethod
    def target_property(self) -> str:
        """The GuacaMol target property used as the label."""
        ...

    @property
    def expected_metrics(self) -> dict:
        """Expected metric values for assertions (filled in after first run with seed 42)."""
        return {}

    # ------------------------------------------------------------------ #
    # Main test                                                            #
    # ------------------------------------------------------------------ #

    def test_design_experiment(self, random_model, guacamol_data_path, tmp_path):
        """Test the complete design GuacaMol random surrogate experiment pipeline.

        This test verifies that the active-learning design pipeline works correctly
        with split_mode="low_vs_high" and produces expected metrics for the
        GuacaMol dataset using a random surrogate model.
        """
        n_rounds = 5
        batch_size = 50

        # Build dataset — patch DATAPATH so the module reads from the local fixture
        config = GuacaMolConfig(
            name="guacamol",
            modality="sequence",
            seed=51505,
            target_property=self.target_property,
            split_mode="low_vs_high",
            computed_properties=["MolWt", "MolLogP", "QED"],
            max_molecules=500,
            train_ratio=0.1,
            validation_frac=0.1,
            test_ratio=0.2,
        )
        with patch("alf_tools.datasets.guacamol.DATAPATH", guacamol_data_path):
            dataset = GuacaMol(config)

        surrogate_model = Surrogate(model=random_model)

        # Build optimizer
        acquisition_fn = Greedy()
        search_fn = DatasetSearch()
        optimizer = Optimizer(acquisition_fn=acquisition_fn, search_fn=search_fn)

        # Build oracle
        oracle = Oracle(scorer=dataset)

        # Use pytest's tmp_path for temporary directory
        save_path = tmp_path / f"design_guacamol_{self.target_property.lower()}_random"
        save_path.mkdir()

        metrics_logger = TerminalStateLogger()
        file_logger = FileStateLogger(output_path=save_path)
        state_loggers = [metrics_logger, file_logger]

        # Create and run the design task
        task = DesignTask(num_acq_rounds=n_rounds, acq_batch_size=batch_size)
        state = task.setup(dataset=dataset, surrogate=surrogate_model)

        # Capture the initial pool size and initial training labels before any acquisitions
        initial_pool_size = len(state.dataset.candidate_pool)
        initial_train_labels = state.dataset.train_dataset.labels.copy()

        task.run(
            state=state,
            state_loggers=state_loggers,
            optimizer=optimizer,
            oracle=oracle,
        )

        # Load and verify results
        metrics_file = save_path / "metrics.csv"
        assert metrics_file.exists(), "Metrics file should be created"

        metrics = pd.read_csv(metrics_file)

        # Verify pool shrinks by batch_size each round
        self._assert_pool_shrinks(metrics, initial_pool_size, batch_size, n_rounds)

        # Verify low_vs_high split puts low-scoring molecules in initial training set
        self._assert_low_vs_high_split(dataset, initial_train_labels)

        # Verify dataset metrics are sensible
        self._assert_dataset_metrics(metrics)

        # Verify surrogate metrics are sensible finite floats
        self._assert_surrogate_metrics(metrics)

    # ------------------------------------------------------------------ #
    # Unique validations                                                   #
    # ------------------------------------------------------------------ #

    def _assert_pool_shrinks(
        self,
        metrics: pd.DataFrame,
        initial_pool_size: int,
        batch_size: int,
        n_rounds: int,
    ) -> None:
        """Assert that the candidate pool shrinks by batch_size after each acquisition round.

        The design task logs metrics at:
          - round 0  (initial_train_round): pool still at full initial size
          - rounds 1..n_rounds: pool decrements by batch_size each round

        Args:
            metrics: DataFrame loaded from metrics.csv.
            initial_pool_size: Pool size captured immediately after task.setup().
            batch_size: Expected decrement per acquisition round.
            n_rounds: Total number of acquisition rounds.
        """
        pool_col = "dataset/num_candidate_pool"
        assert pool_col in metrics.columns, f"Column '{pool_col}' not found in metrics"

        pool_sizes = metrics[pool_col].tolist()
        # metrics.csv has n_rounds + 1 rows (initial train + n_rounds acquisition rounds)
        assert len(pool_sizes) == n_rounds + 1, (
            f"Expected {n_rounds + 1} rows in metrics.csv, got {len(pool_sizes)}"
        )

        # Row 0 corresponds to the initial training round — pool is untouched
        assert pool_sizes[0] == initial_pool_size, (
            f"Pool size at round 0 should equal initial pool size {initial_pool_size}, "
            f"got {pool_sizes[0]}"
        )

        # Rows 1..n_rounds: pool should decrease by batch_size each round
        for i in range(1, n_rounds + 1):
            expected = initial_pool_size - i * batch_size
            assert pool_sizes[i] == expected, (
                f"Pool size at round {i} should be {expected} "
                f"(initial {initial_pool_size} - {i}*{batch_size}), "
                f"got {pool_sizes[i]}"
            )

    def _assert_low_vs_high_split(
        self, dataset: GuacaMol, initial_train_labels: np.ndarray
    ) -> None:
        """Assert that split_mode='low_vs_high' puts only low-scoring molecules in train.

        Uses the training labels captured before the design task ran (acquisitions grow
        the training set with high-scoring candidates, so post-run labels are not valid
        for this check). Verifies that the maximum initial training label is below the
        median of all labels in the raw dataset — confirming that training candidates
        start exclusively from the low-scoring half of the corpus.

        Args:
            dataset: The GuacaMol dataset loaded with split_mode='low_vs_high'.
            initial_train_labels: Training labels captured before task.run() was called.
        """
        assert dataset._raw_dataset is not None, "Dataset must be loaded before checking split"

        all_labels = dataset._raw_dataset.labels

        assert len(initial_train_labels) > 0, "Initial training set must be non-empty"

        all_median = float(np.median(all_labels))
        train_max = float(np.max(initial_train_labels))

        assert train_max < all_median, (
            f"low_vs_high split: max initial train label ({train_max:.4f}) should be below "
            f"the median of all labels ({all_median:.4f}), indicating training candidates "
            "are drawn from the low-scoring half of the corpus."
        )

    # ------------------------------------------------------------------ #
    # Shared metric assertions                                             #
    # ------------------------------------------------------------------ #

    def _assert_dataset_metrics(self, metrics: pd.DataFrame) -> None:
        """Assert dataset split sizes and label means are valid finite floats."""
        # Train and test splits must be non-empty
        for split_name in ["train", "test"]:
            num_col = f"dataset/num_{split_name}"
            assert num_col in metrics.columns, f"Column '{num_col}' not found in metrics"
            num_candidates = metrics[num_col].iloc[0]
            assert num_candidates > 0, (
                f"Expected non-zero {split_name} candidates, got {num_candidates}"
            )

        # Label means must be finite floats
        for mean_col in ["dataset/train_mean", "dataset/test_mean"]:
            assert mean_col in metrics.columns, f"Column '{mean_col}' not found in metrics"
            mean_value = metrics[mean_col].iloc[0]
            assert math.isfinite(float(mean_value)), (
                f"Dataset metric {mean_col} should be finite, got {mean_value}"
            )

        # Check expected values if provided
        if self.expected_metrics:
            dataset_expected = self.expected_metrics.get("dataset", {})
            for metric_name, expected_value in dataset_expected.items():
                actual_value = metrics[f"dataset/{metric_name}"].iloc[0]
                assert np.isclose(actual_value, expected_value, atol=1e-4), (
                    f"Dataset metric {metric_name} mismatch: expected {expected_value}, "
                    f"got {actual_value}"
                )

    def _assert_surrogate_metrics(self, metrics: pd.DataFrame) -> None:
        """Assert surrogate model performance metrics are finite floats."""
        for metric_name in ["test_mse", "test_spearman", "test_pearson", "test_pairwise_xent"]:
            col = f"surrogate/{metric_name}"
            assert col in metrics.columns, f"Column '{col}' not found in metrics"
            # Check all rounds
            for val in metrics[col].tolist():
                assert math.isfinite(float(val)), (
                    f"Surrogate metric {metric_name} should be finite, got {val}"
                )

        # Check expected values if provided
        if self.expected_metrics:
            surrogate_expected = self.expected_metrics.get("surrogate", {})
            for metric_name, expected_values in surrogate_expected.items():
                actual_values = metrics[f"surrogate/{metric_name}"].tolist()
                assert np.isclose(actual_values, expected_values, atol=1e-4).all(), (
                    f"Surrogate metric {metric_name} mismatch: expected {expected_values}, "
                    f"got {actual_values}"
                )


# ------------------------------------------------------------------ #
# Concrete test classes                                               #
# ------------------------------------------------------------------ #


class TestDesignQED(BaseTestDesignGuacaMol):
    """Tests a design experiment with a random surrogate on the GuacaMol QED property."""

    @property
    def target_property(self) -> str:
        return "QED"

    @property
    def expected_metrics(self) -> dict:
        return {
            "surrogate": {
                # 6 rounds: initial train round + 5 acquisition rounds
                "test_mse": [
                    1.2979, 1.5068, 1.0928, 1.2605, 1.4744, 1.3956,
                ],
                "test_spearman": [
                    -0.1560, -0.1140, -0.1718,  0.0415,  0.0755, -0.0064,
                ],
                "test_pearson": [
                    -0.1720, -0.0779, -0.1431, -0.0012,  0.1047,  0.0067,
                ],
                "test_pairwise_xent": [
                    0.4795, 0.4536, 0.4887, 0.4400, 0.4234, 0.4394,
                ],
            },
            "dataset": {
                # Values at round 0 (initial train round)
                "num_train": 45,
                "num_test": 100,
                "train_mean": 0.3806,
                "test_mean": 0.5474,
            },
        }


class TestDesignMolWt(BaseTestDesignGuacaMol):
    """Tests a design experiment with a random surrogate on the GuacaMol MolWt property."""

    @property
    def target_property(self) -> str:
        return "MolWt"

    @property
    def expected_metrics(self) -> dict:
        return {
            "surrogate": {
                "test_mse": [
                    20517.1696, 20522.9172, 20437.2304, 20457.3037, 20485.6165, 20500.9615,
                ],
                "test_spearman": [
                    -0.1616, -0.1200, -0.1703,  0.0438,  0.0773, -0.0233,
                ],
                "test_pearson": [
                    -0.1814, -0.0894, -0.1447,  0.0116,  0.0853, -0.0060,
                ],
                "test_pairwise_xent": [
                    0.4811, 0.4550, 0.4889, 0.4396, 0.4223, 0.4445,
                ],
            },
            "dataset": {
                "num_train": 45,
                "num_test": 100,
                "train_mean": 28.9164,
                "test_mean": 129.5674,
            },
        }


class TestDesignMolLogP(BaseTestDesignGuacaMol):
    """Tests a design experiment with a random surrogate on the GuacaMol MolLogP property."""

    @property
    def target_property(self) -> str:
        return "MolLogP"

    @property
    def expected_metrics(self) -> dict:
        return {
            "surrogate": {
                "test_mse": [
                    4.9274, 4.9923, 4.1756, 4.2621, 4.5424, 4.6876,
                ],
                "test_spearman": [
                    -0.1583, -0.1214, -0.1699,  0.0354,  0.0787, -0.0176,
                ],
                "test_pearson": [
                    -0.1958, -0.0529, -0.1888,  0.0004,  0.1047,  0.0070,
                ],
                "test_pairwise_xent": [
                    0.4801, 0.4553, 0.4882, 0.4417, 0.4219, 0.4424,
                ],
            },
            "dataset": {
                "num_train": 45,
                "num_test": 100,
                "train_mean": -1.1874,
                "test_mean": 1.4895,
            },
        }
