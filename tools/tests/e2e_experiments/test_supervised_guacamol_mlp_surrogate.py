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

import numpy as np
import pandas as pd
import pytest
import torch
from alf_core import (
    FileStateLogger,
    SupervisedTask,
    Surrogate,
    TerminalStateLogger,
)
from alf_tools.datasets import GuacaMol, GuacaMolConfig
from alf_tools.datasets.guacamol import FILENAME_ALL
from alf_tools.models import MLPModel, MLPModelConfig, MLPTrainConfig

FIXTURES = Path(__file__).parent.parent / "fixtures" / "guacamol"
LARGE_FIXTURE = FIXTURES / "large.smiles"


@pytest.fixture
def set_seed():
    """Fixture to set random seeds for reproducible tests.

    Returns:
        A function that sets the random seeds.
    """

    def _set_seed(seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return _set_seed


@pytest.fixture
def guacamol_data_path(tmp_path):
    """Fixture that prepares a local DATAPATH directory containing the GuacaMol corpus.

    Copies the large.smiles fixture into a temp directory using the filename expected by
    the GuacaMol module so that no network download is triggered.

    Returns:
        Path to the prepared data directory.
    """
    data_dir = tmp_path / "guacamol_data"
    data_dir.mkdir()
    shutil.copy(LARGE_FIXTURE, data_dir / FILENAME_ALL)
    return data_dir


class BaseTestSupervisedGuacaMol(abc.ABC):
    """Abstract base class for supervised GuacaMol MLP surrogate E2E tests."""

    @property
    @abc.abstractmethod
    def target_property(self) -> str:
        """The GuacaMol target property used as the label."""
        ...

    @property
    def expected_metrics(self) -> dict:
        """Expected metric values for assertions (filled in Task 7)."""
        return {}

    def test_supervised_experiment(self, set_seed, guacamol_data_path, tmp_path):
        """Test the complete supervised GuacaMol MLP experiment pipeline.

        This test verifies that the supervised learning pipeline works correctly
        and produces expected metrics for the GuacaMol dataset using an MLP surrogate model.
        """
        # Set seed for reproducible results
        set_seed(42)

        config = GuacaMolConfig(
            name="guacamol",
            modality="sequence",
            seed=51505,
            train_ratio=0.1,
            test_ratio=0.2,
            validation_frac=0.5,
            target_property=self.target_property,
            split_mode="random",
            computed_properties=["MolWt", "MolLogP", "QED"],
            max_molecules=500,
            data_dir=guacamol_data_path,
        )
        dataset = GuacaMol(config)

        # Build surrogate
        mlp_model = MLPModel(
            model_config=MLPModelConfig(hidden_dims=[64, 32]),
            train_config=MLPTrainConfig(num_epochs=5, log_frequency=5),
        )
        surrogate_model = Surrogate(model=mlp_model)

        # Use pytest's tmp_path for temporary directory
        save_path = tmp_path / f"supervised_guacamol_{self.target_property.lower()}_mlp"
        save_path.mkdir()

        metrics_logger = TerminalStateLogger()
        file_logger = FileStateLogger(output_path=save_path)
        state_loggers = [metrics_logger, file_logger]

        # Create and run the supervised task
        task = SupervisedTask()
        state = task.setup(dataset=dataset, surrogate=surrogate_model)
        task.run(
            state=state,
            state_loggers=state_loggers,
        )

        # Load and verify results
        metrics_file = save_path / "metrics.csv"
        assert metrics_file.exists(), "Metrics file should be created"

        metrics = pd.read_csv(metrics_file)

        # Test surrogate performance metrics
        self._assert_surrogate_metrics(metrics)

        # Test dataset metrics
        self._assert_dataset_metrics(metrics)

        # Assert training_history was populated for the supervised round
        num_epochs = mlp_model.train_config.num_epochs
        assert len(state.round_metrics.training_history) == num_epochs, (
            f"Expected {num_epochs} epoch entries in training_history, "
            f"got {len(state.round_metrics.training_history)}"
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
        # Train split must have candidates
        train_num_col = "dataset/num_train"
        assert train_num_col in metrics.columns, f"Column '{train_num_col}' not found in metrics"
        num_train = metrics[train_num_col].iloc[0]
        assert num_train > 0, f"Expected non-zero train candidates, got {num_train}"

        # Test split must have candidates
        test_num_col = "dataset/num_test"
        assert test_num_col in metrics.columns, f"Column '{test_num_col}' not found in metrics"
        num_test = metrics[test_num_col].iloc[0]
        assert num_test > 0, f"Expected non-zero test candidates, got {num_test}"

        # Label means must be finite floats
        for mean_col in ["dataset/train_mean", "dataset/test_mean"]:
            assert mean_col in metrics.columns, f"Column '{mean_col}' not found in metrics"
            mean_value = metrics[mean_col].iloc[0]
            assert math.isfinite(float(mean_value)), (
                f"Dataset metric {mean_col} should be finite, got {mean_value}"
            )

        if self.expected_metrics:
            dataset_expected = self.expected_metrics.get("dataset", {})
            for metric_name, expected_value in dataset_expected.items():
                actual_value = metrics[f"dataset/{metric_name}"].iloc[0]
                assert np.isclose(actual_value, expected_value, atol=1e-4), (
                    f"Dataset metric {metric_name} mismatch: expected {expected_value}, "
                    f"got {actual_value}"
                )


class TestSupervisedQED(BaseTestSupervisedGuacaMol):
    """Tests a supervised experiment with an MLP surrogate on the GuacaMol QED property."""

    @property
    def target_property(self) -> str:
        """Quantitative Estimate of Druglikeness (0-1 scale)."""
        return "QED"

    @property
    def expected_metrics(self) -> dict:
        """Expected metrics for testing the surrogate model and dataset.

        Returns:
            dict: Surrogate model metrics (MSE, Spearman, Pearson, pairwise cross-entropy)
            and dataset metrics (split sizes and test mean) for the GuacaMol QED property
        """
        return {
            "surrogate": {
                "test_mse": 0.2872,
                "test_spearman": -0.479,
                "test_pearson": -0.3436,
                "test_pairwise_xent": 0.3443,
            },
            "dataset": {
                "num_train": 25,
                "num_test": 100,
                "train_mean": 0.6164,
                "test_mean": 0.5356,
            },
        }


class TestSupervisedMolWt(BaseTestSupervisedGuacaMol):
    """Tests a supervised experiment with an MLP surrogate on the GuacaMol MolWt property."""

    @property
    def target_property(self) -> str:
        """Average molecular weight (Da)."""
        return "MolWt"

    @property
    def expected_metrics(self) -> dict:
        """Expected metrics for testing the surrogate model and dataset.

        Returns:
            dict: Surrogate model metrics (MSE, Spearman, Pearson, pairwise cross-entropy)
            and dataset metrics (split sizes and test mean) for the GuacaMol MolWt property
        """
        return {
            "surrogate": {
                "test_mse": 17558.1114,
                "test_spearman": -0.5782,
                "test_pearson": -0.4518,
                "test_pairwise_xent": 0.3446,
            },
            "dataset": {
                "num_train": 25,
                "num_test": 100,
                "train_mean": 159.2121,
                "test_mean": 115.3263,
            },
        }


class TestSupervisedMolLogP(BaseTestSupervisedGuacaMol):
    """Tests a supervised experiment with an MLP surrogate on the GuacaMol MolLogP property."""

    @property
    def target_property(self) -> str:
        """Wildman-Crippen octanol-water partition coefficient."""
        return "MolLogP"

    @property
    def expected_metrics(self) -> dict:
        """Expected metrics for testing the surrogate model and dataset.

        Returns:
            dict: Surrogate model metrics (MSE, Spearman, Pearson, pairwise cross-entropy)
            and dataset metrics (split sizes and test mean) for the GuacaMol MolLogP property
        """
        return {
            "surrogate": {
                "test_mse": 2.8017,
                "test_spearman": 0.035,
                "test_pearson": -0.0162,
                "test_pairwise_xent": 0.3432,
            },
            "dataset": {
                "num_train": 25,
                "num_test": 100,
                "train_mean": 1.8406,
                "test_mean": 1.1909,
            },
        }
