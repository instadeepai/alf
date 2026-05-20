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
import pandas as pd
import pytest
import torch
from alf_core import (
    BaseDatasetConfig,
    FileStateLogger,
    Oracle,
    SupervisedTask,
    Surrogate,
    TerminalStateLogger,
)
from alf_core.utils.enums import ProblemType
from alf_tools.datasets import GFP
from alf_tools.models import CNNModel, CNNModelConfig, CNNTrainConfig


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
def gfp_dataset():
    """Fixture to create a GFP dataset for testing.

    Returns:
        A GFP dataset.
    """
    config = BaseDatasetConfig(
        name="gfp",
        modality="sequence",
        seed=51505,
        train_ratio=0.1,
        test_ratio=0.2,
        validation_frac=0.5,
        split_type="random",
        problem_type="regression",
    )
    return GFP(config)


@pytest.fixture
def surrogate_model():
    """Fixture to create a surrogate model for testing.

    Returns:
        A surrogate model.
    """
    return Surrogate(
        model=CNNModel(
            model_config=CNNModelConfig(problem_type=ProblemType.REGRESSION),
            train_config=CNNTrainConfig(num_epochs=10),
        )
    )


@pytest.fixture
def oracle(gfp_dataset):
    """Fixture to create optimizer components.

    Returns:
        An Oracle.
    """
    oracle = Oracle(scorer=gfp_dataset)
    return oracle


@pytest.fixture
def expected_metrics():
    """Fixture containing expected metric values for assertions.

    Returns:
        Expected metric values for assertions.
    """
    return {
        "surrogate": {
            "test_mse": 2.82286,
            "test_spearman": 0.08317,
            "test_pearson": 0.18264,
            "test_pairwise_xent": 0.34481,
        },
        "dataset": {
            "num_train": 50.00000,
            "train_mean": 3.14036,
            "num_validation": 50.00000,
            "validation_mean": 3.06438,
            "num_test": 200.00000,
            "test_mean": 3.14817,
            "num_candidate_pool": 700.00000,
        },
    }


class TestSupervised:
    """Tests a supervised experiment with a CNN surrogate model on the GFP dataset."""

    def test_supervised_gfp_cnn_experiment(
        self, set_seed, gfp_dataset, surrogate_model, expected_metrics, tmp_path
    ):
        """Test the complete supervised GFP CNN experiment pipeline.

        This test verifies that the supervised learning pipeline works correctly
        and produces expected metrics for the GFP dataset using a CNN surrogate model.
        """
        # Set seed for reproducible results
        set_seed(42)

        # Use pytest's tmp_path for temporary directory
        save_path = tmp_path / "supervised_gfp_cnn"
        save_path.mkdir()

        metrics_logger = TerminalStateLogger()
        file_logger = FileStateLogger(output_path=save_path)
        state_loggers = [metrics_logger, file_logger]

        # Create and run the supervised task
        task = SupervisedTask()
        state = task.setup(dataset=gfp_dataset, surrogate=surrogate_model)
        task.run(
            state=state,
            state_loggers=state_loggers,
        )

        # Load and verify results
        metrics_file = save_path / "metrics.csv"
        assert metrics_file.exists(), "Metrics file should be created"

        metrics = pd.read_csv(metrics_file)

        # Test surrogate performance metrics
        self._assert_surrogate_metrics(metrics, expected_metrics["surrogate"])

        # Test dataset metrics
        self._assert_dataset_metrics(metrics, expected_metrics["dataset"])

        # Assert training_history was populated for the supervised round
        num_epochs = surrogate_model.model.train_config.num_epochs
        assert len(state.round_metrics.training_history) == num_epochs, (
            f"Expected {num_epochs} epoch entries in training_history, "
            f"got {len(state.round_metrics.training_history)}"
        )

    def _assert_surrogate_metrics(self, metrics: pd.DataFrame, expected: dict):
        """Assert surrogate model performance metrics."""
        for metric_name, expected_value in expected.items():
            actual_value = metrics[f"surrogate/{metric_name}"].iloc[0]
            assert np.isclose(actual_value, expected_value, atol=1e-4), (
                f"Surrogate metric {metric_name} mismatch: expected {expected_value}, "
                f"got {actual_value}"
            )

    def _assert_dataset_metrics(self, metrics: pd.DataFrame, expected: dict):
        """Assert dataset metrics."""
        for metric_name, expected_value in expected.items():
            actual_value = metrics[f"dataset/{metric_name}"].iloc[0]
            assert np.isclose(actual_value, expected_value, atol=1e-4), (
                f"Dataset metric {metric_name} mismatch: expected {expected_value}, "
                f"got {actual_value}"
            )
