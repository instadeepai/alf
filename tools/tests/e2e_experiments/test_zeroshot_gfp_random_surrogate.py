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
from alf_core import (
    BaseDatasetConfig,
    FileStateLogger,
    Surrogate,
    TerminalStateLogger,
    ZeroShotTask,
)
from alf_tools.datasets import GFP


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
        train_ratio=0.0,
        validation_frac=0.0,
        test_ratio=1.0,
        split_type="random",
        problem_type="regression",
    )
    return GFP(config)


@pytest.fixture
def surrogate_model(random_model):
    """Fixture to create a random surrogate model for testing.

    Returns:
        A random surrogate model.
    """
    return Surrogate(model=random_model)


@pytest.fixture
def expected_metrics():
    """Fixture containing expected metric values for assertions.

    Returns:
        Expected metric values for assertions.
    """
    return {
        "surrogate": {
            "test_mse": 11.54878,
            "test_spearman": 0.01734,
            "test_pearson": 0.03023,
            "test_pairwise_xent": 0.43931,
        },
        "dataset": {
            "num_train": 0.0,
            "num_validation": 0.0,
            "num_test": 1000.0,
            "test_mean": 3.13320,
            "num_candidate_pool": 0.0,
        },
    }


class TestZeroShotGFPRandomSurrogate:
    """Tests a zero-shot experiment with a random surrogate model on the GFP dataset."""

    def test_zeroshot_gfp_random_surrogate_experiment(
        self, gfp_dataset, surrogate_model, expected_metrics, tmp_path
    ):
        """Test the complete zero-shot GFP random surrogate experiment pipeline.

        This test verifies that the zero-shot pipeline works correctly
        and produces expected metrics for the GFP dataset using a random surrogate model.
        """
        # Use pytest's tmp_path for temporary directory
        save_path = tmp_path / "zeroshot_gfp_random_surrogate"
        save_path.mkdir()

        metrics_logger = TerminalStateLogger()
        file_logger = FileStateLogger(output_path=save_path)
        state_loggers = [metrics_logger, file_logger]

        # Create and run the zero-shot task
        task = ZeroShotTask()
        state = task.setup(dataset=gfp_dataset, surrogate=surrogate_model)
        task.run(state=state, state_loggers=state_loggers)

        # Load and verify results
        metrics_file = save_path / "metrics.csv"
        assert metrics_file.exists(), "Metrics file should be created"

        metrics = pd.read_csv(metrics_file)

        # Test surrogate performance metrics
        self._assert_surrogate_metrics(metrics, expected_metrics["surrogate"])

        # Test dataset metrics
        self._assert_dataset_metrics(metrics, expected_metrics["dataset"])

    def _assert_surrogate_metrics(self, metrics: pd.DataFrame, expected: dict):
        """Assert surrogate model performance metrics."""
        for metric_name, expected_value in expected.items():
            actual_value = metrics[f"surrogate/{metric_name}"].iloc[0]
            assert np.isclose(actual_value, expected_value, atol=1e-5), (
                f"Surrogate metric {metric_name} mismatch: expected {expected_value}, "
                f"got {actual_value}"
            )

    def _assert_dataset_metrics(self, metrics: pd.DataFrame, expected: dict):
        """Assert dataset metrics."""
        for metric_name, expected_value in expected.items():
            actual_value = metrics[f"dataset/{metric_name}"].iloc[0]
            assert np.isclose(actual_value, expected_value, atol=1e-5), (
                f"Dataset metric {metric_name} mismatch: expected {expected_value}, "
                f"got {actual_value}"
            )
