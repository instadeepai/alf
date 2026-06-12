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

"""Tests for the design task."""

import numpy as np
import pandas as pd
import pytest
from alf_core.tasks.design_task import DesignTask
from alf_core.utils.state_logger import FileStateLogger, TerminalStateLogger


@pytest.fixture
def expected_metrics():
    """Fixture containing expected metric values for assertions.

    Returns:
        Expected metric values for assertions.
    """
    return {
        "acquired_candidates": {
            "round_mean": 4.77942,
            "round_max": 9.22434,
            "round_min": 0.66444,
        },
        "optimizer": {
            "top_10pc_recall": 1.00000,
            "top_100_recall": 1.00000,
            "regret": -0.01515,
        },
        "surrogate": {
            "test_mse": 35.11162,
            "test_spearman": -0.00425,
            "test_pearson": 0.00483,
            "test_pairwise_xent": 0.44950,
        },
        "dataset": {
            "num_train": 520.00000,
            "train_mean": 5.05563,
            "num_validation": 130.00000,
            "validation_mean": 4.72903,
            "num_test": 200.00000,
            "test_mean": 5.13172,
            "num_candidate_pool": 150.00000,
            "candidate_pool_mean": 4.83720,
        },
    }


class TestDesignTask:
    """Tests a design experiment with a dummy surrogate model on a dummy dataset."""

    def test_design_dummy_surrogate_experiment(
        self,
        dummy_dataset,
        dummy_surrogate,
        dummy_optimizer,
        oracle,
        expected_metrics,
        tmp_path,
    ):
        """Test the complete design dummy surrogate experiment pipeline.

        This test verifies that the design pipeline works correctly
        and produces expected metrics for the dummy dataset using a dummy surrogate model.
        """
        # Use pytest's tmp_path for temporary directory
        save_path = tmp_path / "design_dummy_surrogate"
        save_path.mkdir()

        metrics_logger = TerminalStateLogger()
        file_logger = FileStateLogger(output_path=save_path)
        state_loggers = [metrics_logger, file_logger]

        # Create and run the design task
        task = DesignTask(num_acq_rounds=5, acq_batch_size=10)
        state = task.setup(dataset=dummy_dataset, surrogate=dummy_surrogate)
        task.run(
            state=state,
            state_loggers=state_loggers,
            optimizer=dummy_optimizer,
            oracle=oracle,
        )

        # Load and verify results
        metrics_file = save_path / "metrics.csv"
        assert metrics_file.exists(), "Metrics file should be created"

        metrics = pd.read_csv(metrics_file)

        # Test acquired candidates metrics (last round)
        self._assert_acquired_candidates_metrics(metrics, expected_metrics["acquired_candidates"])

        # Test optimizer metrics (last round)
        self._assert_optimizer_metrics(metrics, expected_metrics["optimizer"])

        # Test surrogate metrics (last round)
        self._assert_surrogate_metrics(metrics, expected_metrics["surrogate"])

        # Test dataset metrics (last round)
        self._assert_dataset_metrics(metrics, expected_metrics["dataset"])

        # Test the end-of-campaign summary metric is emitted and well-formed
        self._assert_campaign_summary(metrics)

    def _assert_acquired_candidates_metrics(self, metrics: pd.DataFrame, expected: dict):
        """Assert acquired candidates metrics."""
        for metric_name, expected_value in expected.items():
            # Get the last populated round value; the trailing campaign_summary
            # row only carries auc_top_k, leaving these columns NaN.
            actual_value = metrics[f"acquired_candidates/{metric_name}"].dropna().iloc[-1]
            assert np.isclose(actual_value, expected_value, atol=1e-5), (
                f"Acquired candidates metric {metric_name} mismatch: "
                f"expected {expected_value}, got {actual_value}"
            )

    def _assert_optimizer_metrics(self, metrics: pd.DataFrame, expected: dict):
        """Assert optimizer metrics."""
        for metric_name, expected_value in expected.items():
            # Get the last populated round value
            actual_value = metrics[f"optimizer/{metric_name}"].dropna().iloc[-1]
            assert np.isclose(actual_value, expected_value, atol=1e-5), (
                f"Optimizer metric {metric_name} mismatch: "
                f"expected {expected_value}, got {actual_value}"
            )

    def _assert_surrogate_metrics(self, metrics: pd.DataFrame, expected: dict):
        """Assert surrogate model performance metrics."""
        for metric_name, expected_value in expected.items():
            # Get the last populated round value
            actual_value = metrics[f"surrogate/{metric_name}"].dropna().iloc[-1]
            assert np.isclose(actual_value, expected_value, atol=1e-5), (
                f"Surrogate metric {metric_name} mismatch: "
                f"expected {expected_value}, got {actual_value}"
            )

    def _assert_dataset_metrics(self, metrics: pd.DataFrame, expected: dict):
        """Assert dataset metrics."""
        for metric_name, expected_value in expected.items():
            # Get the last populated round value
            actual_value = metrics[f"dataset/{metric_name}"].dropna().iloc[-1]
            assert np.isclose(actual_value, expected_value, atol=1e-5), (
                f"Dataset metric {metric_name} mismatch: "
                f"expected {expected_value}, got {actual_value}"
            )

    def _assert_campaign_summary(self, metrics: pd.DataFrame):
        """Assert the end-of-campaign auc_top_k summary metric is emitted and in range."""
        assert "auc_top_k" in metrics.columns, (
            "campaign_summary row with auc_top_k should be logged after all rounds"
        )
        auc = metrics["auc_top_k"].dropna()
        assert len(auc) == 1, "auc_top_k should be logged exactly once, at campaign end"
        assert 0.0 <= auc.iloc[-1] <= 1.0, f"auc_top_k out of range: {auc.iloc[-1]}"
