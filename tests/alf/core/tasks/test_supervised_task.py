"""Tests for the supervised task."""

import shutil

import numpy as np
import pandas as pd
import pytest

from alf.core.tasks.supervised_task import SupervisedTask
from alf.core.utils.task_state_logger import FileTaskStateLogger, TerminalTaskStateLogger


@pytest.fixture
def expected_metrics():
    """Fixture containing expected metric values for assertions."""
    return {
        "surrogate": {
            "test_mse": 35.11162,
            "test_spearman": -0.00425,
            "test_pearson": 0.00483,
            "test_pairwise_xent": 0.44950,
        },
        "dataset": {
            "num_train": 480.00000,
            "train_mean": 5.04921,
            "num_validation": 120.00000,
            "validation_mean": 4.83871,
            "num_test": 200.00000,
            "test_mean": 5.13172,
            "num_candidate_pool": 200.00000,
            "candidate_pool_mean": 4.82509,
        },
    }


class TestSupervisedTask:
    """Tests a supervised experiment with a dummy surrogate model on a dummy dataset."""

    def test_supervised_dummy_surrogate_experiment(
        self, dummy_dataset, dummy_surrogate, expected_metrics, tmp_path
    ):
        """Test the complete supervised dummy surrogate experiment pipeline.

        This test verifies that the supervised learning pipeline works correctly
        and produces expected metrics for the dummy dataset using a dummy surrogate model.
        """
        # Use pytest's tmp_path for temporary directory
        save_path = tmp_path / "supervised_dummy_surrogate"
        save_path.mkdir()

        metrics_logger = TerminalTaskStateLogger()
        file_logger = FileTaskStateLogger(file_path=str(save_path))
        loggers = [metrics_logger, file_logger]

        # Create and run the supervised task
        task = SupervisedTask()
        state = task.setup(dataset=dummy_dataset, surrogate=dummy_surrogate)
        task.run(state=state, loggers=loggers)

        # Load and verify results
        metrics_file = save_path / "metrics.csv"
        assert metrics_file.exists(), "Metrics file should be created"

        metrics = pd.read_csv(metrics_file)

        # Test surrogate performance metrics
        self._assert_surrogate_metrics(metrics, expected_metrics["surrogate"])

        # Test dataset metrics
        self._assert_dataset_metrics(metrics, expected_metrics["dataset"])

        # Clean up: remove the results folder after assertions
        shutil.rmtree(save_path)

    def _assert_surrogate_metrics(self, metrics: pd.DataFrame, expected: dict):
        """Assert surrogate model performance metrics."""
        for metric_name, expected_value in expected.items():
            actual_value = metrics[f"surrogate/{metric_name}"].iloc[0]
            assert np.isclose(actual_value, expected_value, atol=1e-5), (
                f"Surrogate metric {metric_name} mismatch: "
                f"expected {expected_value}, got {actual_value}"
            )

    def _assert_dataset_metrics(self, metrics: pd.DataFrame, expected: dict):
        """Assert dataset metrics."""
        for metric_name, expected_value in expected.items():
            actual_value = metrics[f"dataset/{metric_name}"].iloc[0]
            assert np.isclose(actual_value, expected_value, atol=1e-5), (
                f"Dataset metric {metric_name} mismatch: "
                f"expected {expected_value}, got {actual_value}"
            )
