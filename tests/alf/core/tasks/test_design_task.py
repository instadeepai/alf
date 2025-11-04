"""Tests for the design task."""
import pytest
import numpy as np
import pandas as pd
import shutil

from alf.core.tasks.design_task import DesignTask
from alf.core.utils.logger import TerminalLogger


@pytest.fixture
def expected_metrics():
    """Fixture containing expected metric values for assertions."""
    return {
        "acquired_candidates": {
            "round_mean": 4.77942,
            "round_max": 9.22434,
            "round_min": 0.66444
        },
        "optimizer": {
            "top_percentile_recall": 1.00000,
            "top_n_recall": 1.00000,
            "regret": -0.01515
        },
        "surrogate": {
            "test_mse": 35.11162,
            "test_spearman": -0.00425,
            "test_pearson": 0.00483,
            "test_pairwise_xent": 0.44950
        },
        "dataset": {
            "num_train": 525.00000,
            "train_mean": 5.03691,
            "num_validation": 125.00000,
            "validation_mean": 4.79461,
            "num_test": 200.00000,
            "test_mean": 5.13172,
            "num_candidate_pool": 150.00000,
            "candidate_pool_mean": 4.83720
        }
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
        tmp_path
    ):
        """
        Test the complete design dummy surrogate experiment pipeline.
        
        This test verifies that the design pipeline works correctly
        and produces expected metrics for the dummy dataset using a dummy surrogate model.
        """
        # Use pytest's tmp_path for temporary directory
        save_path = tmp_path / "design_dummy_surrogate"
        save_path.mkdir()
        
        # Create and run the design task
        task = DesignTask(num_acq_rounds=5, acq_batch_size=10)
        state = task.setup(dataset=dummy_dataset, surrogate=dummy_surrogate)
        task.run(
            state=state,
            logger=TerminalLogger(),
            optimizer=dummy_optimizer,
            oracle=oracle,
            save_path=str(save_path)
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
        
        # Clean up: remove the results folder after assertions
        shutil.rmtree(save_path)
    
    def _assert_acquired_candidates_metrics(self, metrics: pd.DataFrame, expected: dict):
        """Assert acquired candidates metrics."""
        for metric_name, expected_value in expected.items():
            # Get the last round value (design tasks have multiple rounds)
            actual_value = metrics[f"acquired_candidates/{metric_name}"].iloc[-1]
            assert np.isclose(
                actual_value, 
                expected_value, 
                atol=1e-5
            ), f"Acquired candidates metric {metric_name} mismatch: expected {expected_value}, got {actual_value}"
    
    def _assert_optimizer_metrics(self, metrics: pd.DataFrame, expected: dict):
        """Assert optimizer metrics."""
        for metric_name, expected_value in expected.items():
            # Get the last round value
            actual_value = metrics[f"optimizer/{metric_name}"].iloc[-1]
            assert np.isclose(
                actual_value, 
                expected_value, 
                atol=1e-5
            ), f"Optimizer metric {metric_name} mismatch: expected {expected_value}, got {actual_value}"
    
    def _assert_surrogate_metrics(self, metrics: pd.DataFrame, expected: dict):
        """Assert surrogate model performance metrics."""
        for metric_name, expected_value in expected.items():
            # Get the last round value
            actual_value = metrics[f"surrogate/{metric_name}"].iloc[-1]
            assert np.isclose(
                actual_value, 
                expected_value, 
                atol=1e-5
            ), f"Surrogate metric {metric_name} mismatch: expected {expected_value}, got {actual_value}"
    
    def _assert_dataset_metrics(self, metrics: pd.DataFrame, expected: dict):
        """Assert dataset metrics."""
        for metric_name, expected_value in expected.items():
            # Get the last round value
            actual_value = metrics[f"dataset/{metric_name}"].iloc[-1]
            assert np.isclose(
                actual_value, 
                expected_value, 
                atol=1e-5
            ), f"Dataset metric {metric_name} mismatch: expected {expected_value}, got {actual_value}"
