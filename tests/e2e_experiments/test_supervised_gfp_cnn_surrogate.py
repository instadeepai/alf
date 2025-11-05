import pytest
import numpy as np
import torch
import pandas as pd
import shutil
from pathlib import Path

from alf.tools.datasets.gfp import GFP
from alf.tools.models.cnn import CNNModel, CNNTrainConfig
from alf.core.utils.logger import TerminalLogger
from alf.core.tasks.supervised_task import SupervisedTask
from alf.core.oracle.oracle import Oracle
from alf.core.surrogate.surrogate import Surrogate


@pytest.fixture
def set_seed():
    """Fixture to set random seeds for reproducible tests."""
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
    """Fixture to create a GFP dataset for testing."""
    return GFP(
        name="gfp", 
        modality="sequence", 
        seed=51505, 
        split_config={
            # old version             "split_ratio": {"train": 0.1, "test": 0.2, "validation": 0.5}, 
            "split_ratio": {"train": 0.6, "test": 0.2, "validation_frac": 0.833}, 
            "split_type": "random"
        }
    )


@pytest.fixture
def surrogate_model():
    """Fixture to create a surrogate model for testing."""
    return Surrogate(model=CNNModel(train_config=CNNTrainConfig(num_epochs=10)))


@pytest.fixture
def oracle(gfp_dataset):
    """Fixture to create optimizer components."""
    oracle = Oracle(module=gfp_dataset)
    return oracle


@pytest.fixture
def expected_metrics():
    """Fixture containing expected metric values for assertions."""
    return {
        "surrogate": {
            "test_mse": 2.82286,
            "test_spearman": 0.08317,
            "test_pearson": 0.18264,
            "test_pairwise_xent": 0.34481
        },
        "dataset": {
            "num_train": 50.00000,
            "train_mean": 3.14036,
            "num_validation": 50.00000,
            "validation_mean": 3.06438,
            "num_test": 200.00000,
            "test_mean": 3.14817,
            "num_candidate_pool": 700.00000
        }
    }


class TestSupervised:
    """Tests a supervised experiment with a CNN surrogate model on the GFP dataset."""
    
    def test_supervised_gfp_cnn_experiment(
        self, 
        set_seed, 
        gfp_dataset, 
        surrogate_model, 
        expected_metrics, 
        tmp_path
    ):
        """
        Test the complete supervised GFP CNN experiment pipeline.
        
        This test verifies that the supervised learning pipeline works correctly
        and produces expected metrics for the GFP dataset using a CNN surrogate model.
        """
        # Set seed for reproducible results
        set_seed(42)
        
        # Use pytest's tmp_path for temporary directory
        save_path = tmp_path / "supervised_gfp_cnn"
        save_path.mkdir()
        
        # Create and run the supervised task
        task = SupervisedTask()
        state = task.setup(dataset=gfp_dataset, surrogate=surrogate_model)
        task.run(
            state=state, 
            logger=TerminalLogger(), 
            oracle=oracle, 
            save_path=str(save_path)
        )
        
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
        print(metrics.iloc[-1])
        for metric_name, expected_value in expected.items():
            actual_value = metrics[f"surrogate/{metric_name}"].iloc[0]
            assert np.isclose(
                actual_value, 
                expected_value, 
                atol=1e-4
            ), f"Surrogate metric {metric_name} mismatch: expected {expected_value}, got {actual_value}"
    
    def _assert_dataset_metrics(self, metrics: pd.DataFrame, expected: dict):
        """Assert dataset metrics."""
        print(metrics.iloc[-1])
        for metric_name, expected_value in expected.items():
            actual_value = metrics[f"dataset/{metric_name}"].iloc[0]
            assert np.isclose(
                actual_value, 
                expected_value, 
                atol=1e-4
            ), f"Dataset metric {metric_name} mismatch: expected {expected_value}, got {actual_value}"