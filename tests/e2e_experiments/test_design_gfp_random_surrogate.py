import pytest
import numpy as np
import pandas as pd
import shutil

from alf.tools.datasets.gfp import GFP
from alf.core.surrogate.surrogate import Surrogate
from alf.core.optimizer.optimizer import Optimizer
from alf.core.optimizer.search import DatasetSearch
from alf.core.oracle.oracle import Oracle
from alf.core.utils.logger import TerminalLogger
from alf.core.tasks.design_task import DesignTask
from alf.tools.models.random import RandomModel
from alf.tools.optimizer.acquisition_functions.greedy import Greedy


@pytest.fixture
def gfp_dataset():
    """Fixture to create a GFP dataset for testing."""
    return GFP(
        name="gfp", 
        modality="sequence", 
        seed=51505, 
        split_config={
            "split_ratio": {"train": 0.08, "validation": 0.02, "test": 0.2}, 
            "split_type": "random"
        }
    )


@pytest.fixture
def surrogate_model():
    """Fixture to create a random surrogate model for testing."""
    return Surrogate(model=RandomModel(seed=42))


@pytest.fixture
def acquisition_fn():
    """Fixture to create acquisition function."""
    return Greedy()


@pytest.fixture
def search_fn():
    """Fixture to create search strategy."""
    return DatasetSearch()


@pytest.fixture
def optimizer(acquisition_fn, search_fn):
    """Fixture to create optimizer."""
    return Optimizer(acquisition_fn=acquisition_fn, search_fn=search_fn)


@pytest.fixture
def oracle(gfp_dataset):
    """Fixture to create oracle."""
    return Oracle(module=gfp_dataset)


@pytest.fixture
def expected_metrics():
    """Fixture containing expected metric values for assertions."""
    return {
        "dataset": {
            "train_mean": [3.0956020055045004, 3.1138987210297757, 3.107286033359058, 3.0764754215457217, 3.1073208870911864, 3.1253016808944385],
            "validation_mean": [3.12944, 3.20147, 3.23996, 3.19169, 3.14611, 3.13312],
            "test_mean": [3.14816517831885, 3.14816517831885, 3.14816517831885, 3.14816517831885, 3.14816517831885, 3.14816517831885],
            "num_train": [80, 178, 276, 374, 472, 570],
            "num_validation": [20, 22, 24, 26, 28, 30],
            "num_test": [200, 200, 200, 200, 200, 200]
        },
        "acquired_candidates": {
            "round_mean": [3.0956020055045004, 3.1446933434218005, 3.1066377061713997, 2.9821563063825, 3.2116081743086, 3.2066899330146996],
            "round_max": [3.95286314416, 4.01842590272, 3.92656957125, 3.87539383402, 3.91434911327, 3.9957144665],
            "round_min": [1.30102999533, 1.30022633628, 1.30102997845, 1.3000657597799998, 1.29944409644, 1.30102999164]
        },
        "surrogate": {
            "test_spearman": [-0.1132738318457961, -0.0664366609165229, 0.018894472361809, 0.0143913597839946, -0.0230390759768994, 0.1438550963774094],
            "test_pearson": [-0.0898151255645407, -0.0558276497970699, -0.0068104328775336, -0.016707345915439, -0.0192779720121573, 0.138561843375223],
            "test_pairwise_xent": [0.4597295220441595, 0.4597777189394591, 0.4264968492740766, 0.4417389609683779, 0.4628584084686833, 0.4097662801029683],
            "test_mse": [12.12838086761283, 11.282318387495234, 11.68684258415281, 11.472448168672829, 12.788615223717644, 11.00927285656157]
        }
    }


class TestDesignGFPRandomSurrogate:
    """Tests a design experiment with a random surrogate model on the GFP dataset."""
    
    def test_design_gfp_random_surrogate_experiment(
        self, 
        gfp_dataset, 
        surrogate_model, 
        optimizer, 
        oracle, 
        expected_metrics, 
        tmp_path
    ):
        """
        Test the complete design GFP random surrogate experiment pipeline.
        
        This test verifies that the design pipeline works correctly
        and produces expected metrics for the GFP dataset using a random surrogate model.
        """
        # Use pytest's tmp_path for temporary directory
        save_path = tmp_path / "design_gfp_random_surrogate"
        save_path.mkdir()
        
        # Create and run the design task
        task = DesignTask(num_acq_rounds=5, acq_batch_size=100)
        state = task.setup(dataset=gfp_dataset, surrogate=surrogate_model)
        task.run(
            state=state, 
            logger=TerminalLogger(), 
            optimizer=optimizer, 
            oracle=oracle, 
            save_path=str(save_path)
        )
        
        # Load and verify results
        metrics_file = save_path / "metrics.csv"
        assert metrics_file.exists(), "Metrics file should be created"
        
        metrics = pd.read_csv(metrics_file)
        
        # Test dataset metrics
        self._assert_dataset_metrics(metrics, expected_metrics["dataset"])
        
        # Test acquired candidates metrics
        self._assert_acquired_candidates_metrics(metrics, expected_metrics["acquired_candidates"])
        
        # Test surrogate metrics
        self._assert_surrogate_metrics(metrics, expected_metrics["surrogate"])
        
        # Clean up: remove the results folder after assertions
        shutil.rmtree(save_path)
    
    def _assert_dataset_metrics(self, metrics: pd.DataFrame, expected: dict):
        """Assert dataset metrics."""
        for metric_name, expected_values in expected.items():
            actual_values = metrics[f"dataset/{metric_name}"].tolist()
            assert np.isclose(
                actual_values, 
                expected_values, 
                atol=1e-10
            ).all(), f"Dataset metric {metric_name} mismatch: expected {expected_values}, got {actual_values}"
    
    def _assert_acquired_candidates_metrics(self, metrics: pd.DataFrame, expected: dict):
        """Assert acquired candidates metrics."""
        for metric_name, expected_values in expected.items():
            actual_values = metrics[f"acquired_candidates/{metric_name}"].tolist()
            assert np.isclose(
                actual_values, 
                expected_values, 
                atol=1e-10
            ).all(), f"Acquired candidates metric {metric_name} mismatch: expected {expected_values}, got {actual_values}"
    
    def _assert_surrogate_metrics(self, metrics: pd.DataFrame, expected: dict):
        """Assert surrogate model performance metrics."""
        for metric_name, expected_values in expected.items():
            actual_values = metrics[f"surrogate/{metric_name}"].tolist()
            assert np.isclose(
                actual_values, 
                expected_values, 
                atol=1e-10
            ).all(), f"Surrogate metric {metric_name} mismatch: expected {expected_values}, got {actual_values}"