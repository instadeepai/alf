import pytest
import numpy as np
from alf.tools.datasets.gfp import GFP


@pytest.fixture
def gfp_dataset():
    """Create a GFP dataset fixture for testing."""
    return GFP(
        name="gfp",
        modality="sequence", 
        seed=51505,
        split_config={
            "split_ratio": {"train": 0.05, "validation": 0.05, "test": 0.2},
            "split_type": "random"
        }
    )


class TestGFPDataset:
    """Test class for GFP dataset functionality."""
    
    def test_dataset_initialization(self, gfp_dataset):
        """Test that dataset initializes correctly."""
        assert gfp_dataset.name == "gfp"
        assert gfp_dataset.modality == "sequence"
        assert gfp_dataset.seed == 51505
    
    def test_dataset_split_sizes(self, gfp_dataset):
        """Test that dataset splits have correct sizes."""
        assert len(gfp_dataset.train_dataset) == 50, "Train dataset should have 50 samples"
        assert len(gfp_dataset.test_dataset) == 200, "Test dataset should have 200 samples"
        assert len(gfp_dataset.validation_dataset) == 50, "Validation dataset should have 50 samples"
        assert len(gfp_dataset.candidate_pool) == 700, "Candidate pool should have 700 samples"
    
    def test_dataset_split_consistency(self, gfp_dataset):
        """Test that all splits sum to the total candidate pool size."""
        total_split_size = (
            len(gfp_dataset.train_dataset) + 
            len(gfp_dataset.test_dataset) + 
            len(gfp_dataset.validation_dataset) +
            len(gfp_dataset.candidate_pool)
        )
        assert total_split_size == len(gfp_dataset._raw_dataset), \
            "Sum of all splits should equal raw dataset size"
    
    def test_dataset_label_statistics(self, gfp_dataset):
        """Test that dataset label statistics match expected values."""
        train_mean = np.mean(gfp_dataset.train_dataset.labels)
        validation_mean = np.mean(gfp_dataset.validation_dataset.labels)
        test_mean = np.mean(gfp_dataset.test_dataset.labels)
        
        assert train_mean == pytest.approx(3.1403610808843996, rel=1e-10), \
            f"Train dataset mean should be ~3.140, got {train_mean}"
        assert validation_mean == pytest.approx(3.0643773572906, rel=1e-10), \
            f"Validation dataset mean should be ~3.064, got {validation_mean}"
        assert test_mean == pytest.approx(3.14816517831885, rel=1e-10), \
            f"Test dataset mean should be ~3.148, got {test_mean}"
    
    def test_dataset_data_types(self, gfp_dataset):
        """Test that dataset contains expected data types."""
        # Test that all datasets have labels
        assert hasattr(gfp_dataset.train_dataset, 'labels'), "Train dataset should have labels"
        assert hasattr(gfp_dataset.test_dataset, 'labels'), "Test dataset should have labels"
        assert hasattr(gfp_dataset.validation_dataset, 'labels'), "Validation dataset should have labels"
        
        # Test that labels are numpy arrays
        assert isinstance(gfp_dataset.train_dataset.labels, np.ndarray), \
            "Train dataset labels should be numpy array"
        assert isinstance(gfp_dataset.test_dataset.labels, np.ndarray), \
            "Test dataset labels should be numpy array"
        assert isinstance(gfp_dataset.validation_dataset.labels, np.ndarray), \
            "Validation dataset labels should be numpy array"
    
    def test_dataset_no_empty_splits(self, gfp_dataset):
        """Test that no dataset splits are empty."""
        assert len(gfp_dataset.train_dataset) > 0, "Train dataset should not be empty"
        assert len(gfp_dataset.test_dataset) > 0, "Test dataset should not be empty"
        assert len(gfp_dataset.validation_dataset) > 0, "Validation dataset should not be empty"
        assert len(gfp_dataset.candidate_pool) > 0, "Candidate pool should not be empty"