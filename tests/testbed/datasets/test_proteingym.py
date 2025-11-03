from testbed.datasets.proteingym import ProteinGym
import numpy as np

def test_proteingym_dataset_singles():
    """Test that the Proteingym dataset initializes correctly."""
    dataset = ProteinGym(
        name="proteingym",
        modality="sequence",
        seed=51505,
        split_config={
            "split_ratio": {"train": 0.6, "validation": 0.2, "test": 0.2},
            "split_type": "random"
        },
        dataset_config={
            "dms_name": "IF1_ECOLI_Kelsic_2016",
            "dms_type": "singles"
        }
    )
    print(dataset)

def test_proteingym_dataset_multiples():
    """Test that the Proteingym dataset initializes correctly."""
    dataset = ProteinGym(
        name="proteingym",
        modality="sequence",
        seed=51505,
        split_config={
            "split_ratio": {"train": 0.6, "validation": 0.2, "test": 0.2},
            "split_type": "random"
        },
        dataset_config={
            "dms_name": "CAPSD_AAV2S_Sinai_2021",
            "dms_type": "multiples"
        }
    )
    print(dataset)

def test_proteingym_dataset_cross_validation_singles():
    """Test that the Proteingym dataset initializes correctly."""
    dataset = ProteinGym(
        name="proteingym",
        modality="sequence",
        seed=51505,
        split_config={
            "split_ratio": {"train": 0.783, "validation": 0.0, "test": 0.218},
            "split_type": "random"
        },
        dataset_config={
            "dms_name": "IF1_ECOLI_Kelsic_2016",
            "dms_type": "singles",
            "cross_validation": True,
            "cross_validation_type": "random",
            "cross_validation_fold": 0
        }
    )
    assert len(dataset.train_dataset) == 1070
    assert len(dataset.test_dataset) == 297
    assert len(dataset.validation_dataset) == 0
    assert len(dataset.candidate_pool) == 0

    assert np.isclose(np.mean(dataset.train_dataset.labels), 0.790617)
    assert np.isclose(np.mean(dataset.test_dataset.labels), 0.799398)

def test_proteingym_dataset_cross_validation_multiples():
    """Test that the Proteingym dataset initializes correctly."""
    dataset = ProteinGym(
        name="proteingym",
        modality="sequence",
        seed=51505,
        split_config={
            "split_ratio": {"train": 0.8, "validation": 0.0, "test": 0.21},
            "split_type": "random"
        },
        dataset_config={
            "dms_name": "CAPSD_AAV2S_Sinai_2021",
            "dms_type": "multiples",
            "cross_validation": True,
            "cross_validation_type": "random",
            "cross_validation_fold": 0
        }
    )
    assert len(dataset.train_dataset) == 33849
    assert len(dataset.test_dataset) == 8479, f"Test dataset size should be 8479, got {len(dataset.test_dataset)}"
    assert len(dataset.validation_dataset) == 0
    assert len(dataset.candidate_pool) == 0

    assert np.isclose(np.mean(dataset.train_dataset.labels), -1.227048)
    assert np.isclose(np.mean(dataset.test_dataset.labels), -1.221083)

test_proteingym_dataset_singles()
test_proteingym_dataset_multiples()
test_proteingym_dataset_cross_validation_singles()
test_proteingym_dataset_cross_validation_multiples()