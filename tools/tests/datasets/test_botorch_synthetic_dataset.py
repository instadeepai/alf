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

"""Tests for BoTorch synthetic test function dataset adapter."""

import numpy as np
import pytest
from alf_core import BaseDatasetConfig, Candidate, Modality
from alf_tools.datasets.botorch_synthetic_dataset import BoTorchSyntheticDataset


@pytest.fixture
def base_config():
    """Create a base dataset configuration for testing.

    Returns:
        BaseDatasetConfig: Configuration for test datasets.
    """
    return BaseDatasetConfig(
        name="test_branin",
        modality=Modality.TABULAR,
        seed=42,
        train_ratio=0.1,
        validation_frac=0.2,
        test_ratio=0.1,
        split_type="random",
        problem_type="regression",
    )


def test_branin_initialization(base_config):
    """Test initialization of Branin dataset."""
    dataset = BoTorchSyntheticDataset(
        config=base_config,
        function_name="branin",
        noise_std=0.0,
        n_initial_samples=100,
    )

    assert dataset.function_name == "branin"
    assert dataset.dim == 2
    assert dataset.noise_std == 0.0
    assert dataset.n_initial_samples == 100
    assert dataset.bounds.shape == (2, 2)
    assert dataset.true_optimum is not None


def test_hartmann_initialization(base_config):
    """Test initialization of Hartmann datasets with different dimensions."""
    # Hartmann 3D
    dataset_3d = BoTorchSyntheticDataset(
        config=base_config,
        function_name="hartmann3",
        n_initial_samples=100,
    )
    assert dataset_3d.dim == 3
    assert dataset_3d.bounds.shape == (2, 3)

    # Hartmann 6D
    dataset_6d = BoTorchSyntheticDataset(
        config=base_config,
        function_name="hartmann6",
        n_initial_samples=100,
    )
    assert dataset_6d.dim == 6
    assert dataset_6d.bounds.shape == (2, 6)


def test_variable_dimension_functions(base_config):
    """Test functions that support variable dimensions."""
    # Ackley with dim=5
    dataset = BoTorchSyntheticDataset(
        config=base_config,
        function_name="ackley",
        dim=5,
        n_initial_samples=100,
    )
    assert dataset.dim == 5
    assert dataset.bounds.shape == (2, 5)


def test_missing_dim_raises_error(base_config):
    """Test that missing dim for variable-dimension functions raises error."""
    with pytest.raises(ValueError, match="Must specify 'dim'"):
        BoTorchSyntheticDataset(
            config=base_config,
            function_name="ackley",
            n_initial_samples=100,
        )


def test_invalid_function_name(base_config):
    """Test that invalid function name raises error."""
    with pytest.raises(ValueError, match="Unknown function"):
        BoTorchSyntheticDataset(
            config=base_config,
            function_name="invalid_function",
            n_initial_samples=100,
        )


def test_wrong_modality_raises_error():
    """Test that non-TABULAR modality raises error."""
    config = BaseDatasetConfig(
        name="test",
        modality=Modality.SEQUENCE,  # Wrong modality
        seed=42,
        train_ratio=0.1,
        validation_frac=0.2,
        test_ratio=0.1,
        split_type="random",
        problem_type="regression",
    )

    with pytest.raises(ValueError, match="require modality=TABULAR"):
        BoTorchSyntheticDataset(
            config=config,
            function_name="branin",
            n_initial_samples=100,
        )


def test_load_dataset(base_config):
    """Test loading the dataset generates correct number of samples."""
    n_samples = 100
    dataset = BoTorchSyntheticDataset(
        config=base_config,
        function_name="branin",
        noise_std=0.0,
        n_initial_samples=n_samples,
    )

    labelled_candidates = dataset.load_dataset()

    assert len(labelled_candidates) == n_samples
    assert len(labelled_candidates.candidates) == n_samples
    assert len(labelled_candidates.labels) == n_samples
    assert all(isinstance(c, Candidate) for c in labelled_candidates.candidates)
    assert all(c.modality == Modality.TABULAR for c in labelled_candidates.candidates)


def test_dataset_setup_and_splits(base_config):
    """Test that dataset setup creates all required splits."""
    dataset = BoTorchSyntheticDataset(
        config=base_config,
        function_name="branin",
        n_initial_samples=1000,
    )

    dataset.setup()

    # Check splits exist
    assert len(dataset.train_dataset) > 0
    assert len(dataset.validation_dataset) > 0
    assert len(dataset.test_dataset) > 0
    assert len(dataset.candidate_pool) > 0

    # Check metadata
    assert dataset.metadata is not None
    assert dataset.metadata["function_name"] == "branin"
    assert dataset.metadata["dim"] == 2


def test_query_candidates(base_config):
    """Test querying candidates evaluates them correctly."""
    dataset = BoTorchSyntheticDataset(
        config=base_config,
        function_name="branin",
        noise_std=0.0,
        n_initial_samples=100,
    )
    dataset.setup()

    # Create test candidates
    test_data = [
        np.array([0.5, 0.5]),
        np.array([0.1, 0.9]),
    ]
    test_candidates = [Candidate(data=d, modality=Modality.TABULAR) for d in test_data]

    # Query
    labelled = dataset.query(test_candidates)

    assert len(labelled) == 2
    assert labelled.candidates == test_candidates
    assert labelled.labels.shape == (2,)
    assert all(np.isfinite(labelled.labels))


def test_noise_addition(base_config):
    """Test that noise is properly added to evaluations."""
    noise_std = 0.1

    dataset = BoTorchSyntheticDataset(
        config=base_config,
        function_name="branin",
        noise_std=noise_std,
        n_initial_samples=100,
    )

    # Create deterministic candidate
    test_data = np.array([0.5, 0.5])
    test_candidate = Candidate(data=test_data, modality=Modality.TABULAR)

    # Query multiple times and check variance
    labels = []
    for _ in range(50):
        labelled = dataset.query([test_candidate])
        labels.append(labelled.labels[0])

    labels = np.array(labels)
    observed_std = np.std(labels)

    # Should be approximately noise_std (allow some variance due to finite samples)
    assert 0.05 < observed_std < 0.15


def test_negate_parameter(base_config):
    """Test that negate parameter correctly flips function values."""
    # With negation (default)
    dataset_neg = BoTorchSyntheticDataset(
        config=base_config,
        function_name="branin",
        negate=True,
        n_initial_samples=100,
    )
    dataset_neg.setup()

    # Without negation
    config_no_neg = BaseDatasetConfig(
        name="test_branin_no_neg",
        modality=Modality.TABULAR,
        seed=42,
        train_ratio=0.1,
        validation_frac=0.2,
        test_ratio=0.1,
        split_type="random",
        problem_type="regression",
    )
    dataset_no_neg = BoTorchSyntheticDataset(
        config=config_no_neg,
        function_name="branin",
        negate=False,
        n_initial_samples=100,
    )
    dataset_no_neg.setup()

    # Query same point
    test_candidate = Candidate(data=np.array([0.5, 0.5]), modality=Modality.TABULAR)
    label_neg = dataset_neg.query([test_candidate]).labels[0]
    label_no_neg = dataset_no_neg.query([test_candidate]).labels[0]

    # Should be negatives of each other
    assert np.isclose(label_neg, -label_no_neg)


def test_bounds_within_valid_range(base_config):
    """Test that all generated samples are within bounds."""
    dataset = BoTorchSyntheticDataset(
        config=base_config,
        function_name="branin",
        n_initial_samples=1000,
    )

    labelled_candidates = dataset.load_dataset()
    lower_bounds = dataset.bounds[0]
    upper_bounds = dataset.bounds[1]

    for candidate in labelled_candidates.candidates:
        data = candidate.data
        assert np.all(data >= lower_bounds)
        assert np.all(data <= upper_bounds)


def test_reproducibility_with_seed(base_config):
    """Test that same seed produces same dataset."""
    dataset1 = BoTorchSyntheticDataset(
        config=base_config,
        function_name="branin",
        noise_std=0.1,
        n_initial_samples=100,
    )
    labelled1 = dataset1.load_dataset()

    # Create another with same config (same seed)
    dataset2 = BoTorchSyntheticDataset(
        config=base_config,
        function_name="branin",
        noise_std=0.1,
        n_initial_samples=100,
    )
    labelled2 = dataset2.load_dataset()

    # Should produce identical results
    for c1, c2, l1, l2 in zip(
        labelled1.candidates, labelled2.candidates, labelled1.labels, labelled2.labels
    ):
        assert np.allclose(c1.data, c2.data)
        assert np.isclose(l1, l2)
