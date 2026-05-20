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

"""Tests for BoTorch utility functions."""

import numpy as np
import pytest
import torch
from alf_core import Candidate, Predictions
from alf_core.dataclasses.candidate import Modality
from alf_tools.utils.botorch_utils import (
    candidates_to_tensor,
    get_bounds_tensor,
    predictions_to_posterior,
    tensor_to_candidates,
)
from botorch.posteriors.gpytorch import GPyTorchPosterior


def test_candidates_to_tensor_basic():
    """Test basic conversion of candidates to tensor."""
    candidates = [
        Candidate(data=np.array([1.0, 2.0]), modality=Modality.TABULAR),
        Candidate(data=np.array([3.0, 4.0]), modality=Modality.TABULAR),
        Candidate(data=np.array([5.0, 6.0]), modality=Modality.TABULAR),
    ]

    X = candidates_to_tensor(candidates)

    assert X.shape == (3, 2)
    assert torch.allclose(X, torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    assert X.dtype == torch.float32


def test_candidates_to_tensor_with_torch_data():
    """Test conversion when candidate data is already torch tensors."""
    candidates = [
        Candidate(data=torch.tensor([1.0, 2.0]), modality=Modality.TABULAR),
        Candidate(data=torch.tensor([3.0, 4.0]), modality=Modality.TABULAR),
    ]

    X = candidates_to_tensor(candidates)

    assert X.shape == (2, 2)
    assert torch.allclose(X, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_candidates_to_tensor_empty_list():
    """Test that empty list raises ValueError."""
    with pytest.raises(ValueError, match="Cannot convert empty list"):
        candidates_to_tensor([])


def test_candidates_to_tensor_inconsistent_shapes():
    """Test that inconsistent shapes raise ValueError."""
    candidates = [
        Candidate(data=np.array([1.0, 2.0]), modality=Modality.TABULAR),
        Candidate(data=np.array([3.0, 4.0, 5.0]), modality=Modality.TABULAR),  # Wrong shape
    ]

    with pytest.raises(ValueError, match="same shape"):
        candidates_to_tensor(candidates)


def test_candidates_to_tensor_device():
    """Test that device parameter works correctly."""
    candidates = [
        Candidate(data=np.array([1.0, 2.0]), modality=Modality.TABULAR),
    ]

    # CPU device
    X_cpu = candidates_to_tensor(candidates, device=torch.device("cpu"))
    assert X_cpu.device.type == "cpu"

    # If CUDA is available, test GPU
    if torch.cuda.is_available():
        X_gpu = candidates_to_tensor(candidates, device=torch.device("cuda"))
        assert X_gpu.device.type == "cuda"


def test_tensor_to_candidates_basic():
    """Test basic conversion of tensor to candidates."""
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    candidates = tensor_to_candidates(X)

    assert len(candidates) == 3
    assert all(isinstance(c, Candidate) for c in candidates)
    assert all(c.modality == Modality.TABULAR for c in candidates)
    assert np.allclose(candidates[0].data, np.array([1.0, 2.0]))
    assert np.allclose(candidates[1].data, np.array([3.0, 4.0]))
    assert np.allclose(candidates[2].data, np.array([5.0, 6.0]))


def test_tensor_to_candidates_with_features():
    """Test conversion with custom features."""
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    features = {"source": "optimized"}

    candidates = tensor_to_candidates(X, features=features)

    assert len(candidates) == 2
    assert all(c.features == features for c in candidates)


def test_tensor_to_candidates_custom_modality():
    """Test conversion with custom modality."""
    X = torch.tensor([[1.0, 2.0]])

    candidates = tensor_to_candidates(X, modality=Modality.EMBEDDING)

    assert candidates[0].modality == Modality.EMBEDDING


def test_predictions_to_posterior_basic():
    """Test basic conversion of predictions to posterior."""
    means = np.array([1.0, 2.0, 3.0])
    variances = np.array([0.1, 0.2, 0.15])
    predictions = Predictions(means=means, variances=variances)

    posterior = predictions_to_posterior(predictions)

    assert isinstance(posterior, GPyTorchPosterior)
    assert posterior.mean.shape == torch.Size([3, 1])
    # Check values regardless of shape
    mean_flat = posterior.mean.flatten()
    assert torch.allclose(mean_flat, torch.tensor([1.0, 2.0, 3.0]))


def test_predictions_to_posterior_without_variances():
    """Test that predictions without variances raise ValueError."""
    predictions = Predictions(means=np.array([1.0, 2.0, 3.0]), variances=None)

    with pytest.raises(ValueError, match="without variances"):
        predictions_to_posterior(predictions)


def test_predictions_to_posterior_device():
    """Test posterior creation with custom device."""
    means = np.array([1.0, 2.0])
    variances = np.array([0.1, 0.2])
    predictions = Predictions(means=means, variances=variances)

    posterior = predictions_to_posterior(predictions, device=torch.device("cpu"))

    assert posterior.mean.device.type == "cpu"


def test_predictions_to_posterior_shape_handling():
    """Test that predictions are reshaped correctly for posterior."""
    # Test with 1D arrays
    means = np.array([1.0, 2.0, 3.0])
    variances = np.array([0.1, 0.2, 0.15])
    predictions = Predictions(means=means, variances=variances)

    posterior = predictions_to_posterior(predictions)

    assert posterior.mean.shape == torch.Size([3, 1])
    assert posterior.covariance_matrix.shape == torch.Size([3, 3])


def test_get_bounds_tensor_from_numpy():
    """Test conversion of numpy bounds to tensor."""
    bounds = np.array([[0.0, 0.0], [1.0, 1.0]])  # Shape (2, 2)

    bounds_tensor = get_bounds_tensor(bounds)

    assert bounds_tensor.shape == torch.Size([2, 2])
    assert torch.allclose(bounds_tensor, torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
    assert bounds_tensor.dtype == torch.float32


def test_get_bounds_tensor_from_list():
    """Test conversion of list of tuples to tensor."""
    bounds = [(0.0, 1.0), (0.0, 1.0), (-1.0, 1.0)]  # 3 dimensions

    bounds_tensor = get_bounds_tensor(bounds)

    assert bounds_tensor.shape == torch.Size([2, 3])
    expected = torch.tensor([[0.0, 0.0, -1.0], [1.0, 1.0, 1.0]])
    assert torch.allclose(bounds_tensor, expected)


def test_get_bounds_tensor_device():
    """Test bounds tensor creation with custom device."""
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    bounds_tensor = get_bounds_tensor(bounds, device=torch.device("cpu"))

    assert bounds_tensor.device.type == "cpu"


def test_roundtrip_conversion():
    """Test that converting candidates -> tensor -> candidates preserves data."""
    original_candidates = [
        Candidate(data=np.array([1.0, 2.0, 3.0]), modality=Modality.TABULAR),
        Candidate(data=np.array([4.0, 5.0, 6.0]), modality=Modality.TABULAR),
    ]

    # Convert to tensor
    X = candidates_to_tensor(original_candidates)

    # Convert back to candidates
    reconstructed_candidates = tensor_to_candidates(X)

    # Check data is preserved
    for orig, recon in zip(original_candidates, reconstructed_candidates):
        assert np.allclose(orig.data, recon.data)
        assert recon.modality == Modality.TABULAR


def test_predictions_posterior_statistics():
    """Test that posterior statistics match predictions."""
    means = np.array([1.0, 2.0, 3.0])
    variances = np.array([0.1, 0.2, 0.15])
    predictions = Predictions(means=means, variances=variances)

    posterior = predictions_to_posterior(predictions)

    # Check mean matches (flatten to handle different shapes)
    mean_flat = posterior.mean.flatten()
    assert torch.allclose(mean_flat, torch.tensor(means, dtype=torch.float32), atol=1e-5)

    # Check variance matches (diagonal of covariance)
    posterior_variance = posterior.variance.squeeze()
    assert torch.allclose(
        posterior_variance, torch.tensor(variances, dtype=torch.float32), atol=1e-5
    )
