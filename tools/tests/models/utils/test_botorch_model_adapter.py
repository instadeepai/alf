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

"""Tests for BoTorchModelAdapter.

This module tests the BoTorchModelAdapter class which provides a unified
interface for both native BoTorch models and ALF BaseModel instances to work
with BoTorch acquisition functions.

Note: This module uses shared fixtures from tools/tests/conftest.py including:
- mock_alf_model_with_variances
- mock_alf_model_without_variances
- botorch_gp_model
- test_candidates_2d
- test_tensor_2d
- test_tensor_3d
"""

import pytest
import torch
from alf_tools.models.utils.botorch_model_adapter import BoTorchModelAdapter
from alf_tools.utils.botorch_utils import candidates_to_tensor
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior

# =============================================================================
# Initialization Tests
# =============================================================================


def test_adapter_init_with_botorch_model(botorch_gp_model):
    """Test initialization with a native BoTorch model."""
    adapter = BoTorchModelAdapter(botorch_gp_model)

    assert adapter._wrapped_model is botorch_gp_model
    assert adapter._is_botorch_model is True
    assert adapter._is_alf_model is False


def test_adapter_init_with_alf_model(mock_alf_model_with_variances):
    """Test initialization with an ALF BaseModel."""
    adapter = BoTorchModelAdapter(mock_alf_model_with_variances)

    assert adapter._wrapped_model is mock_alf_model_with_variances
    assert adapter._is_botorch_model is False
    assert adapter._is_alf_model is True


def test_adapter_init_with_invalid_model():
    """Test that initialization fails with an invalid model type."""
    invalid_model = "not_a_model"

    with pytest.raises(TypeError, match="must be either a BoTorch Model or ALF BaseModel"):
        BoTorchModelAdapter(invalid_model)


def test_adapter_init_with_none():
    """Test that initialization fails with None."""
    with pytest.raises(TypeError, match="must be either a BoTorch Model or ALF BaseModel"):
        BoTorchModelAdapter(None)


# =============================================================================
# Posterior Tests - BoTorch Model
# =============================================================================


def test_posterior_botorch_model_2d(botorch_gp_model, test_tensor_2d):
    """Test posterior computation with native BoTorch model and 2D input."""
    adapter = BoTorchModelAdapter(botorch_gp_model)

    posterior = adapter.posterior(test_tensor_2d)

    assert isinstance(posterior, Posterior)
    assert isinstance(posterior, GPyTorchPosterior)
    # Mean shape can be (3,) or (3, 1)
    assert posterior.mean.shape[0] == 3
    # Check that we got reasonable values (not NaN, not inf)
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_posterior_botorch_model_3d(botorch_gp_model, test_tensor_3d):
    """Test posterior computation with native BoTorch model and 3D input."""
    adapter = BoTorchModelAdapter(botorch_gp_model)

    # 3D input: (batch_size=2, q=3, d=2)
    posterior = adapter.posterior(test_tensor_3d)

    assert isinstance(posterior, Posterior)
    # BoTorch handles batching internally
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_posterior_botorch_model_passes_arguments(botorch_gp_model, test_tensor_2d):
    """Test that posterior arguments are passed through to BoTorch model."""
    adapter = BoTorchModelAdapter(botorch_gp_model)

    # Call with observation_noise parameter
    posterior_with_noise = adapter.posterior(test_tensor_2d, observation_noise=True)
    posterior_without_noise = adapter.posterior(test_tensor_2d, observation_noise=False)

    # With noise should have higher variance
    assert (posterior_with_noise.variance > posterior_without_noise.variance).any()


# =============================================================================
# Posterior Tests - ALF Model
# =============================================================================


def test_posterior_alf_model_2d(mock_alf_model_with_variances, test_tensor_2d):
    """Test posterior computation with ALF model and 2D input."""
    adapter = BoTorchModelAdapter(mock_alf_model_with_variances)

    posterior = adapter.posterior(test_tensor_2d)

    assert isinstance(posterior, Posterior)
    assert isinstance(posterior, GPyTorchPosterior)

    # Check shape - mean should be (3,) or (3, 1)
    mean_flat = posterior.mean.flatten()
    assert mean_flat.shape[0] == 3

    # Check values match what mock model returns (0+5, 1+5, 2+5)
    expected_means = torch.tensor([5.0, 6.0, 7.0], dtype=torch.float64)
    assert torch.allclose(mean_flat, expected_means, atol=0.01)

    # Check variances match mock model (all 0.2)
    variance_flat = posterior.variance.flatten()
    assert torch.allclose(
        variance_flat, torch.tensor([0.2, 0.2, 0.2], dtype=torch.float64), atol=0.01
    )


def test_posterior_alf_model_3d(mock_alf_model_with_variances, test_tensor_3d):
    """Test posterior computation with ALF model and 3D input."""
    adapter = BoTorchModelAdapter(mock_alf_model_with_variances)

    # 3D input: (batch_size=2, q=3, d=2)
    posterior = adapter.posterior(test_tensor_3d)

    assert isinstance(posterior, Posterior)
    assert isinstance(posterior, GPyTorchPosterior)

    # After reshaping, should have batch_shape=(2,) and event_shape=(3,)
    # Mean shape can be [2, 3] or [2, 3, 1] depending on output dimension handling
    assert posterior.mean.shape in [torch.Size([2, 3]), torch.Size([2, 3, 1])]
    assert posterior.covariance_matrix.shape == torch.Size([2, 3, 3])

    # Check that covariance matrices are block diagonal
    # (independence assumption between batch elements)
    covar = posterior.covariance_matrix
    for i in range(2):
        # Diagonal elements should be non-zero (variances)
        assert (torch.diag(covar[i]) > 0).all()


def test_posterior_alf_model_without_variances_raises_error(
    mock_alf_model_without_variances, test_tensor_2d
):
    """Test that ALF model without variances raises informative error."""
    adapter = BoTorchModelAdapter(mock_alf_model_without_variances)

    with pytest.raises(ValueError, match="does not provide prediction variances"):
        adapter.posterior(test_tensor_2d)

    # Check that error message is informative
    with pytest.raises(
        ValueError,
        match="BoTorch acquisition functions require uncertainty estimates",
    ):
        adapter.posterior(test_tensor_2d)


def test_posterior_alf_model_preserves_batch_structure(mock_alf_model_with_variances):
    """Test that 3D reshaping preserves batch structure correctly."""
    adapter = BoTorchModelAdapter(mock_alf_model_with_variances)

    # Create specific 3D tensor: batch_size=3, q=2, d=2
    X = torch.tensor(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            [[0.9, 1.0], [1.1, 1.2]],
        ],
        dtype=torch.float32,
    )

    posterior = adapter.posterior(X)

    # Should have batch_shape=(3,) and event_shape=(2,)
    # Mean shape can be [3, 2] or [3, 2, 1] depending on output dimension handling
    assert posterior.mean.shape in [torch.Size([3, 2]), torch.Size([3, 2, 1])]

    # Each batch should have independent covariance
    covar = posterior.covariance_matrix
    assert covar.shape == torch.Size([3, 2, 2])

    # Covariance should be diagonal (independent points)
    for i in range(3):
        # Off-diagonal elements should be zero or very small
        off_diag = covar[i] - torch.diag(torch.diag(covar[i]))
        assert torch.allclose(off_diag, torch.zeros_like(off_diag), atol=1e-5)


# =============================================================================
# Property Tests
# =============================================================================


def test_num_outputs_botorch_model(botorch_gp_model):
    """Test num_outputs property with BoTorch model."""
    adapter = BoTorchModelAdapter(botorch_gp_model)

    assert adapter.num_outputs == botorch_gp_model.num_outputs


def test_num_outputs_alf_model(mock_alf_model_with_variances):
    """Test num_outputs property with ALF model (defaults to 1)."""
    adapter = BoTorchModelAdapter(mock_alf_model_with_variances)

    # ALF models default to single output
    assert adapter.num_outputs == 1


def test_batch_shape_botorch_model(botorch_gp_model):
    """Test batch_shape property with BoTorch model."""
    adapter = BoTorchModelAdapter(botorch_gp_model)

    assert adapter.batch_shape == botorch_gp_model.batch_shape


def test_batch_shape_alf_model(mock_alf_model_with_variances):
    """Test batch_shape property with ALF model (defaults to empty)."""
    adapter = BoTorchModelAdapter(mock_alf_model_with_variances)

    # ALF models default to no batch dimension
    assert adapter.batch_shape == torch.Size([])


# =============================================================================
# Integration Tests
# =============================================================================


def test_adapter_with_candidates_conversion(mock_alf_model_with_variances, test_candidates_2d):
    """Test full workflow: candidates -> tensor -> posterior."""
    adapter = BoTorchModelAdapter(mock_alf_model_with_variances)

    # Convert candidates to tensor
    X = candidates_to_tensor(test_candidates_2d)

    # Get posterior
    posterior = adapter.posterior(X)

    assert isinstance(posterior, Posterior)
    assert posterior.mean.shape[0] == len(test_candidates_2d)


def test_adapter_posterior_statistics(mock_alf_model_with_variances, test_tensor_2d):
    """Test that posterior statistics are computed correctly."""
    adapter = BoTorchModelAdapter(mock_alf_model_with_variances)

    posterior = adapter.posterior(test_tensor_2d)

    # Can sample from posterior
    samples = posterior.sample(torch.Size([10]))
    assert samples.shape[0] == 10

    # Can compute mean and variance
    mean = posterior.mean
    variance = posterior.variance

    assert torch.isfinite(mean).all()
    assert torch.isfinite(variance).all()
    assert (variance > 0).all()  # Variances should be positive


def test_adapter_empty_posterior_transform(botorch_gp_model, test_tensor_2d):
    """Test that posterior_transform parameter is accepted (even if None)."""
    adapter = BoTorchModelAdapter(botorch_gp_model)

    # Should work with None posterior_transform
    posterior = adapter.posterior(test_tensor_2d, posterior_transform=None)

    assert isinstance(posterior, Posterior)


def test_adapter_different_input_dimensions(mock_alf_model_with_variances):
    """Test adapter works with different input dimensions."""
    adapter = BoTorchModelAdapter(mock_alf_model_with_variances)

    # Test with different numbers of points
    for n_points in [1, 5, 10, 20]:
        X = torch.rand(n_points, 2, dtype=torch.float32)
        posterior = adapter.posterior(X)
        assert posterior.mean.shape[0] == n_points


def test_adapter_with_single_point(mock_alf_model_with_variances):
    """Test adapter with a single point (edge case)."""
    adapter = BoTorchModelAdapter(mock_alf_model_with_variances)

    X = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    posterior = adapter.posterior(X)

    assert posterior.mean.shape[0] == 1
    assert torch.isfinite(posterior.mean).all()


def test_adapter_deterministic_predictions(mock_alf_model_with_variances):
    """Test that same input gives same output (deterministic)."""
    adapter = BoTorchModelAdapter(mock_alf_model_with_variances)

    X = torch.tensor([[0.5, 0.5]], dtype=torch.float32)

    posterior1 = adapter.posterior(X)
    posterior2 = adapter.posterior(X)

    assert torch.allclose(posterior1.mean, posterior2.mean)
    assert torch.allclose(posterior1.variance, posterior2.variance)
