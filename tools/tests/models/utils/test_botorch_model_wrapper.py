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

"""Tests for BotorchModelWrapper.

This module tests the BotorchModelWrapper class which provides a unified
interface for both native BoTorch models and ALF BaseModel instances to work
with BoTorch acquisition functions.

Note: This module uses shared fixtures from tools/tests/conftest.py including:
- mock_alf_model_with_variances
- mock_alf_model_without_variances
- botorch_gp_model
- trained_gp_model
- test_candidates_2d
- test_tensor_2d
- test_tensor_3d
"""

import numpy as np
import pytest
import torch
from alf_core import Candidate, LabelledCandidates, Modality
from alf_tools.models.gp import FeaturizerConfig, GPModel, GPTrainConfig
from alf_tools.models.utils.botorch_utils import candidates_to_tensor
from alf_tools.optimizer.acquisition_functions.utils.botorch_model_wrapper import (
    BotorchModelWrapper,
)
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior

# =============================================================================
# Initialization Tests
# =============================================================================


def test_wrapper_init_with_botorch_model(botorch_gp_model):
    """Test initialization with a native BoTorch model."""
    wrapper = BotorchModelWrapper(botorch_gp_model)

    assert wrapper._wrapped_model is botorch_gp_model
    assert wrapper._is_botorch_model is True


def test_wrapper_init_with_alf_model(mock_alf_model_with_variances):
    """Test initialization with an ALF BaseModel."""
    wrapper = BotorchModelWrapper(mock_alf_model_with_variances)

    assert wrapper._wrapped_model is mock_alf_model_with_variances
    assert wrapper._is_botorch_model is False


def test_wrapper_init_with_invalid_model():
    """Test that initialization fails with an invalid model type."""
    invalid_model = "not_a_model"

    with pytest.raises(TypeError, match="must be either a BoTorch Model or ALF BaseModel"):
        BotorchModelWrapper(invalid_model)


def test_wrapper_init_with_none():
    """Test that initialization fails with None."""
    with pytest.raises(TypeError, match="must be either a BoTorch Model or ALF BaseModel"):
        BotorchModelWrapper(None)


# =============================================================================
# Posterior Tests - BoTorch Model
# =============================================================================


def test_posterior_botorch_model_2d(botorch_gp_model, test_tensor_2d):
    """Test posterior computation with native BoTorch model and 2D input."""
    wrapper = BotorchModelWrapper(botorch_gp_model)

    posterior = wrapper.posterior(test_tensor_2d)

    assert isinstance(posterior, Posterior)
    assert isinstance(posterior, GPyTorchPosterior)
    # Mean shape can be (3,) or (3, 1)
    assert posterior.mean.shape[0] == 3
    # Check that we got reasonable values (not NaN, not inf)
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_posterior_botorch_model_3d(botorch_gp_model, test_tensor_3d):
    """Test posterior computation with native BoTorch model and 3D input."""
    wrapper = BotorchModelWrapper(botorch_gp_model)

    # 3D input: (batch_size=2, q=3, d=2)
    posterior = wrapper.posterior(test_tensor_3d)

    assert isinstance(posterior, Posterior)
    # BoTorch handles batching internally
    assert torch.isfinite(posterior.mean).all()
    assert torch.isfinite(posterior.variance).all()


def test_posterior_botorch_model_passes_arguments(botorch_gp_model, test_tensor_2d):
    """Test that posterior arguments are passed through to BoTorch model."""
    wrapper = BotorchModelWrapper(botorch_gp_model)

    # Call with observation_noise parameter
    posterior_with_noise = wrapper.posterior(test_tensor_2d, observation_noise=True)
    posterior_without_noise = wrapper.posterior(test_tensor_2d, observation_noise=False)

    # With noise should have higher variance
    assert (posterior_with_noise.variance > posterior_without_noise.variance).any()


# =============================================================================
# Posterior Tests - ALF Model
# =============================================================================


def test_posterior_alf_model_2d(mock_alf_model_with_variances, test_tensor_2d):
    """Test posterior computation with ALF model and 2D input."""
    wrapper = BotorchModelWrapper(mock_alf_model_with_variances)

    posterior = wrapper.posterior(test_tensor_2d)

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
    wrapper = BotorchModelWrapper(mock_alf_model_with_variances)

    # 3D input: (batch_size=2, q=3, d=2)
    posterior = wrapper.posterior(test_tensor_3d)

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
    wrapper = BotorchModelWrapper(mock_alf_model_without_variances)

    with pytest.raises(ValueError, match="does not provide prediction variances"):
        wrapper.posterior(test_tensor_2d)

    # Check that error message is informative
    with pytest.raises(
        ValueError,
        match="BoTorch acquisition functions require uncertainty estimates",
    ):
        wrapper.posterior(test_tensor_2d)


def test_posterior_alf_model_rejects_posterior_transform(
    mock_alf_model_with_variances, test_tensor_2d
):
    """Passing a posterior_transform for an ALF model raises NotImplementedError.

    Silently dropping the transform would corrupt acquisition values (e.g. a
    minimisation transform would be ignored), so the wrapper must fail loudly.
    """
    wrapper = BotorchModelWrapper(mock_alf_model_with_variances)

    transform = ScalarizedPosteriorTransform(weights=torch.tensor([-1.0]))
    with pytest.raises(NotImplementedError, match="posterior_transform"):
        wrapper.posterior(test_tensor_2d, posterior_transform=transform)


def test_posterior_alf_model_rejects_output_indices(mock_alf_model_with_variances, test_tensor_2d):
    """Passing output_indices for an ALF model raises NotImplementedError."""
    wrapper = BotorchModelWrapper(mock_alf_model_with_variances)

    with pytest.raises(NotImplementedError, match="output_indices"):
        wrapper.posterior(test_tensor_2d, output_indices=[0])


@pytest.mark.filterwarnings("ignore:num_acquisitions:UserWarning")
def test_posterior_delegates_to_inner_joint_model(trained_gp_model):
    """For joint-capable ALF models, posterior() uses the inner GP's joint covariance."""
    from linear_operator.operators import DiagLinearOperator

    wrapper = BotorchModelWrapper(trained_gp_model.model)
    # 3 distinct points -> a real GP returns a non-diagonal joint covariance.
    X = torch.linspace(0.0, 1.0, 6, dtype=torch.float64).reshape(1, 3, 2)
    post = wrapper.posterior(X)
    covar = post.mvn.lazy_covariance_matrix
    assert not isinstance(covar, DiagLinearOperator)


def test_posterior_marginal_only_stays_diagonal(mock_alf_model_with_variances):
    """Marginal-only models keep the diagonal predict()-based posterior."""
    from linear_operator.operators import DiagLinearOperator

    wrapper = BotorchModelWrapper(mock_alf_model_with_variances)
    X = torch.zeros(1, 3, 2, dtype=torch.float64)
    post = wrapper.posterior(X)
    assert isinstance(post.mvn.lazy_covariance_matrix, DiagLinearOperator)


def test_posterior_alf_model_preserves_batch_structure(mock_alf_model_with_variances):
    """Test that 3D reshaping preserves batch structure correctly."""
    wrapper = BotorchModelWrapper(mock_alf_model_with_variances)

    # Create specific 3D tensor: batch_size=3, q=2, d=2
    X = torch.tensor(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6], [0.7, 0.8]],
            [[0.9, 1.0], [1.1, 1.2]],
        ],
        dtype=torch.float32,
    )

    posterior = wrapper.posterior(X)

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
    wrapper = BotorchModelWrapper(botorch_gp_model)

    assert wrapper.num_outputs == botorch_gp_model.num_outputs


def test_num_outputs_marginal_only(mock_alf_model_with_variances):
    """num_outputs is 1 for marginal-only (single-output predict) ALF models."""
    wrapper = BotorchModelWrapper(mock_alf_model_with_variances)
    assert wrapper.num_outputs == 1


def test_batch_shape_botorch_model(botorch_gp_model):
    """Test batch_shape property with BoTorch model."""
    wrapper = BotorchModelWrapper(botorch_gp_model)

    assert wrapper.batch_shape == botorch_gp_model.batch_shape


def test_batch_shape_marginal_only(mock_alf_model_with_variances):
    """batch_shape is empty for marginal-only ALF models."""
    wrapper = BotorchModelWrapper(mock_alf_model_with_variances)
    assert wrapper.batch_shape == torch.Size([])


@pytest.mark.filterwarnings("ignore:num_acquisitions:UserWarning")
def test_num_outputs_alf_model_with_botorch_model(trained_gp_model):
    """num_outputs delegates to the wrapped ALF model's trained `botorch_model`."""
    wrapper = BotorchModelWrapper(trained_gp_model.model)

    assert wrapper.num_outputs == 1


@pytest.mark.filterwarnings("ignore:num_acquisitions:UserWarning")
def test_batch_shape_alf_model_with_botorch_model(trained_gp_model):
    """batch_shape delegates to the wrapped ALF model's trained `botorch_model`."""
    wrapper = BotorchModelWrapper(trained_gp_model.model)

    assert wrapper.batch_shape == torch.Size([])


def test_provides_joint_posterior_botorch_model(botorch_gp_model):
    """Native BoTorch models are joint-capable."""
    wrapper = BotorchModelWrapper(botorch_gp_model)
    assert wrapper.provides_joint_posterior is True


@pytest.mark.filterwarnings("ignore:num_acquisitions:UserWarning")
def test_provides_joint_posterior_alf_with_botorch_model(trained_gp_model):
    """ALF models exposing a trained `botorch_model` are joint-capable."""
    wrapper = BotorchModelWrapper(trained_gp_model.model)
    assert wrapper.provides_joint_posterior is True


def test_provides_joint_posterior_marginal_only(mock_alf_model_with_variances):
    """Predict-only ALF models are not joint-capable."""
    wrapper = BotorchModelWrapper(mock_alf_model_with_variances)
    assert wrapper.provides_joint_posterior is False


def test_num_outputs_untrained_gp_model_defaults_to_one():
    """An untrained GPModel has no botorch_model yet; num_outputs defaults to 1."""
    wrapper = BotorchModelWrapper(GPModel(device="cpu"))
    assert wrapper.num_outputs == 1


def test_batch_shape_untrained_gp_model_defaults_to_empty():
    """An untrained GPModel has no botorch_model yet; batch_shape defaults to empty."""
    wrapper = BotorchModelWrapper(GPModel(device="cpu"))
    assert wrapper.batch_shape == torch.Size([])


# =============================================================================
# Integration Tests
# =============================================================================


def test_wrapper_with_candidates_conversion(mock_alf_model_with_variances, test_candidates_2d):
    """Test full workflow: candidates -> tensor -> posterior."""
    wrapper = BotorchModelWrapper(mock_alf_model_with_variances)

    # Convert candidates to tensor
    X = candidates_to_tensor(test_candidates_2d)

    # Get posterior
    posterior = wrapper.posterior(X)

    assert isinstance(posterior, Posterior)
    assert posterior.mean.shape[0] == len(test_candidates_2d)


def test_wrapper_posterior_statistics(mock_alf_model_with_variances, test_tensor_2d):
    """Test that posterior statistics are computed correctly."""
    wrapper = BotorchModelWrapper(mock_alf_model_with_variances)

    posterior = wrapper.posterior(test_tensor_2d)

    # Can sample from posterior
    samples = posterior.sample(torch.Size([10]))
    assert samples.shape[0] == 10

    # Can compute mean and variance
    mean = posterior.mean
    variance = posterior.variance

    assert torch.isfinite(mean).all()
    assert torch.isfinite(variance).all()
    assert (variance > 0).all()  # Variances should be positive


def test_wrapper_empty_posterior_transform(botorch_gp_model, test_tensor_2d):
    """Test that posterior_transform parameter is accepted (even if None)."""
    wrapper = BotorchModelWrapper(botorch_gp_model)

    # Should work with None posterior_transform
    posterior = wrapper.posterior(test_tensor_2d, posterior_transform=None)

    assert isinstance(posterior, Posterior)


def test_wrapper_different_input_dimensions(mock_alf_model_with_variances):
    """Test wrapper works with different input dimensions."""
    wrapper = BotorchModelWrapper(mock_alf_model_with_variances)

    # Test with different numbers of points
    for n_points in [1, 5, 10, 20]:
        X = torch.rand(n_points, 2, dtype=torch.float32)
        posterior = wrapper.posterior(X)
        assert posterior.mean.shape[0] == n_points


def test_wrapper_with_single_point(mock_alf_model_with_variances):
    """Test wrapper with a single point (edge case)."""
    wrapper = BotorchModelWrapper(mock_alf_model_with_variances)

    X = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    posterior = wrapper.posterior(X)

    assert posterior.mean.shape[0] == 1
    assert torch.isfinite(posterior.mean).all()


def test_wrapper_deterministic_predictions(mock_alf_model_with_variances):
    """Test that same input gives same output (deterministic)."""
    wrapper = BotorchModelWrapper(mock_alf_model_with_variances)

    X = torch.tensor([[0.5, 0.5]], dtype=torch.float32)

    posterior1 = wrapper.posterior(X)
    posterior2 = wrapper.posterior(X)

    assert torch.allclose(posterior1.mean, posterior2.mean)
    assert torch.allclose(posterior1.variance, posterior2.variance)


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.filterwarnings("ignore:num_acquisitions:UserWarning")
@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in multiply:RuntimeWarning:alf_core.utils.metrics.regression"
)
def test_wrapper_integration_with_real_gp_model():
    """Integration smoke test: real GPModel -> BotorchModelWrapper -> posterior().

    Training metrics on the tiny dataset emit expected alf_core metric
    warnings (regret-metric fallback and a rank-space ECE `inf * sqrt(0)`
    RuntimeWarning), which are filtered above.
    """
    # Train a real GPModel on tiny data
    X_train = np.array([[0.1, 0.2], [0.4, 0.5], [0.7, 0.8], [0.3, 0.6]], dtype=np.float32)
    y_train = np.array([1.0, 2.0, 1.5, 1.8])
    candidates = [Candidate(data=x, modality=Modality.TABULAR) for x in X_train]
    train_data = LabelledCandidates(candidates=candidates, labels=y_train)

    gp_model = GPModel(
        featurizer_config=FeaturizerConfig(featurizer_type="precomputed"),
        train_config=GPTrainConfig(num_iterations=10, optimizer_type="scipy"),
        device="cpu",
    )
    gp_model.train(train_data)

    # Wrap in wrapper
    wrapper = BotorchModelWrapper(gp_model)

    # Call posterior with 2D input
    X_test = torch.tensor([[0.2, 0.3], [0.5, 0.6]], dtype=torch.float32)
    posterior = wrapper.posterior(X_test)

    # Verify posterior has correct shape and finite values
    assert posterior is not None
    mean = posterior.mean
    assert mean.shape[0] == 2
    assert torch.all(torch.isfinite(mean))
