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

"""Universal adapter for models to work with BoTorch acquisition functions.

This module provides a unified interface that allows both native BoTorch models
and ALF BaseModel instances to work seamlessly with BoTorch acquisition
functions. This essentially wraps the ALF models to be used as BoTorch models
"""

from typing import TYPE_CHECKING

import torch
from alf_core.model.base_model import BaseModel
from alf_tools.utils.botorch_utils import (
    predictions_to_posterior,
    tensor_to_candidates,
)
from botorch.models.model import Model
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import DiagLinearOperator
from torch import Tensor

if TYPE_CHECKING:
    from botorch.acquisition.objective import PosteriorTransform


class BoTorchModelAdapter(Model):
    """Universal adapter for BoTorch acquisition functions compatibility.

    Accepts either a native BoTorch `Model` (direct pass-through to
    `posterior()`) or an ALF `BaseModel` (adapts `predict()` to BoTorch's
    `posterior()` interface). The adapter detects the model type automatically.

    Models that don't provide prediction variances (e.g., CNNModel, deterministic
    models) cannot be used with BoTorch acquisition functions. Expected Improvement
    and Upper Confidence Bound require uncertainty estimates; attempting to use such
    models raises a `ValueError`.

    Example with BoTorch GP Model:
        >>> from botorch.models import SingleTaskGP
        >>> from alf_tools.optimizer.acquisition_functions.utils.botorch_model_adapter import (
        ...     BoTorchModelAdapter,
        ... )
        >>>
        >>> # Native BoTorch model
        >>> gp_model = SingleTaskGP(train_X, train_Y)
        >>> adapter = BoTorchModelAdapter(gp_model)
        >>> posterior = adapter.posterior(test_X)

    Example with ALF BaseModel:
        >>> from alf_tools.models.botorch_exact_gp_model import BoTorchGPModel
        >>> from alf_tools.optimizer.acquisition_functions.utils.botorch_model_adapter import (
        ...     BoTorchModelAdapter,
        ... )
        >>>
        >>> # ALF BaseModel wrapping BoTorch model
        >>> alf_model = BoTorchGPModel()
        >>> alf_model.train(train_data, val_data)
        >>> adapter = BoTorchModelAdapter(alf_model)
        >>> posterior = adapter.posterior(test_X)

    Example with acquisition function:
        >>> from botorch.acquisition import qExpectedImprovement
        >>> from alf_tools.optimizer.acquisition_functions.utils.botorch_model_adapter import (
        ...     BoTorchModelAdapter,
        ... )
        >>>
        >>> adapter = BoTorchModelAdapter(surrogate_model)
        >>> acq_fn = qExpectedImprovement(model=adapter, best_f=best_value)
        >>> acq_values = acq_fn(candidates)

    Args:
        model: Either a BoTorch Model or an ALF BaseModel instance.
            If BaseModel, it must provide prediction variances.

    Raises:
        ValueError: If a BaseModel doesn't provide variances in predictions.
        TypeError: If the model is neither a BoTorch Model nor ALF BaseModel.
    """

    def __init__(self, model: Model | BaseModel):
        """Initialize the adapter with a model.

        Args:
            model: Either a native BoTorch Model or an ALF BaseModel.

        Raises:
            TypeError: If model is not a BoTorch Model or ALF BaseModel.
        """
        super().__init__()
        self._wrapped_model = model
        self._is_botorch_model = isinstance(model, Model)
        self._is_alf_model = isinstance(model, BaseModel)

        if not (self._is_botorch_model or self._is_alf_model):
            raise TypeError(
                f"Model must be either a BoTorch Model or ALF BaseModel, got {type(model).__name__}"
            )

    def posterior(
        self,
        X: torch.Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: "PosteriorTransform | None" = None,
    ) -> Posterior:
        """Compute the posterior distribution at input points.

        Args:
            X: Input tensor of shape (batch_size, q, d) or (batch_size, d)
                where d is the input dimension and q is number of points.
            output_indices: Optional list of output indices for
                multi-output models.
            observation_noise: Whether to include observation noise in
                predictions. Can be bool or Tensor for observed noise.
            posterior_transform: Optional posterior transformation.

        Returns:
            Posterior distribution with mean and variance at input points.

        Raises:
            ValueError: If ALF BaseModel doesn't provide variances.
        """
        # Case 1: Native BoTorch model - pass through directly
        if self._is_botorch_model:
            return self._wrapped_model.posterior(  # type: ignore[union-attr]
                X=X,
                output_indices=output_indices,
                observation_noise=observation_noise,
                posterior_transform=posterior_transform,
            )

        # Case 2: ALF BaseModel - adapt predict() to posterior()
        # BoTorch passes either 2D (n, d) or 3D (batch, q, d) tensors
        # ALF models expect lists of candidates, so we need to flatten 3D inputs
        # then reshape the output posterior to match BoTorch's expectations

        # Save original shape for later reshaping
        if X.ndim == 3:
            batch_size, q, d = X.shape
            X_2d = X.reshape(batch_size * q, d)
            needs_reshape = True
        else:
            batch_size = 1
            q = X.shape[0]
            X_2d = X
            needs_reshape = False

        # Convert tensor to candidates
        candidates = tensor_to_candidates(X_2d)

        # Get predictions from ALF model
        predictions = self._wrapped_model.predict(candidates)  # type: ignore[union-attr]

        # Check if model provides variances (required for BoTorch
        # acquisition functions)
        if predictions.variances is None:
            raise ValueError(
                f"Model {type(self._wrapped_model).__name__} does not "
                f"provide prediction variances. BoTorch acquisition "
                f"functions require uncertainty estimates (variances) to "
                f"compute acquisition values. "
                f"\n\nModels without variance (e.g., CNNModel, "
                f"deterministic models) cannot be used with BoTorch "
                f"acquisition functions like Expected Improvement or Upper "
                f"Confidence Bound. Consider using a Gaussian Process model "
                f"(e.g., BoTorchGPModel) or another probabilistic model "
                f"that provides uncertainty estimates."
            )

        # Convert predictions to BoTorch posterior
        posterior = predictions_to_posterior(predictions, device=X_2d.device)

        # If input was 3D, reshape the posterior to have proper batch structure
        # BoTorch expects MVN with batch_shape=(batch_size,) and event_shape=(q,)
        if needs_reshape:
            # Reshape mean and covariance to match batch structure
            # Current: MVN with batch_shape=() and event_shape=(batch_size*q,)
            # Target: MVN with batch_shape=(batch_size,) and event_shape=(q,)
            lazy_covar = posterior.mvn.lazy_covariance_matrix
            if not isinstance(lazy_covar, DiagLinearOperator):
                raise NotImplementedError(
                    f"Model {type(self._wrapped_model).__name__} returns a non-diagonal "
                    "covariance. The batch reshape assumes per-point independence; "
                    "cross-candidate correlations are not supported."
                )

            mean = posterior.mvn.mean.reshape(batch_size, q)
            covar_matrix = posterior.mvn.covariance_matrix

            # Create block diagonal covariance for independent evaluations
            # Shape: (batch_size, q, q)
            new_covar = torch.zeros(
                batch_size, q, q, dtype=covar_matrix.dtype, device=covar_matrix.device
            )
            for i in range(batch_size):
                start_idx = i * q
                end_idx = start_idx + q
                new_covar[i] = covar_matrix[start_idx:end_idx, start_idx:end_idx]

            mvn = MultivariateNormal(mean, new_covar)

            posterior = GPyTorchPosterior(mvn)

        return posterior

    @property
    def num_outputs(self) -> int:
        """The number of outputs of the model.

        Raises:
            NotImplementedError: If the model is a multi-output ALF BaseModel.

        Returns:
            Number of outputs. For most models, this is 1 (single-output).
            Multi-output models should override this.
        """
        # If native BoTorch model, use its num_outputs
        if self._is_botorch_model:
            return int(self._wrapped_model.num_outputs)  # type: ignore[union-attr, no-any-return]

        # For ALF BaseModel, assume single output (most common case)
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        """The batch shape of the model.

        This is a batch shape from an I/O perspective, independent of the
        internal representation of the model.

        Returns:
            Batch shape. Empty for most models (no batching).
        """
        # If native BoTorch model, use its batch_shape
        if self._is_botorch_model:
            return self._wrapped_model.batch_shape  # type: ignore[union-attr]

        # For ALF BaseModel, assume no batch dimension (most common case)
        return torch.Size([])
