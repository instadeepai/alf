# Copyright 2026 InstaDeep Ltd. All rights reserved.
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

"""Universal wrapper for models to work with BoTorch acquisition functions.

This module provides a unified interface that allows both native BoTorch models
and ALF BaseModel instances to work seamlessly with BoTorch acquisition
functions. This essentially wraps the ALF models to be used as BoTorch models
"""

from typing import TYPE_CHECKING

import torch
from alf_core.model.base_model import BaseModel
from botorch.models.model import Model as BotorchModel
from botorch.posteriors import Posterior
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import DiagLinearOperator
from torch import Tensor

from alf_tools.models.utils.botorch_utils import (
    predictions_to_posterior,
    tensor_to_candidates,
)

if TYPE_CHECKING:
    from botorch.acquisition.objective import PosteriorTransform


def resolve_botorch_model(model: "BotorchModel | BaseModel") -> "BotorchModel | None":
    """Return the joint-posterior BoTorch `Model` a surrogate provides, if any.

    This is the single capability check used across the BoTorch integration:
    native BoTorch `Model` instances are returned as-is; ALF `BaseModel`
    instances are expected to expose a trained `botorch_model` attribute (the
    documented joint-posterior contract), which is returned when available.

    Args:
        model: A native BoTorch `Model` or an ALF `BaseModel`.

    Returns:
        The joint-posterior BoTorch model, or `None` for marginal-only models —
        those without a `botorch_model`, or whose `botorch_model` is unavailable
        because the model is untrained.
    """
    if isinstance(model, BotorchModel):
        return model
    try:
        inner = getattr(model, "botorch_model", None)
    except RuntimeError:
        return None
    return inner if isinstance(inner, BotorchModel) else None


class BotorchModelWrapper(BotorchModel):
    """Universal wrapper for BoTorch acquisition functions compatibility.

    Accepts either a native BoTorch `Model` (direct pass-through to
    `posterior()`) or an ALF `BaseModel` (adapts `predict()` to BoTorch's
    `posterior()` interface). The wrapper detects the model type automatically.

    ALF models split into two posterior paths:

    Joint-capable ALF models that expose a trained `botorch_model` attribute
    (e.g. `GPModel`) delegate `posterior()` to the inner
    BoTorch model's native `posterior()`. This yields a true joint covariance
    and fully honours `observation_noise`, `posterior_transform` and
    `output_indices`, all of which are forwarded to the inner model. q-based
    acquisitions such as `qLogNoisyExpectedImprovement` are therefore evaluated
    with the exact joint covariance, and `num_outputs`/`batch_shape` are
    delegated to the underlying BoTorch model.

    Marginal-only (predict-only) ALF models route `posterior()` through
    `predict()`, which yields a diagonal (per-point independent) posterior.
    q-based acquisitions over q>1 are evaluated under a per-point independence
    approximation — baseline-candidate correlations are zero — rather than with
    the exact joint covariance. On this path the `observation_noise` flag is
    ignored: the returned posterior always carries whatever variance `predict()`
    reports. `posterior_transform` and `output_indices` cannot be honoured and
    raise `NotImplementedError` if passed. Such models report the single-output
    defaults `num_outputs=1` and `batch_shape=torch.Size([])`.

    Models that don't provide prediction variances (e.g., CNNModel, deterministic
    models) cannot be used with BoTorch acquisition functions. Expected Improvement
    and Upper Confidence Bound require uncertainty estimates; attempting to use such
    models raises a `ValueError`.

    Args:
        model: Either a BoTorch ``BotorchModel`` or an ALF ``BaseModel`` instance.
            If ``BaseModel``, it must provide prediction variances.

    Raises:
        TypeError: If the model is neither a BoTorch Model nor ALF BaseModel.
        ValueError: If a BaseModel doesn't provide variances in predictions
            (raised during `posterior()`).
    """

    def __init__(self, model: BotorchModel | BaseModel):
        """Initialize the wrapper with a model.

        Args:
            model: Either a native BoTorch Model or an ALF BaseModel.

        Raises:
            TypeError: If model is not a BoTorch Model or ALF BaseModel.
        """
        if not isinstance(model, (BotorchModel, BaseModel)):
            raise TypeError(
                f"Model must be either a BoTorch Model or ALF BaseModel, got {type(model).__name__}"
            )
        super().__init__()
        self._wrapped_model = model
        self._is_botorch_model = isinstance(model, BotorchModel)

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
                multi-output models. Not supported for ALF models.
            observation_noise: Whether to include observation noise in
                predictions. Can be bool or Tensor for observed noise.
                Ignored for ALF models — `predict()` decides the variance
                semantics (noise-inclusive for `GPModel`).
            posterior_transform: Optional posterior transformation. Not
                supported for ALF models.

        Returns:
            Posterior distribution with mean and variance at input points.

        Raises:
            ValueError: If ALF BaseModel doesn't provide variances.
            NotImplementedError: If `posterior_transform` or `output_indices`
                is passed for an ALF BaseModel — `predict()`-based posteriors
                cannot apply them, and ignoring them silently would corrupt
                acquisition values (e.g. a minimisation transform would be
                dropped).
        """
        # Case 1: Native BoTorch model - pass through directly
        if self._is_botorch_model:
            return self._wrapped_model.posterior(  # type: ignore[union-attr]
                X=X,
                output_indices=output_indices,
                observation_noise=observation_noise,
                posterior_transform=posterior_transform,
            )

        # Case 1b: ALF model exposing a trained joint posterior -> delegate to it
        # for a true joint covariance (honours observation_noise/transforms).
        # Cast X (and tensor observation_noise) to the inner model's dtype: ALF GP
        # models train in double by default, whereas BoTorch acquisitions hand us
        # X in single precision, and SingleTaskGP rejects a dtype mismatch. The
        # `next(..., X)` default keeps a (degenerate) parameter-free inner model
        # from raising StopIteration.
        inner = self._inner_botorch_model()
        if inner is not None:
            inner_dtype = next(inner.parameters(), X).dtype
            if isinstance(observation_noise, Tensor):
                observation_noise = observation_noise.to(inner_dtype)
            return inner.posterior(
                X=X.to(inner_dtype),
                output_indices=output_indices,
                observation_noise=observation_noise,
                posterior_transform=posterior_transform,
            )

        # Case 2: marginal-only ALF BaseModel - adapt predict() to a diagonal
        # (per-point independent) posterior.
        if posterior_transform is not None or output_indices is not None:
            raise NotImplementedError(
                "posterior_transform and output_indices are not supported when "
                "adapting an ALF BaseModel: posterior() routes through predict(), "
                "which cannot apply them. Silently ignoring them would corrupt "
                "acquisition values (e.g. a minimisation transform would be dropped)."
            )
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
                f"(e.g., GPModel) or another probabilistic model "
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
            # Read the stored diagonal directly — O(B*q) — instead of
            # materialising the dense (B*q, B*q) covariance matrix.
            diag_vals = lazy_covar.diagonal().reshape(batch_size, q)
            mvn = MultivariateNormal(mean, DiagLinearOperator(diag_vals))

            posterior = GPyTorchPosterior(mvn)

        return posterior

    def _inner_botorch_model(self) -> BotorchModel | None:
        """Return the wrapped ALF model's trained `botorch_model`, if available.

        Returns:
            The underlying BoTorch model when the wrapped ALF model exposes
            a trained `botorch_model`. Returns `None` when the attribute is
            absent, raises `RuntimeError` on access (untrained model), or
            does not hold a BoTorch `Model`.
        """
        return resolve_botorch_model(self._wrapped_model)

    # CONTRACT: Any ALF model that can produce a TRUE joint posterior (cross-point
    # covariance) MUST expose a `botorch_model` attribute returning a trained
    # BoTorch `Model`. The wrapper consumes it for `num_outputs`, `batch_shape`,
    # and `posterior()`. Models without it are treated as marginal-only and get a
    # diagonal (per-point independent) posterior built from `predict()`.
    # TODO: if this duck-typed contract grows fragile, promote it to an explicit
    # runtime_checkable Protocol (JointPosteriorProvider).
    @property
    def provides_joint_posterior(self) -> bool:
        """Whether this model can produce a true joint posterior over q>1 points.

        Returns:
            `True` for native BoTorch models and for ALF models exposing a
            trained `botorch_model`; `False` for marginal-only `predict()`-based
            models (and for untrained models, which cannot yet supply one).
        """
        if self._is_botorch_model:
            return True
        return self._inner_botorch_model() is not None

    @property
    def num_outputs(self) -> int:
        """The number of outputs of the model.

        Returns:
            Number of outputs for native BoTorch models, or for ALF models
            exposing a trained `botorch_model` (delegated to it). Returns 1 for
            marginal-only ALF models (single-output `predict()`).
        """
        if self._is_botorch_model:
            return int(self._wrapped_model.num_outputs)  # type: ignore[union-attr, no-any-return]
        inner = self._inner_botorch_model()
        if inner is not None:
            return int(inner.num_outputs)
        # Marginal-only model: ALF predict() returns a 1-D means array (ALF is
        # single-objective), so the wrapped model is single-output by construction.
        return 1

    @property
    def batch_shape(self) -> torch.Size:
        """The batch shape of the model.

        Returns:
            Batch shape for native BoTorch models, or for ALF models
            exposing a trained `botorch_model` (delegated to it). Returns an
            empty batch shape for marginal-only ALF models.
        """
        if self._is_botorch_model:
            return self._wrapped_model.batch_shape  # type: ignore[union-attr]
        inner = self._inner_botorch_model()
        if inner is not None:
            return inner.batch_shape
        # Marginal-only model: no batched models -> empty batch shape.
        return torch.Size([])
