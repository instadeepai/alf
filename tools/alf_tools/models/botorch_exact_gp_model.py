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

"""BoTorch-based Gaussian Process models for ALF.

This module provides GP models built on BoTorch's SingleTaskGP, which offers:
- Better default hyperparameter priors
- Built-in input/output transforms (normalisation, standardisation)
- Simplified model fitting with fit_gpytorch_mll
- Seamless integration with BoTorch acquisition functions
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from alf_core import BaseModel, Candidate, LabelledCandidates, Predictions
from alf_core.model.base_model import BaseTrainConfig
from alf_core.model.normaliser import InputNormaliser
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim.fit import fit_gpytorch_mll_torch
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import Adam

from alf_tools.models.gp import GPModelConfig
from alf_tools.models.utils.config_utils import build_from_target
from alf_tools.models.utils.torch_utils import get_device
from alf_tools.utils.botorch_utils import candidates_to_tensor

logger = logging.getLogger("alf-tools")


@dataclass
class BoTorchTrainConfig(BaseTrainConfig):
    """Training configuration for BoTorchGPModel.

    Args:
        normalise_inputs: Whether to normalise inputs to [0, 1]. Default: True.
        standardise_outputs: Whether to standardise outputs. Default: True.
        learning_rate: Learning rate for Adam optimizer. Default: 0.1.
            Only used when optimizer='torch'.
        num_iterations: Number of optimisation iterations. Default: 100.
        optimizer: Optimisation backend. 'scipy' uses L-BFGS-B (default);
            'torch' uses Adam.
        max_attempts: Maximum fitting attempts on numerical failure. Default: 5.
        device: Device string ('cpu', 'cuda'). None auto-detects. Default: None.
        dtype: Tensor dtype. Default: torch.float32.
    """

    normalise_inputs: bool = True
    standardise_outputs: bool = True
    learning_rate: float = 0.1
    num_iterations: int = 100
    optimizer: Literal["scipy", "torch"] = "scipy"
    max_attempts: int = 5
    device: str | None = None
    dtype: torch.dtype = torch.float32


class BoTorchGPModel(BaseModel):
    """Gaussian Process model using BoTorch's SingleTaskGP.

    This model wraps BoTorch's SingleTaskGP, which is built on GPyTorch but provides
    better defaults and easier integration with BoTorch acquisition functions.

    Key Features:
    - Modern hyperparameter priors from Hvarfner et al. 2024
    - Automatic output standardisation (zero mean, unit variance)
    - Efficient model fitting with fit_gpytorch_mll
    - Compatible with all BoTorch acquisition functions

    The model expects continuous inputs (tensors) and works best when:
    - Inputs are normalised to [0, 1]^d
    - Outputs are standardised (handled automatically)

    Example:
        >>> from alf_tools.models.botorch_exact_gp_model import BoTorchGPModel
        >>> from alf_tools.datasets.botorch_synthetic_dataset import BoTorchSyntheticDataset
        >>>
        >>> # Create dataset
        >>> dataset = BoTorchSyntheticDataset(function_name="Branin")
        >>> train_data = dataset.load_dataset()
        >>>
        >>> # Train model
        >>> model = BoTorchGPModel()
        >>> model.train(train_data, val_data=None)
        >>> metrics = model.get_training_summary_metrics()
        >>> print(f"Final loss: {metrics['final_loss']:.4f}")
        >>>
        >>> # Make predictions
        >>> candidates = dataset.candidate_pool.candidates[:10]
        >>> predictions = model.predict(candidates)
        >>> print(f"Mean predictions: {predictions.means}")

    """

    def __init__(
        self,
        model_config: GPModelConfig | None = None,
        train_config: BoTorchTrainConfig | None = None,
    ):
        """Initialize BoTorch GP model.

        Args:
            model_config: Kernel and prior configuration. Defaults to
                GPModelConfig(ard=False) which uses an RBF kernel with a LogNormal
                lengthscale prior and no ARD.
            train_config: Training hyperparameters. Defaults to
                BoTorchTrainConfig().

        Raises:
            ValueError: If train_config.optimizer is not 'scipy' or 'torch'.
        """
        super().__init__()
        # Default ard=False preserves prior BoTorchGPModel behaviour (was use_ard=False)
        self.model_config = model_config or GPModelConfig(ard=False)
        self.train_config = train_config or BoTorchTrainConfig()

        if self.train_config.optimizer not in ("scipy", "torch"):
            raise ValueError(
                f"optimizer must be 'scipy' or 'torch', got {self.train_config.optimizer!r}"
            )

        if self.model_config.kernel_type not in ("rbf", "matern"):
            raise ValueError(
                f"BoTorchGPModel only supports 'rbf' and 'matern' kernels, "
                f"got {self.model_config.kernel_type!r}."
            )

        if self.train_config.device is None:
            self.device = get_device()
        else:
            self.device = torch.device(self.train_config.device)

        self.model: SingleTaskGP | None = None
        self.train_X: torch.Tensor | None = None
        self.train_Y: torch.Tensor | None = None
        self._input_normaliser: InputNormaliser | None = None
        self._training_metrics: dict[str, list[float]] = {"loss": [], "iteration": []}

    def featurise(self, inputs: list[Candidate]) -> torch.Tensor:
        """Convert candidates to feature tensors.

        For BoTorch GPs working with continuous data, this simply extracts
        the tensor data from each candidate.

        Args:
            inputs: List of Candidate objects containing tensor data.

        Returns:
            Tensor of shape (n, d) containing all input features.

        Raises:
            ValueError: If candidates don't contain valid tensor data.
        """
        return candidates_to_tensor(inputs, device=self.device, dtype=self.train_config.dtype)

    def _validate_shape(self, tensor: torch.Tensor) -> None:
        """Validate that a tensor is 2-D.

        Args:
            tensor: Tensor to validate.

        Raises:
            ValueError: If tensor is not 2-D.
        """
        if tensor.ndim != 2:
            raise ValueError(f"Expected 2D input tensor, got shape {tensor.shape}")

    def _make_kernel(self, ard_num_dims, ls_prior, ls_constraint):
        if self.model_config.kernel_type == "matern":
            base_kernel = MaternKernel(
                nu=self.model_config.matern_nu,
                ard_num_dims=ard_num_dims,
                lengthscale_prior=ls_prior,
                lengthscale_constraint=ls_constraint,
            )
        else:
            base_kernel = RBFKernel(
                ard_num_dims=ard_num_dims,
                lengthscale_prior=ls_prior,
                lengthscale_constraint=ls_constraint,
            )
        return base_kernel

    def train(
        self,
        train_data: LabelledCandidates,
        val_data: LabelledCandidates | None = None,
    ) -> None:
        """Train the GP model on training data.

        Args:
            train_data: Training dataset with candidates and labels.
            val_data: Optional validation dataset (not used for GP training,
                but included for API compatibility).

        Raises:
            ValueError: If training data is empty or has invalid format.
        """
        self._training_metrics = {"loss": [], "iteration": []}

        if len(train_data.candidates) == 0:
            raise ValueError("Training data cannot be empty")

        logger.info(f"Training BoTorch GP model on {len(train_data.candidates)} samples")

        # Convert candidates to tensors
        self.train_X = candidates_to_tensor(
            train_data.candidates, device=self.device, dtype=self.train_config.dtype
        )
        self.train_Y = torch.tensor(
            train_data.labels, dtype=self.train_config.dtype, device=self.device
        ).unsqueeze(-1)

        # Validate shapes
        self._validate_shape(self.train_X)
        self._validate_shape(self.train_Y)

        # Apply input normalisation if requested
        if self.train_config.normalise_inputs:
            train_x_np = self.train_X.cpu().numpy()
            self._input_normaliser = InputNormaliser()
            self._input_normaliser.fit(train_x_np)
            self.train_X = torch.tensor(
                self._input_normaliser.transform(train_x_np),
                dtype=self.train_config.dtype,
                device=self.device,
            )

        # Build kernel from model_config
        ls_prior = build_from_target(self.model_config.lengthscale_prior)
        ls_constraint = build_from_target(self.model_config.lengthscale_constraint)
        ard_num_dims = self.train_X.shape[-1] if self.model_config.ard else None

        if (
            self.model_config.ard
            and self.model_config.lengthscale_prior is not None
            and self.model_config.lengthscale_prior.get("_target_")
            == "gpytorch.priors.LogNormalPrior"
            and abs(self.model_config.lengthscale_prior.get("loc", 0) - math.sqrt(2)) < 1e-9
        ):
            logger.warning(
                "ARD is enabled with the default LogNormal prior (loc=sqrt(2)). "
                "Consider setting loc=sqrt(2) + log(d)*0.5 for dimension-aware Hvarfner "
                "priors, where d is the input dimensionality (%d).",
                self.train_X.shape[-1],
            )

        if self.model_config.ard and self.model_config.lengthscale_prior is None:
            logger.warning(
                "ARD is enabled with no lengthscale prior. "
                "Consider setting a dimension-aware prior such as "
                "LogNormal(loc=sqrt(2) + log(d)*0.5, scale=sqrt(3)) "
                "where d is the input dimensionality (%d).",
                self.train_X.shape[-1],
            )

        base_kernel = self._make_kernel(ard_num_dims, ls_prior, ls_constraint)
        covar_module = ScaleKernel(base_kernel)

        outcome_transform = Standardize(m=1) if self.train_config.standardise_outputs else None
        self.model = SingleTaskGP(
            train_X=self.train_X,
            train_Y=self.train_Y,
            covar_module=covar_module,
            outcome_transform=outcome_transform,
        )
        self.model = self.model.to(device=self.device, dtype=self.train_config.dtype)

        # Set up MLL and optimizer
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        # Fit model using BoTorch's fit_gpytorch_mll
        try:
            if self.train_config.optimizer == "scipy":
                # Use L-BFGS-B optimizer with scipy
                logging_optimizer = "scipy L-BFGS-B"

                fit_gpytorch_mll(
                    mll,
                    optimizer_kwargs={"options": {"maxiter": self.train_config.num_iterations}},
                    max_attempts=self.train_config.max_attempts,
                )
            else:  # torch
                # Use torch Adam optimizer
                optim = Adam
                logging_optimizer = "torch Adam"

                fit_gpytorch_mll(
                    mll,
                    optimizer=fit_gpytorch_mll_torch,
                    optimizer_kwargs={
                        "step_limit": self.train_config.num_iterations,
                        "optimizer": lambda params: optim(
                            params, lr=self.train_config.learning_rate
                        ),
                    },
                    max_attempts=self.train_config.max_attempts,
                )
            logger.info(
                f"Successfully trained BoTorch GP model using {logging_optimizer} "
                f"(step_limit={self.train_config.num_iterations}, "
                f"max_attempts={self.train_config.max_attempts})"
                + (
                    f", lr={self.train_config.learning_rate}"
                    if self.train_config.optimizer == "torch"
                    else ""
                )
            )

            # Record final loss
            self.model.eval()
            with torch.no_grad():
                output = self.model(self.train_X)
                loss_tensor = -mll(output, self.train_Y.squeeze(-1))  # type: ignore
                loss = float(loss_tensor.item())  # type: ignore

            if not math.isfinite(loss):
                logger.warning(
                    f"NaN/Inf loss detected after training ({loss}). "
                    "The model may be in a bad state."
                )

            self._training_metrics["loss"].append(loss)
            self._training_metrics["iteration"].append(self.train_config.num_iterations)

        except Exception as e:
            self.model = None
            self.train_X = None
            self.train_Y = None
            self._input_normaliser = None
            logger.error(f"Error training BoTorch GP model: {e}")
            raise

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Make predictions for new candidate points.

        Args:
            candidate_points: List of candidates to predict on.

        Returns:
            Predictions object with means and variances.

        Raises:
            RuntimeError: If model hasn't been trained yet.
            ValueError: If candidates are empty or have invalid format.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before making predictions")

        if len(candidate_points) == 0:
            raise ValueError("Candidates list cannot be empty")

        # Convert candidates to tensor
        test_X = candidates_to_tensor(
            candidate_points, device=self.device, dtype=self.train_config.dtype
        )

        # Validate shapes
        self._validate_shape(test_X)
        if self.train_X is not None and test_X.shape[1] != self.train_X.shape[1]:
            raise ValueError(
                f"Input dimension mismatch: expected {self.train_X.shape[1]}, got {test_X.shape[1]}"
            )

        # Apply input normalisation if fitted
        if self._input_normaliser is not None:
            test_x_np = test_X.cpu().numpy()
            test_X = torch.tensor(
                self._input_normaliser.transform(test_x_np),
                dtype=self.train_config.dtype,
                device=self.device,
            )

        # Make predictions
        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(test_X)
            mean = posterior.mean.squeeze(-1).cpu().numpy()
            variance = posterior.variance.squeeze(-1).cpu().numpy()

        return Predictions(
            means=mean,
            variances=variance,
        )

    def get_training_summary_metrics(self) -> dict[str, float | int | np.number]:
        """Get summary metrics from the most recent training run.

        Returns:
            Dictionary with final loss value.
        """
        if not self._training_metrics["loss"]:
            return {}

        return {
            "final_loss": float(self._training_metrics["loss"][-1]),
        }

    def sample(self, condition: object | None = None) -> list[Candidate]:
        """Sample from the GP model.

        Note: This is not typically used for BoTorch GPs, which are discriminative
        models. This method is included for API compatibility but raises
        NotImplementedError.

        Args:
            condition: Optional conditioning information for sampling (not used).

        Raises:
            NotImplementedError: GP models don't support generative sampling.
        """
        raise NotImplementedError(
            "BoTorchGPModel is a discriminative model and does not support "
            "sampling. For generating candidate pools, use a search function."
        )

    @property
    def botorch_model(self) -> SingleTaskGP:
        """Get the underlying BoTorch model.

        Returns:
            BoTorch SingleTaskGP model.

        Raises:
            RuntimeError: If model hasn't been trained yet.
        """
        if self.model is None:
            raise RuntimeError("Model must be trained before getting the underlying BoTorch model")
        return self.model
