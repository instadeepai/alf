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
- Built-in input/output transforms (normalization, standardization)
- Simplified model fitting with fit_gpytorch_mll
- Seamless integration with BoTorch acquisition functions
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Union

import numpy as np
import torch
from alf_core import BaseModel, Candidate, LabelledCandidates, Predictions
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim.fit import fit_gpytorch_mll_torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import Adam

from alf_tools.models.utils.torch_utils import get_device
from alf_tools.utils.botorch_utils import candidates_to_tensor

logger = logging.getLogger("alf-tools")


class BoTorchGPModel(BaseModel):
    """Gaussian Process model using BoTorch's SingleTaskGP.

    This model wraps BoTorch's SingleTaskGP, which is built on GPyTorch but provides
    better defaults and easier integration with BoTorch acquisition functions.

    Key Features:
    - Modern hyperparameter priors from Hvarfner et al. 2024
    - Automatic output standardization (zero mean, unit variance)
    - Efficient model fitting with fit_gpytorch_mll
    - Compatible with all BoTorch acquisition functions

    The model expects continuous inputs (tensors) and works best when:
    - Inputs are normalized to [0, 1]^d
    - Outputs are standardized (handled automatically)

    Example:
        >>> from alf_tools.models.botorch_gp_models import BoTorchGPModel
        >>> from alf_tools.datasets.botorch_test_functions import BoTorchSyntheticDataset
        >>>
        >>> # Create dataset
        >>> dataset = BoTorchSyntheticDataset(function_name="Branin")
        >>> train_data = dataset.load_dataset()
        >>>
        >>> # Train model
        >>> model = BoTorchGPModel()
        >>> results = model.fit(train_data)
        >>> print(f"Final loss: {results.training_metrics['loss'][-1]:.4f}")
        >>>
        >>> # Make predictions
        >>> candidates = dataset.candidate_pool.candidates[:10]
        >>> predictions = model.predict(candidates)
        >>> print(f"Mean predictions: {predictions.means}")

    """

    def __init__(
        self,
        normalize_inputs: bool = True,
        standardize_outputs: bool = True,
        num_iterations: int = 100,
        learning_rate: float = 0.1,
        optimizer: str = "scipy",
        max_attempts: int = 5,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize BoTorch GP model.

        Args:
            normalize_inputs: Whether to normalize inputs to [0, 1]. Default: True.
                If your data is already normalized, set to False.
            standardize_outputs: Whether to standardize outputs (zero mean, unit variance).
                Default: True. BoTorch handles this automatically with Standardize transform.
            num_iterations: Number of optimization iterations for MLL. Default: 100.
                For scipy optimizer: controls 'maxiter' in L-BFGS-B.
                For torch optimizer: controls step_limit.
            learning_rate: Learning rate for Adam optimizer. Default: 0.1.
                Only used if optimizer='torch'. Ignored for scipy optimizer.
            optimizer: Optimization backend to use. Default: 'scipy'.
                - 'scipy': Uses L-BFGS-B (faster, better for small-medium datasets)
                - 'torch': Uses Adam (more flexible, better for large datasets)
            max_attempts: Maximum number of fitting attempts. Default: 5.
                If fitting fails (e.g., due to numerical issues), it will retry
                up to max_attempts times with different initializations.
            device: Device to run on ('cpu' or 'cuda'). If None, auto-detects.
            dtype: Data type for tensors. Default: torch.float32.

        Raises:
            ValueError: If optimizer is not 'scipy' or 'torch'.
            RuntimeError: If device cannot be determined or is unavailable.
            Exception: If model fitting fails after max_attempts.
        """
        super().__init__()
        self.normalize_inputs = normalize_inputs
        self.standardize_outputs = standardize_outputs
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.max_attempts = max_attempts
        self.dtype = dtype

        if optimizer not in ["scipy", "torch"]:
            raise ValueError(f"optimizer must be 'scipy' or 'torch', got {optimizer}")

        # Device setup
        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)

        # Model will be initialized during fit
        self.model: Optional[SingleTaskGP] = None
        self.train_X: Optional[torch.Tensor] = None
        self.train_Y: Optional[torch.Tensor] = None

        # Training metrics
        self._training_metrics: dict[str, list[float]] = {
            "loss": [],
            "iteration": [],
        }

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
        return candidates_to_tensor(inputs, device=self.device)

    def train(
        self,
        train_data: LabelledCandidates,
        val_data: Optional[LabelledCandidates] = None,
    ) -> None:
        """Train the GP model on training data.

        Args:
            train_data: Training dataset with candidates and labels.
            val_data: Optional validation dataset (not used for GP training,
                but included for API compatibility).

        Raises:
            ValueError: If training data is empty or has invalid format.
        """
        if len(train_data.candidates) == 0:
            raise ValueError("Training data cannot be empty")

        logger.info(f"Training BoTorch GP model on {len(train_data.candidates)} samples")

        # Convert candidates to tensors
        self.train_X = candidates_to_tensor(train_data.candidates, device=self.device)
        self.train_Y = torch.tensor(
            train_data.labels, dtype=self.dtype, device=self.device
        ).unsqueeze(-1)

        # Validate shapes
        if self.train_X.ndim != 2:
            raise ValueError(f"Expected 2D input tensor, got shape {self.train_X.shape}")
        if self.train_Y.ndim != 2:
            raise ValueError(f"Expected 2D output tensor, got shape {self.train_Y.shape}")

        # Initialize SingleTaskGP
        # Note: SingleTaskGP automatically applies Standardize outcome transform
        # if standardize_outputs=True (which is the default)
        self.model = SingleTaskGP(
            train_X=self.train_X,
            train_Y=self.train_Y,
        )
        self.model = self.model.to(device=self.device, dtype=self.dtype)

        # Set up MLL and optimizer
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        # Fit model using BoTorch's fit_gpytorch_mll
        try:
            if self.optimizer == "scipy":
                # Use L-BFGS-B optimizer with scipy
                fit_gpytorch_mll(
                    mll,
                    optimizer_kwargs={"options": {"maxiter": self.num_iterations}},
                    max_attempts=self.max_attempts,
                )
                logger.info(
                    f"Successfully trained BoTorch GP model using scipy L-BFGS-B "
                    f"(maxiter={self.num_iterations}, max_attempts={self.max_attempts})"
                )
            else:  # torch
                # Use torch Adam optimizer
                fit_gpytorch_mll(
                    mll,
                    optimizer=fit_gpytorch_mll_torch,
                    optimizer_kwargs={
                        "step_limit": self.num_iterations,
                        "optimizer": lambda params: Adam(params, lr=self.learning_rate),
                    },
                    max_attempts=self.max_attempts,
                )
                logger.info(
                    f"Successfully trained BoTorch GP model using torch Adam "
                    f"(step_limit={self.num_iterations}, lr={self.learning_rate}, "
                    f"max_attempts={self.max_attempts})"
                )

            # Record final loss
            self.model.eval()
            with torch.no_grad():
                output = self.model(self.train_X)
                loss_tensor = -mll(output, self.train_Y.squeeze(-1))  # type: ignore
                loss = float(loss_tensor.item())  # type: ignore

            self._training_metrics["loss"].append(loss)
            self._training_metrics["iteration"].append(self.num_iterations)

        except Exception as e:
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
        test_X = candidates_to_tensor(candidate_points, device=self.device)

        # Validate shapes
        if test_X.ndim != 2:
            raise ValueError(f"Expected 2D input tensor, got shape {test_X.shape}")
        if self.train_X is not None and test_X.shape[1] != self.train_X.shape[1]:
            raise ValueError(
                f"Input dimension mismatch: expected {self.train_X.shape[1]}, got {test_X.shape[1]}"
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

    def get_training_summary_metrics(self) -> dict[str, Union[float, int, np.number]]:
        """Get summary metrics from the most recent training run.

        Returns:
            Dictionary with final loss value.
        """
        if not self._training_metrics["loss"]:
            return {}

        return {
            "final_loss": float(self._training_metrics["loss"][-1]),
        }

    def sample(self, condition: Any | None = None) -> list[Candidate]:
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
