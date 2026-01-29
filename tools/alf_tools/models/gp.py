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

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Literal, TypeAlias, Union

import gpytorch
import numpy as np
import torch
from alf_core import BaseModel, Candidate, LabelledCandidates, Predictions, Results
from alf_tools.models.model_utils import (
    create_char_to_idx_mapping,
    extract_sequences_from_inputs,
    get_device,
    one_hot_encode,
)
from alf_tools.utils.constants import PROTEIN_ALPHABET
from jaxtyping import Float

logger = logging.getLogger("alf-tools")

KernelTypes: TypeAlias = Literal["rbf", "matern", "linear", "polynomial", "rbf_linear"]


@dataclass
class GPModelConfig:
    """Configuration for Gaussian Process model.

    Args:
        kernel_type: Type of kernel to use. Supported: 'rbf', 'matern', 'linear',
            'polynomial', 'rbf_linear' (RBF + Linear).
        matern_nu: Smoothness parameter for Matern kernel (0.5, 1.5, or 2.5).
            Only used when kernel_type='matern'.
        ard: Whether to use Automatic Relevance Determination (separate lengthscale
            per dimension).
        lengthscale_prior: Prior distribution for kernel lengthscale. If None, uses
            GPyTorch defaults.
        outputscale_prior: Prior distribution for kernel output scale. If None, uses
            GPyTorch defaults.
        noise_constraint: Constraint on the likelihood noise. If None, uses reasonable
            defaults (e.g., GreaterThan(1e-4)).
        mean_type: Type of mean function ('constant' or 'zero').
    """

    kernel_type: KernelTypes = "rbf"
    matern_nu: float = 2.5
    ard: bool = True
    lengthscale_prior: gpytorch.priors.Prior | None = None
    outputscale_prior: gpytorch.priors.Prior | None = None
    noise_constraint: gpytorch.constraints.Interval | None = None
    mean_type: Literal["constant", "zero"] = "constant"


@dataclass
class GPTrainConfig:
    """Configuration for Gaussian Process training.

    Args:
        learning_rate: Learning rate for the optimizer.
        num_iterations: Number of optimization iterations.
        optimizer_type: Type of optimizer to use ('adam' or 'lbfgs').
        log_frequency: Frequency of logging training metrics (in iterations).
        early_stopping_patience: Number of iterations without improvement
            before stopping. If None, no early stopping is used.
        early_stopping_delta: Minimum change in loss to qualify as an
            improvement. If None, no early stopping is used.
    """

    learning_rate: float = 0.01
    num_iterations: int = 100
    optimizer_type: Literal["adam", "lbfgs"] = "adam"
    log_frequency: int = 10
    early_stopping_patience: int | None = None
    early_stopping_delta: float = 1e-4


# TODO: Might not need this.
@dataclass
class FeaturizerConfig:
    """Configuration for sequence featurization.

    Args:
        featurizer_type: Type of featurization to use.
            - 'one_hot': One-hot encoding of amino acids
            - 'custom': User-provided featurization function
            - 'precomputed': Features are pre-computed and provided directly
        custom_featurizer: Custom featurization function. Only used when
            featurizer_type='custom'. Should take list of sequences and return
            torch.Tensor of shape (batch_size, num_features).
        flatten_one_hot: Whether to flatten one-hot encoded sequences from
            (batch_size, alphabet_size, seq_length) to (batch_size, alphabet_size * seq_length).
            Only used for one_hot encoding.
    """

    featurizer_type: Literal["one_hot", "custom", "precomputed"] = "precomputed"
    custom_featurizer: (
        Callable[[list[str]], Float[torch.Tensor, "batch_size n_features"]] | None
    ) = None
    flatten_one_hot: bool = True


class ExactGPModel(gpytorch.models.ExactGP):
    """GPyTorch ExactGP model for Gaussian Process regression.

    This wraps GPyTorch's ExactGP to provide a flexible GP implementation with
    configurable kernels and mean functions.
    """

    def __init__(
        self,
        train_x: Float[torch.Tensor, "n_samples n_features"],
        train_y: Float[torch.Tensor, "n_samples"],
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        kernel_type: KernelTypes = "rbf",
        matern_nu: float = 2.5,
        ard: bool = True,
        mean_type: str = "constant",
        lengthscale_prior: gpytorch.priors.Prior | None = None,
        outputscale_prior: gpytorch.priors.Prior | None = None,
    ):
        """Initialize the ExactGP model.

        Args:
            train_x: Training features of shape (n_samples, n_features).
            train_y: Training targets of shape (n_samples,).
            likelihood: GPyTorch likelihood.
            kernel_type: Type of kernel to use.
            matern_nu: Smoothness parameter for Matern kernel.
            ard: Whether to use Automatic Relevance Determination.
            mean_type: Type of mean function.
            lengthscale_prior: Prior for kernel lengthscale.
            outputscale_prior: Prior for kernel output scale.

        Raises:
            ValueError: If mean_type is not one of the supported types.
        """
        super().__init__(train_x, train_y, likelihood)

        # Initialize mean module
        if mean_type == "constant":
            self.mean_module = gpytorch.means.ConstantMean()
        elif mean_type == "zero":
            self.mean_module = gpytorch.means.ZeroMean()
        else:
            raise ValueError(f"Unknown mean_type: {mean_type}")

        # Initialize covariance module (kernel)
        self.covar_module = self._build_kernel(
            kernel_type=kernel_type,
            input_dim=train_x.shape[-1],
            ard=ard,
            matern_nu=matern_nu,
            lengthscale_prior=lengthscale_prior,
            outputscale_prior=outputscale_prior,
        )

    def _build_kernel(
        self,
        kernel_type: str,
        input_dim: int,
        ard: bool,
        matern_nu: float,
        lengthscale_prior: gpytorch.priors.Prior | None,
        outputscale_prior: gpytorch.priors.Prior | None,
    ) -> gpytorch.kernels.Kernel:
        """Build the kernel based on configuration.

        Args:
            kernel_type: Type of kernel to build.
            input_dim: Dimensionality of input features.
            ard: Whether to use ARD.
            matern_nu: Smoothness for Matern kernel.
            lengthscale_prior: Prior for lengthscale.
            outputscale_prior: Prior for output scale.

        Returns:
            Configured GPyTorch kernel.

        Raises:
            ValueError: If kernel_type is not supported.
        """
        # Determine ARD dimensions
        ard_num_dims = input_dim if ard else None

        # Build base kernel
        if kernel_type == "rbf":
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
        elif kernel_type == "matern":
            base_kernel = gpytorch.kernels.MaternKernel(nu=matern_nu, ard_num_dims=ard_num_dims)
        elif kernel_type == "linear":
            base_kernel = gpytorch.kernels.LinearKernel(ard_num_dims=ard_num_dims)
        elif kernel_type == "polynomial":
            base_kernel = gpytorch.kernels.PolynomialKernel(power=2, ard_num_dims=ard_num_dims)
        elif kernel_type == "rbf_linear":
            # Composite kernel: RBF + Linear
            rbf_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
            linear_kernel = gpytorch.kernels.LinearKernel(ard_num_dims=ard_num_dims)
            base_kernel = rbf_kernel + linear_kernel
        else:
            raise ValueError(
                f"Unsupported kernel_type: {kernel_type}. "
                f"Supported types: 'rbf', 'matern', 'linear', 'polynomial', 'rbf_linear'"
            )

        # Set priors if provided
        if lengthscale_prior is not None and hasattr(base_kernel, "lengthscale"):
            base_kernel.register_prior(
                "lengthscale_prior",
                lengthscale_prior,
                lambda m: m.lengthscale,
                lambda m, v: m._set_lengthscale(v),
            )

        # Wrap with scale kernel to learn output scale
        kernel = gpytorch.kernels.ScaleKernel(base_kernel)

        if outputscale_prior is not None:
            kernel.register_prior(
                "outputscale_prior",
                outputscale_prior,
                lambda m: m.outputscale,
                lambda m, v: m._set_outputscale(v),
            )

        return kernel

    def forward(
        self, x: Float[torch.Tensor, "n_samples n_features"]
    ) -> gpytorch.distributions.MultivariateNormal:
        """Forward pass through the GP.

        Args:
            x: Input features of shape (n_samples, n_features).

        Returns:
            Multivariate normal distribution representing the GP posterior.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPModelTrainer(BaseModel):
    """Gaussian Process model for sequence fitness prediction.

    Uses GPyTorch for efficient GP inference with flexible featurization,
    kernel selection, and uncertainty quantification.
    """

    def __init__(
        self,
        name: str = "gp_model",
        model_config: GPModelConfig | None = None,
        train_config: GPTrainConfig | None = None,
        featurizer_config: FeaturizerConfig | None = None,
        alphabet: str = PROTEIN_ALPHABET,
        device: str | None = None,
    ):
        """Initialize the GPModel.

        Args:
            name: Name of the model.
            model_config: Configuration for GP model architecture.
            train_config: Configuration for GP training.
            featurizer_config: Configuration for sequence featurization.
            alphabet: Sequence alphabet to use for one-hot encoding.
            device: Device to use ('cuda', 'cpu', or None for auto-detect).
        """
        # Use defaults if configs not provided
        self.model_config = model_config or GPModelConfig()
        self.train_config = train_config or GPTrainConfig()
        self.featurizer_config = featurizer_config or FeaturizerConfig()

        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.char_to_idx = create_char_to_idx_mapping(alphabet)

        # Device setup
        self.device = get_device(device)

        # Model and likelihood initialized on first fit
        self.gp_model: ExactGPModel | None = None
        self.likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None
        self.feature_dim: int | None = None

        # Store training data for GP predictions
        self.train_x: Float[torch.Tensor, "n_samples n_features"] | None = None
        self.train_y: Float[torch.Tensor, "n_samples"] | None = None

        # Track metrics
        self.training_metrics: dict[str, Union[float, int, np.number]] = {}

    def _apply_custom_featurizer(self, sequences: list[str]) -> torch.Tensor:
        """Apply custom featurization function.

        Args:
            sequences: List of sequences as strings.

        Returns:
            Feature tensor of shape (batch_size, num_features).

        Raises:
            ValueError: If custom_featurizer is not set in config.
        """
        if self.featurizer_config.custom_featurizer is None:
            raise ValueError(
                "custom_featurizer must be set in FeaturizerConfig when using "
                "featurizer_type='custom'"
            )
        return self.featurizer_config.custom_featurizer(sequences)

    def featurise(self, inputs: Union[LabelledCandidates, list[Candidate]]) -> torch.Tensor:
        """Convert inputs to feature tensors.

        Args:
            inputs: Either LabelledCandidates or list of Candidates to featurize.

        Returns:
            Feature tensor of shape (batch_size, num_features).

        Raises:
            ValueError: If input type is invalid or featurizer_type is unsupported.
        """
        # Extract sequences from inputs
        sequences = extract_sequences_from_inputs(inputs)

        # Dispatch to appropriate featurization method
        if self.featurizer_config.featurizer_type == "one_hot":
            return one_hot_encode(
                sequences,
                self.char_to_idx,
                self.alphabet_size,
                flatten=self.featurizer_config.flatten_one_hot,
            )
        elif self.featurizer_config.featurizer_type == "custom":
            return self._apply_custom_featurizer(sequences)
        elif self.featurizer_config.featurizer_type == "precomputed":
            # For precomputed, sequences should already be tensors or arrays
            if isinstance(sequences[0], (torch.Tensor, np.ndarray)):
                if isinstance(sequences[0], np.ndarray):
                    return torch.tensor(np.stack(sequences), dtype=torch.float32)
                return torch.stack(sequences)
            else:
                raise ValueError(
                    "For featurizer_type='precomputed', Candidate.data must be "
                    "torch.Tensor or np.ndarray"
                )
        else:
            raise ValueError(
                f"Unsupported featurizer_type: {self.featurizer_config.featurizer_type}"
            )

    def _initialize_likelihood(self) -> gpytorch.likelihoods.GaussianLikelihood:
        """Initialize the Gaussian likelihood with noise constraints.

        Returns:
            Configured Gaussian likelihood.
        """
        # Use configured noise constraint or default
        if self.model_config.noise_constraint is not None:
            noise_constraint = self.model_config.noise_constraint
        else:
            # Default: constrain noise to be >= 1e-4
            noise_constraint = gpytorch.constraints.GreaterThan(1e-4)

        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_constraint)
        return likelihood

    def _initialize_gp_model(
        self,
        train_x: Float[torch.Tensor, "n_samples n_features"],
        train_y: Float[torch.Tensor, "n_samples"],
    ) -> ExactGPModel:
        """Initialize the GP model with training data.

        Args:
            train_x: Training features of shape (n_samples, n_features).
            train_y: Training targets of shape (n_samples,).

        Returns:
            Initialized ExactGPModel.

        Raises:
            ValueError: If likelihood has not been initialized before calling this method.
        """
        if self.likelihood is None:
            raise ValueError("Likelihood must be initialized before GP model")

        gp_model = ExactGPModel(
            train_x=train_x,
            train_y=train_y,
            likelihood=self.likelihood,
            kernel_type=self.model_config.kernel_type,
            matern_nu=self.model_config.matern_nu,
            ard=self.model_config.ard,
            mean_type=self.model_config.mean_type,
            lengthscale_prior=self.model_config.lengthscale_prior,
            outputscale_prior=self.model_config.outputscale_prior,
        )

        return gp_model.to(self.device)

    def _optimize_hyperparameters(
        self,
        train_x: Float[torch.Tensor, "n_samples n_features"],
        train_y: Float[torch.Tensor, "n_samples"],
    ) -> dict[str, float]:
        """Optimize GP hyperparameters using marginal log likelihood.

        Args:
            train_x: Training features of shape (n_samples, n_features).
            train_y: Training targets of shape (n_samples,).

        Returns:
            Dictionary of training metrics (e.g., final loss, learned hyperparameters).

        Raises:
            ValueError: If GP model or likelihood is not initialized.
        """
        if self.gp_model is None or self.likelihood is None:
            raise ValueError("GP model and likelihood must be initialized before optimization")

        # Set to training mode
        self.gp_model.train()
        self.likelihood.train()

        # Use marginal log likelihood as loss
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

        # Setup optimizer
        if self.train_config.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.gp_model.parameters(), lr=self.train_config.learning_rate
            )
        elif self.train_config.optimizer_type == "lbfgs":
            optimizer = torch.optim.LBFGS(
                self.gp_model.parameters(),
                lr=self.train_config.learning_rate,
                max_iter=20,
                line_search_fn="strong_wolfe",
            )
        else:
            raise ValueError(f"Unsupported optimizer_type: {self.train_config.optimizer_type}")

        # Training loop
        losses = []
        best_loss = float("inf")
        patience_counter = 0

        for i in range(self.train_config.num_iterations):
            if self.train_config.optimizer_type == "adam":
                optimizer.zero_grad()
                output = self.gp_model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()
                loss_value = loss.item()
            else:  # LBFGS

                def closure():
                    optimizer.zero_grad()
                    output = self.gp_model(train_x)
                    loss = -mll(output, train_y)
                    loss.backward()
                    return loss

                loss = optimizer.step(closure)
                loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss

            losses.append(loss_value)

            # Logging
            if (i + 1) % self.train_config.log_frequency == 0:
                logger.info(
                    f"Iteration {i + 1}/{self.train_config.num_iterations} - Loss: {loss_value:.4f}"
                )

            # Early stopping
            if self.train_config.early_stopping_patience is not None:
                if loss_value < best_loss - self.train_config.early_stopping_delta:
                    best_loss = loss_value
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.train_config.early_stopping_patience:
                    logger.info(f"Early stopping at iteration {i + 1}")
                    break

        # Return metrics
        metrics = {
            "final_mll": -losses[-1],  # Convert back to positive MLL
            "final_loss": losses[-1],
            "num_iterations": len(losses),
        }

        return metrics

    def train(
        self,
        train_data: LabelledCandidates,
        val_data: LabelledCandidates | None = None,
    ) -> None:
        """Train the GP model by optimizing hyperparameters.

        Args:
            train_data: Training data containing sequences and oracle values.
            val_data: Optional validation data (used for monitoring, not for training).

        Note:
            For exact GPs, all training data is used for predictions. Validation
            data is only used for logging validation metrics during training.
        """
        logger.info(f"Training GP with {len(train_data)} samples")

        # Featurize training data
        train_x = self.featurise(train_data).to(self.device)
        train_y = torch.tensor(train_data.labels, dtype=torch.float32).to(self.device)

        # Store training data for later predictions
        self.train_x = train_x
        self.train_y = train_y

        # Initialize likelihood if first time
        if self.likelihood is None:
            self.likelihood = self._initialize_likelihood().to(self.device)

        # Initialize or reinitialize GP model
        self.feature_dim = train_x.shape[-1]
        self.gp_model = self._initialize_gp_model(train_x, train_y)

        total_params = sum(p.numel() for p in self.gp_model.parameters())
        logger.info(f"GP initialized with {total_params:,} parameters")

        # Optimize hyperparameters
        train_metrics = self._optimize_hyperparameters(train_x, train_y)

        # Store training metrics
        self.training_metrics = train_metrics

        # Evaluate on training data
        self.gp_model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            train_preds = self.likelihood(self.gp_model(train_x))
            train_means = train_preds.mean.cpu().numpy()
            train_vars = train_preds.variance.cpu().numpy()

        train_predictions_obj = Predictions(means=train_means, variances=train_vars)
        train_results = Results(predictions=train_predictions_obj, targets=train_data.labels)
        self.training_metrics.update({
            f"final_train_{k}": v for k, v in train_results.metrics.items()
        })

        logger.info(f"Training completewith metrics: {self.training_metrics}")

        # Evaluate on validation data if provided
        if val_data is not None and len(val_data) > 0:
            val_predictions = self.predict(val_data.candidates)
            val_results = Results(predictions=val_predictions, targets=val_data.labels)
            self.training_metrics.update({
                f"final_val_{k}": v for k, v in val_results.metrics.items()
            })
            logger.info(f"Validation complete - {val_results.metrics}")

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Make predictions with uncertainty quantification.

        Args:
            candidate_points: List of candidates to predict fitness for.

        Returns:
            Predictions containing predicted fitness means and variances.

        Raises:
            RuntimeError: If the model is not trained.
            ValueError: If the input is invalid.
        """
        if self.gp_model is None or self.likelihood is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Set to evaluation mode
        self.gp_model.eval()
        self.likelihood.eval()

        # Featurize input
        test_x = self.featurise(candidate_points).to(self.device)

        # Make predictions with fast predictive variance computation
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.gp_model(test_x))
            means = predictions.mean.cpu().numpy()
            variances = predictions.variance.cpu().numpy()

        return Predictions(means=means, variances=variances)

    def sample(self, *args: Any, **kwargs: Any) -> list[Candidate]:
        """Sample candidate points from the GP posterior.

        This could be used to generate new sequences by:
        1. Sampling from the GP in feature space
        2. Decoding features back to sequences (requires inverse featurization)

        Note:
            This is challenging for discrete sequence spaces and may not be
            practically useful. Consider raising NotImplementedError.
        """
        raise NotImplementedError(
            "Sampling from GP posterior in sequence space is not implemented. "
            "GPs are typically used for prediction, not generation."
        )

    def get_training_summary_metrics(
        self,
    ) -> dict[str, Union[float, int, np.number]]:
        """Return training metrics including learned hyperparameters.

        Returns:
            Dictionary of training metrics including:
            - final_mll: Final marginal log likelihood
            - noise: Learned noise parameter
            - lengthscale: Learned lengthscale(s)
            - outputscale: Learned output scale
            - training_iterations: Number of iterations completed
        """
        return self.training_metrics

    def get_hyperparameters(self) -> dict[str, Any]:
        """Get current GP hyperparameters.

        Returns:
            Dictionary containing learned hyperparameters:
            - noise: Likelihood noise
            - lengthscale: Kernel lengthscale(s)
            - outputscale: Kernel output scale
            - mean_constant: Mean function constant (if applicable)

        Raises:
            RuntimeError: If model is not trained.
        """
        if self.gp_model is None or self.likelihood is None:
            raise RuntimeError("Model not trained. Call train() first.")

        hyperparams = {}

        # Extract noise
        hyperparams["noise"] = self.likelihood.noise.item()

        # Extract kernel hyperparameters
        # Handle ScaleKernel wrapper
        if hasattr(self.gp_model.covar_module, "outputscale"):
            hyperparams["outputscale"] = self.gp_model.covar_module.outputscale.item()

        # Extract lengthscale from base kernel
        base_kernel = self.gp_model.covar_module.base_kernel
        if hasattr(base_kernel, "lengthscale"):
            lengthscale = base_kernel.lengthscale.detach().cpu().numpy()
            # If ARD, return array; otherwise return scalar
            if lengthscale.size == 1:
                hyperparams["lengthscale"] = lengthscale.item()
            else:
                hyperparams["lengthscale"] = lengthscale.squeeze()

        # Extract mean constant if constant mean
        if isinstance(self.gp_model.mean_module, gpytorch.means.ConstantMean):
            hyperparams["mean_constant"] = self.gp_model.mean_module.constant.item()

        return hyperparams
