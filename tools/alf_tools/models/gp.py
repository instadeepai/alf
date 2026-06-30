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

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import gpytorch
import numpy as np
import torch
from alf_core import (
    BaseDataset,
    BaseModel,
    BaseTrainConfig,
    Candidate,
    InputNormaliser,
    InputStandardiser,
    LabelledCandidates,
    OutputStandardiser,
    Predictions,
    ProblemType,
    Results,
    SurrogateEpochMetrics,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim.core import OptimizationResult
from botorch.optim.fit import fit_gpytorch_mll_scipy
from jaxtyping import Float

from alf_tools.models.utils import (
    build_from_target,
    create_char_to_idx_mapping,
    extract_sequences_from_inputs,
    get_device,
    one_hot_encode,
    transform_data,
)
from alf_tools.models.utils.botorch_utils import KernelTypes, _build_kernel, candidates_to_tensor
from alf_tools.utils.constants import PROTEIN_ALPHABET

logger = logging.getLogger("alf-tools")


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
        mean_type: Type of mean function ('constant' or 'zero').
        lengthscale_prior: Prior for the kernel lengthscale, as a `_target_` dict.
            Defaults to LogNormal(sqrt(2), sqrt(3)) (Hvarfner et al. 2024 fixed form).
            Set to `None` to use no prior.
        lengthscale_constraint: Constraint on the kernel lengthscale, as a
            `_target_` dict. Defaults to `None` (no constraint).
        outputscale_prior: Prior for the kernel output scale, as a `_target_` dict.
            Defaults to `None`.
        noise_prior: Prior for the likelihood noise, as a `_target_` dict.
            Defaults to LogNormal(-4.0, 1.0), matching `SingleTaskGP`'s default
            likelihood noise prior (Hvarfner et al. 2024). Set to `None` for a
            prior-free likelihood.
        noise_constraint: Constraint on the likelihood noise, as a `_target_` dict.
            Defaults to `None` (a GreaterThan(1e-4) fallback is applied internally).
        build_kernel_fn: Optional custom function to build the kernel. If provided,
            used instead of the default kernel construction logic. Not serializable —
            for advanced Python-only use. Should return a gpytorch.kernels.Kernel.
    """

    kernel_type: KernelTypes = "rbf"
    matern_nu: float = 2.5
    ard: bool = True
    mean_type: Literal["constant", "zero"] = "constant"
    lengthscale_prior: dict | None = field(
        default_factory=lambda: {
            "_target_": "gpytorch.priors.LogNormalPrior",
            "loc": math.sqrt(2),
            "scale": math.sqrt(3),
        }
    )
    lengthscale_constraint: dict | None = None
    outputscale_prior: dict | None = None
    noise_prior: dict | None = field(
        default_factory=lambda: {
            "_target_": "gpytorch.priors.LogNormalPrior",
            "loc": -4.0,
            "scale": 1.0,
        }
    )
    noise_constraint: dict | None = None
    build_kernel_fn: Callable[..., gpytorch.kernels.Kernel] | None = None

    def __post_init__(self) -> None:
        """Validate that prior/constraint fields are dicts or None.

        Raises:
            TypeError: If a prior or constraint field receives a non-dict value.
                Pass a `_target_` dict instead (see `build_from_target`).
        """
        for attr in (
            "lengthscale_prior",
            "lengthscale_constraint",
            "outputscale_prior",
            "noise_prior",
            "noise_constraint",
        ):
            val = getattr(self, attr)
            if val is not None and not isinstance(val, dict):
                raise TypeError(
                    f"GPModelConfig.{attr} must be a '_target_' dict or None "
                    f"(got {type(val).__name__}). "
                    f"See build_from_target() for the expected format."
                )


@dataclass
class GPTrainConfig(BaseTrainConfig):
    """Configuration for Gaussian Process training.

    Overrides `normalise_inputs_strategy` to `minmax` because GP kernels
    measure distances between inputs; scaling continuous features to [0, 1]
    improves marginal log-likelihood optimisation.

    Args:
        normalise_inputs_strategy: Which input normalisation to apply before
            training. Defaults to `minmax`; GP kernels measure distances so
            scaling continuous features to [0, 1] improves MLL.
        standardise_outputs: Whether to apply Z-score standardisation to
            output labels before training. Defaults to True; standardisation
            can improve training. For constant labels, std is clamped to a
            minimum value to avoid division by zero, but standardisation may
            not be meaningful.
        num_iterations: Number of optimisation iterations.
        optimizer_type: Type of optimizer to use ('adam', 'lbfgs' or 'scipy').
            'scipy' fits via BoTorch's `fit_gpytorch_mll` with scipy L-BFGS-B.
        max_attempts: Maximum fitting attempts on numerical failure. Only used
            when `optimizer_type='scipy'`.
        dtype: Working dtype for features, labels, and the GP, as a string
            (e.g. 'float32', 'float64'). Defaults to 'float64'.
        early_stopping_patience: Number of iterations without improvement
            before stopping. If None, no early stopping is used. Only used
            for 'adam' and 'lbfgs'.
        early_stopping_delta: Minimum change in loss to qualify as an
            improvement. Only used for 'adam' and 'lbfgs'.
        learning_rate: Inherited from BaseTrainConfig. Default overridden to 0.01.
        log_frequency: Inherited from BaseTrainConfig. Default: 10.
        label_dtype: Inherited from BaseTrainConfig. `None` (the default)
            falls back to `dtype`. If set, it must match `dtype` —
            `SingleTaskGP` requires features and labels to share one dtype,
            so a mismatch raises ValueError at model construction.
    """

    normalise_inputs_strategy: Literal["minmax", "zscore"] | None = (
        "minmax"  # override BaseTrainConfig default
    )
    standardise_outputs: bool = True  # override BaseTrainConfig default
    learning_rate: float = 0.01  # override BaseTrainConfig default
    num_iterations: int = 100
    optimizer_type: Literal["adam", "lbfgs", "scipy"] = "adam"
    max_attempts: int = 5
    dtype: str = "float64"
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


class GPModel(BaseModel):
    """Gaussian Process model for sequence fitness prediction.

    Uses BoTorch's `SingleTaskGP` backbone for efficient GP inference with
    flexible featurisation, kernel selection, and uncertainty quantification.

    Inputs are featurised via the configurable featuriser (one-hot, custom, or
    precomputed) and optionally passed through an external `InputNormaliser` and
    `OutputStandardiser` fitted at train time. The likelihood and kernel priors
    are configured through `GPModelConfig`, while fitting is controlled by
    `GPTrainConfig`, which exposes three optimiser options ('adam', 'lbfgs', and
    the robust `fit_gpytorch_mll`-based 'scipy') and defaults to float64.

    `predict()` returns the noise-inclusive predictive mean and variance. The
    trained backbone is exposed via the `botorch_model` property for use with
    native BoTorch acquisition functions; its `posterior(X)` gives the noise-free
    latent posterior.
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

        Raises:
            ValueError: If train_config.dtype is not a valid floating-point
                torch dtype, train_config.optimizer_type is not supported,
                train_config.max_attempts is less than 1, or
                train_config.label_dtype is set and differs from
                train_config.dtype.
        """
        # Use defaults if configs not provided
        self.model_config = model_config or GPModelConfig()
        self.train_config = train_config or GPTrainConfig()
        self.featurizer_config = featurizer_config or FeaturizerConfig()
        self.problem_type: ProblemType = ProblemType.REGRESSION

        if self.train_config.optimizer_type not in ("adam", "lbfgs", "scipy"):
            raise ValueError(
                f"Unsupported optimizer_type: {self.train_config.optimizer_type!r}. "
                f"Expected one of 'adam', 'lbfgs' or 'scipy'."
            )
        if self.train_config.max_attempts < 1:
            raise ValueError(
                f"max_attempts must be at least 1, got {self.train_config.max_attempts}"
            )
        resolved_dtype = getattr(torch, self.train_config.dtype, None)
        if not isinstance(resolved_dtype, torch.dtype) or not resolved_dtype.is_floating_point:
            raise ValueError(
                f"dtype {self.train_config.dtype!r} is not a valid floating-point "
                f"torch dtype. Use 'float32', 'float64', etc."
            )
        self._dtype: torch.dtype = resolved_dtype
        if (
            self.train_config.label_dtype is not None
            and self.train_config.label_dtype != resolved_dtype
        ):
            raise ValueError(
                f"label_dtype ({self.train_config.label_dtype}) must match "
                f"GPTrainConfig.dtype ({resolved_dtype}) for GPModel: SingleTaskGP "
                f"requires features and labels to share one dtype. Leave label_dtype "
                f"as None to fall back to dtype."
            )

        self.alphabet = alphabet
        self.alphabet_size = len(alphabet)
        self.char_to_idx = create_char_to_idx_mapping(alphabet)

        # Device setup
        self.device = get_device(device)

        # Model and likelihood initialized on each fit
        self.gp_model: SingleTaskGP | None = None
        self.likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None
        self.feature_dim: int | None = None

        # Store training data for GP predictions
        self.train_x: Float[torch.Tensor, "n_samples n_features"] | None = None
        self.train_y: Float[torch.Tensor, "n_samples"] | None = None

        # Normalisers — fitted on each train() call, used at predict() time
        self._input_transform: InputNormaliser | InputStandardiser | None = None
        self._output_standardiser: OutputStandardiser | None = None

        # Track metrics
        self.training_metrics: dict[str, float | int | np.number] = {}
        self._epoch_metrics: list[SurrogateEpochMetrics] = []

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

    def featurise(
        self, inputs: LabelledCandidates | list[Candidate]
    ) -> Float[torch.Tensor, "batch_size n_features"]:
        """Convert inputs to feature tensors.

        Args:
            inputs: Either LabelledCandidates or list of Candidates to featurize.

        Returns:
            Feature tensor of shape (batch_size, num_features).

        Raises:
            ValueError: If input type is invalid or featurizer_type is unsupported.
        """
        if self.featurizer_config.featurizer_type == "precomputed":
            candidates = inputs.candidates if isinstance(inputs, LabelledCandidates) else inputs
            return candidates_to_tensor(candidates, device=self.device, dtype=self._dtype)

        # Extract sequences from inputs
        sequences = extract_sequences_from_inputs(inputs)

        # Dispatch to appropriate featurization method
        if self.featurizer_config.featurizer_type == "one_hot":
            return one_hot_encode(
                sequences,
                self.char_to_idx,
                self.alphabet_size,
                flatten=self.featurizer_config.flatten_one_hot,
            ).to(self._dtype)
        elif self.featurizer_config.featurizer_type == "custom":
            return self._apply_custom_featurizer(sequences).to(self._dtype)
        else:
            raise ValueError(
                f"Unsupported featurizer_type: {self.featurizer_config.featurizer_type}"
            )

    def _initialize_likelihood(self) -> gpytorch.likelihoods.GaussianLikelihood:
        """Initialize the Gaussian likelihood with noise prior and constraint.

        Returns:
            Configured Gaussian likelihood.
        """
        noise_constraint = build_from_target(
            self.model_config.noise_constraint, expected_base=gpytorch.constraints.Interval
        )
        if noise_constraint is None:
            noise_constraint = gpytorch.constraints.GreaterThan(1e-4)
        return gpytorch.likelihoods.GaussianLikelihood(
            noise_prior=build_from_target(
                self.model_config.noise_prior, expected_base=gpytorch.priors.Prior
            ),
            noise_constraint=noise_constraint,
        )

    def _initialize_gp_model(
        self,
        train_x: Float[torch.Tensor, "n_samples n_features"],
        train_y: Float[torch.Tensor, "n_samples"],
    ) -> SingleTaskGP:
        """Initialize the BoTorch `SingleTaskGP` with training data.

        Args:
            train_x: Training features of shape (n_samples, n_features),
                already normalised.
            train_y: Training targets of shape (n_samples,), already
                standardised.

        Returns:
            Initialized SingleTaskGP.

        Raises:
            RuntimeError: If likelihood has not been initialized before calling this method.
            ValueError: If mean_type is not one of the supported types.
        """
        if self.likelihood is None:
            raise RuntimeError(
                "likelihood is None — _initialize_likelihood() "
                "must be called before _initialize_gp_model()"
            )

        if self.model_config.mean_type == "constant":
            mean_module = gpytorch.means.ConstantMean()
        elif self.model_config.mean_type == "zero":
            mean_module = gpytorch.means.ZeroMean()
        else:
            raise ValueError(f"Unknown mean_type: {self.model_config.mean_type}")

        covar_module = _build_kernel(
            kernel_type=self.model_config.kernel_type,
            input_dim=train_x.shape[-1],
            ard=self.model_config.ard,
            matern_nu=self.model_config.matern_nu,
            lengthscale_prior=build_from_target(
                self.model_config.lengthscale_prior, expected_base=gpytorch.priors.Prior
            ),
            lengthscale_constraint=build_from_target(
                self.model_config.lengthscale_constraint,
                expected_base=gpytorch.constraints.Interval,
            ),
            outputscale_prior=build_from_target(
                self.model_config.outputscale_prior, expected_base=gpytorch.priors.Prior
            ),
            build_kernel_fn=self.model_config.build_kernel_fn,
        )

        gp_model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y.unsqueeze(-1),
            likelihood=self.likelihood,
            covar_module=covar_module,
            mean_module=mean_module,
            outcome_transform=None,
            input_transform=None,
        )

        return gp_model.to(device=self.device, dtype=self._dtype)

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
            RuntimeError: If GP model or likelihood is not initialized.
        """
        if self.gp_model is None or self.likelihood is None:
            uninit = [
                name
                for name, obj in [("gp_model", self.gp_model), ("likelihood", self.likelihood)]
                if obj is None
            ]
            raise RuntimeError(
                f"{' and '.join(uninit)} not initialized — call train() "
                "before _optimize_hyperparameters()"
            )

        # Set to training mode
        self.gp_model.train()
        self.likelihood.train()

        # Use marginal log likelihood as loss
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp_model)

        if self.train_config.optimizer_type == "scipy":
            return self._fit_scipy(mll, train_x, train_y)

        # Setup optimizer
        if self.train_config.optimizer_type == "adam":
            optimizer = torch.optim.Adam(
                self.gp_model.parameters(), lr=self.train_config.learning_rate
            )
        else:  # lbfgs — optimizer_type is validated in __init__, scipy dispatched above
            optimizer = torch.optim.LBFGS(
                self.gp_model.parameters(),
                lr=self.train_config.learning_rate,
                max_iter=20,
                line_search_fn="strong_wolfe",
            )

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

            # Log at configured frequency
            if i % self.train_config.log_frequency == 0:
                logger.info(
                    f"Iteration {i}/{self.train_config.num_iterations} — loss={loss_value:.4f}"
                )

            # Record per-iteration metrics
            self._epoch_metrics.append(
                SurrogateEpochMetrics(
                    epoch=i,
                    train_loss=loss_value,
                    additional_metrics={"mll": -loss_value},
                )
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

    def _fit_scipy(
        self,
        mll: gpytorch.mlls.ExactMarginalLogLikelihood,
        train_x: Float[torch.Tensor, "n_samples n_features"],
        train_y: Float[torch.Tensor, "n_samples"],
    ) -> dict[str, float]:
        """Fit the GP via BoTorch's `fit_gpytorch_mll` with scipy L-BFGS-B.

        Records one `SurrogateEpochMetrics` entry per optimiser step. On a
        retry (numerical failure within `max_attempts`) the step counter
        restarts, so previously recorded entries are cleared to keep epoch
        numbers strictly increasing.

        Args:
            mll: Marginal log likelihood objective to maximise.
            train_x: Training features of shape (n_samples, n_features).
            train_y: Training targets of shape (n_samples,).

        Returns:
            Dictionary of training metrics (`final_mll`, `final_loss`,
            `num_iterations`). `final_loss` and `final_mll` come from an
            exact evaluation of the fitted mll, not the callback history.
        """

        def _record_step(parameters: dict[str, torch.Tensor], result: OptimizationResult) -> None:
            if self._epoch_metrics and result.step <= self._epoch_metrics[-1].epoch:
                # A fitting retry restarted the step counter — drop the
                # previous attempt's entries.
                self._epoch_metrics.clear()
            self._epoch_metrics.append(
                SurrogateEpochMetrics(
                    epoch=result.step,
                    train_loss=result.fval,
                    additional_metrics={"mll": -result.fval},
                )
            )

        fit_gpytorch_mll(
            mll,
            optimizer=fit_gpytorch_mll_scipy,
            optimizer_kwargs={
                "options": {"maxiter": self.train_config.num_iterations},
                "callback": _record_step,
            },
            max_attempts=self.train_config.max_attempts,
        )

        # Always evaluate the fitted mll exactly — callback entries may be
        # stale (e.g. from a failed attempt) or absent if scipy converged
        # before the first callback fired.
        mll.train()
        with torch.no_grad():
            loss = -mll(mll.model(train_x), train_y)
            final_loss = loss.item()

        return {
            "final_mll": -final_loss,
            "final_loss": final_loss,
            "num_iterations": len(self._epoch_metrics),
        }

    def _prepare_train_data(
        self,
        train_data: LabelledCandidates,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare the training data by featurising, normalising, and standardising.

        Fits normalisers exclusively on training data; call transform separately
        for validation data to avoid leakage. Converts numpy labels to tensors.

        Args:
            train_data: Training data containing sequences and oracle values.

        Returns:
            A tuple of training features and targets as tensors on the current device.
        """
        label_dtype = self.train_config.label_dtype or self._dtype
        train_x, train_y, self._input_transform, self._output_standardiser = transform_data(
            self.featurise(train_data),
            train_data.labels,
            self.train_config.normalise_inputs_strategy,
            self.train_config.standardise_outputs,
            label_dtype,
            self.device,
            feature_dtype=self._dtype,
        )
        return train_x, train_y

    def setup(self, dataset: "BaseDataset") -> None:
        """Validate that the dataset uses REGRESSION problem type.

        Args:
            dataset: The dataset this model will be trained on.

        Raises:
            ValueError: If the dataset problem_type is not REGRESSION.
        """
        super().setup(dataset)
        if dataset.config.problem_type != ProblemType.REGRESSION:
            raise ValueError(
                f"GPModel only supports REGRESSION, got {dataset.config.problem_type!r}."
            )

    def _warn_ard_lengthscale_prior(self, input_dim: int) -> None:
        """Warn when ARD is enabled without a dimension-aware lengthscale prior.

        Skipped when `input_dim == 1`: the recommended Hvarfner loc of
        `sqrt(2) + log(d)*0.5` equals the default `sqrt(2)` for d=1, so the
        warning would be a false positive.

        Args:
            input_dim: Dimensionality of the training features.
        """
        if not self.model_config.ard or input_dim == 1:
            return

        prior = self.model_config.lengthscale_prior
        if prior is None:
            logger.warning(
                "ARD is enabled with no lengthscale prior. "
                "Consider setting a dimension-aware prior such as "
                "LogNormal(loc=sqrt(2) + log(d)*0.5, scale=sqrt(3)) "
                "where d is the input dimensionality (%d).",
                input_dim,
            )
        elif (
            prior.get("_target_") == "gpytorch.priors.LogNormalPrior"
            and abs(prior.get("loc", 0) - math.sqrt(2)) < 1e-9
        ):
            logger.warning(
                "ARD is enabled with the default LogNormal prior (loc=sqrt(2)). "
                "Consider setting loc=sqrt(2) + log(d)*0.5 for dimension-aware Hvarfner "
                "priors, where d is the input dimensionality (%d).",
                input_dim,
            )

    def train(
        self,
        train_data: LabelledCandidates,
        val_data: LabelledCandidates | None = None,
    ) -> None:
        """Train the GP model by optimizing hyperparameters.

        Args:
            train_data: Training data containing sequences and oracle values.
            val_data: Validation data (used for monitoring, not for training).

        Raises:
            RuntimeError: If GP fitting produces a NaN/Inf loss.

        Note:
            For exact GPs, all training data is used for predictions. Validation
            data is only used for logging validation metrics during training.

            This method is not thread-safe. Concurrent calls to train() and predict()
            on the same instance will produce undefined behaviour.
        """
        self._epoch_metrics = []
        logger.info(f"Training GP with {len(train_data)} samples")

        train_x, train_y = self._prepare_train_data(train_data)

        # Store training data for later predictions
        self.train_x = train_x
        self.train_y = train_y
        self.feature_dim = train_x.shape[-1]

        self._warn_ard_lengthscale_prior(train_x.shape[-1])

        try:
            # Re-initialise the likelihood on every fit so priors/constraints
            # are fresh per training run
            self.likelihood = self._initialize_likelihood().to(self.device)

            # Initialize or reinitialize GP model
            self.gp_model = self._initialize_gp_model(train_x, train_y)

            total_params = sum(p.numel() for p in self.gp_model.parameters())
            logger.info(f"GP initialized with {total_params:,} parameters")

            # Optimize hyperparameters
            train_metrics = self._optimize_hyperparameters(train_x, train_y)

            if not math.isfinite(train_metrics["final_loss"]):
                raise RuntimeError(
                    f"GP fitting produced NaN/Inf loss ({train_metrics['final_loss']}). "
                    "The model could not be trained on this data."
                )
        except Exception as e:
            self.gp_model = None
            self.likelihood = None
            self.train_x = None
            self.train_y = None
            self.feature_dim = None
            self._input_transform = None
            self._output_standardiser = None
            self.training_metrics = {}
            self._epoch_metrics = []
            logger.error("Error training GP model: %s", e)
            raise

        # Store training metrics
        self.training_metrics = train_metrics

        # Evaluate on training data — inverse-transform to original label scale for metrics
        self.gp_model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            train_posterior = self.gp_model.posterior(train_x, observation_noise=True)
            train_means = train_posterior.mean.squeeze(-1).cpu().numpy()
            train_vars = train_posterior.variance.squeeze(-1).cpu().numpy()

        if self._output_standardiser is not None:
            train_means, train_vars = self._output_standardiser.inverse_transform(
                train_means, train_vars
            )

        train_predictions_obj = Predictions(means=train_means, variances=train_vars)
        train_results = Results(
            predictions=train_predictions_obj,
            targets=train_data.labels,
            problem_type=self.problem_type,
        )
        self.training_metrics.update({
            f"final_train_{k}": v for k, v in train_results.metrics.items()
        })

        logger.info(f"Training complete with metrics: {self.training_metrics}")

        # Evaluate on validation data if provided
        if val_data is not None and len(val_data) > 0:
            val_predictions = self.predict(val_data.candidates)
            val_results = Results(
                predictions=val_predictions,
                targets=val_data.labels,
                problem_type=self.problem_type,
            )
            self.training_metrics.update({
                f"final_val_{k}": v for k, v in val_results.metrics.items()
            })
            logger.info(f"Validation complete - {val_results.metrics}")

    def predict(self, candidate_points: list[Candidate]) -> Predictions:
        """Make predictions with uncertainty quantification.

        Returns the noise-inclusive predictive variance (latent variance plus
        likelihood noise). The noise-free latent posterior is available via
        `botorch_model.posterior(X)`.

        Args:
            candidate_points: List of candidates to predict fitness for.

        Returns:
            Predictions containing predicted fitness means and variances.

        Raises:
            RuntimeError: If the model is not trained.
            ValueError: If the input is invalid, not 2-D after featurisation,
                or its feature dimension does not match the training data.
        """
        if self.gp_model is None or self.likelihood is None:
            raise RuntimeError("Model not trained. Call train() first.")

        # Set to evaluation mode
        self.gp_model.eval()
        self.likelihood.eval()

        # Featurize input
        test_x = self.featurise(candidate_points)
        if test_x.ndim != 2:
            raise ValueError(f"Expected 2D input tensor, got shape {test_x.shape}")
        if self.train_x is not None and test_x.shape[1] != self.train_x.shape[1]:
            raise ValueError(
                f"Input dimension mismatch: expected {self.train_x.shape[1]}, got {test_x.shape[1]}"
            )
        test_x_np = test_x.cpu().numpy()

        # Apply input normalization if fitted
        if self._input_transform is not None:
            test_x_np = self._input_transform.transform(test_x_np)
        test_x = torch.tensor(test_x_np, dtype=self._dtype).to(self.device)

        # Make predictions with fast predictive variance computation
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.gp_model.posterior(test_x, observation_noise=True)
            means = posterior.mean.squeeze(-1).cpu().numpy()
            variances = posterior.variance.squeeze(-1).cpu().numpy()

        if self._output_standardiser is not None:
            means, variances = self._output_standardiser.inverse_transform(means, variances)

        return Predictions(means=means, variances=variances)

    @property
    def botorch_model(self) -> SingleTaskGP:
        """Get the underlying BoTorch model.

        Returns:
            Trained BoTorch SingleTaskGP model.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self.gp_model is None:
            raise RuntimeError("Model must be trained before getting the underlying BoTorch model")
        return self.gp_model

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
    ) -> dict[str, float | int | np.number]:
        """Return training metrics including learned hyperparameters.

        Returns:
            Dictionary of training metrics including:
            - final_mll: Final marginal log likelihood
            - final_loss: Final optimisation loss
            - num_iterations: Number of optimisation iterations completed
            - final_train_*: Metrics on the training set (e.g. final_train_mse,
              final_train_spearman)
            - final_val_*: Metrics on the validation set, present only if
              validation data was provided

            Learned hyperparameters (noise, lengthscale, outputscale) are
            available separately via :meth:`get_hyperparameters`.
        """
        return self.training_metrics

    def get_epoch_metrics(self) -> list[SurrogateEpochMetrics]:
        """Return per-epoch training metrics recorded during hyperparameter optimisation.

        Returns:
            list[SurrogateEpochMetrics]: List of metrics for each epoch, including training
            loss and any additional metrics.The length of the list corresponds to the number
            of training iterations completed.
        """
        return self._epoch_metrics

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

        # Extract lengthscale from base kernel (only present for ScaleKernel wrappers)
        base_kernel = getattr(self.gp_model.covar_module, "base_kernel", None)
        if base_kernel is not None and getattr(base_kernel, "lengthscale", None) is not None:
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
