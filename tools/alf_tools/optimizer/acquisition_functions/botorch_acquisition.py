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

"""Generic BoTorch acquisition function wrapper for ALF.

This module provides a unified interface to all BoTorch acquisition functions,
making it easy to switch between different acquisition strategies.
"""

import logging
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
import torch
from alf_core import AcquisitionFunction, Candidate, LabelledCandidates, State
from alf_tools.models.utils.botorch_utils import (
    candidates_to_tensor,
    get_bounds_tensor,
    tensor_to_candidates,
)
from alf_tools.optimizer.acquisition_functions.botorch_samplers import BoTorchMCSampler
from alf_tools.optimizer.acquisition_functions.utils.botorch_model_wrapper import (
    BotorchModelWrapper,
    resolve_botorch_model,
)
from botorch.acquisition.analytic import (
    LogExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qUpperConfidenceBound,
)
from botorch.generation import gen_candidates_scipy
from botorch.models.model import Model as BotorchModel
from botorch.optim import optimize_acqf
from jaxtyping import Float

logger = logging.getLogger("alf-tools")

# Type alias for supported acquisition function types
AcquisitionType = Literal[
    "qEI",
    "qLogEI",
    "qNEI",
    "qUCB",
    "expected_improvement",
    "upper_confidence_bound",
    "probability_of_improvement",
    "log_noisy_expected_improvement",
]


@dataclass
class BoTorchAcquisitionOptConfig:
    """Configuration for acquisition function optimization.

    These options are passed directly to BoTorch's ``optimize_acqf`` via the
    ``options`` dict, which is forwarded to the underlying scipy optimizer
    (``gen_candidates_scipy``).

    Args:
        batch_limit: Maximum number of candidate points processed in a single
            batch during optimization. Smaller values reduce memory usage at
            the cost of more iterations. Default: 64.
        maxiter: Maximum number of scipy L-BFGS-B iterations per restart.
            Default: 300.
        nonnegative: If ``True``, constrain candidate values to be
            non-negative during optimization. Default: ``False``.
        sample_around_best: If ``True``, draw initial raw samples concentrated
            around the current best observed point in addition to global
            random samples. Improves warm-starting of local optimization.
            Default: ``True``.
        sample_around_best_sigma: Standard deviation of the Gaussian noise
            added around the best point when ``sample_around_best`` is
            ``True``. Default: 0.1.
    """

    batch_limit: int = 64
    maxiter: int = 300
    nonnegative: bool = False
    sample_around_best: bool = True
    sample_around_best_sigma: float = 0.1


class BoTorchAcquisition(AcquisitionFunction):
    """Generic wrapper for BoTorch acquisition functions.

    This class provides a unified interface to switch between different BoTorch
    acquisition functions without changing code structure. It supports both:
    1. **Discrete mode**: Score a provided pool of candidates
    2. **Continuous mode**: Optimize acquisition function directly

    Supported acquisition functions:
    - **qEI** (qExpectedImprovement): Standard batch expected improvement
    - **qLogEI** (qLogExpectedImprovement): Log batch expected improvement
    - **qNEI** (qNoisyExpectedImprovement): For noisy observations
    - **qUCB** (qUpperConfidenceBound): Upper confidence bound with exploration bonus

    Example - Switching acquisition functions:
        >>> from alf_tools.optimizer.acquisition_functions import (
        ...     BoTorchAcquisition, BoTorchMCSampler
        ... )
        >>> from alf_tools.optimizer.search import ContinuousSearch
        >>>
        >>> # Create sampler configuration
        >>> sampler = BoTorchMCSampler(sampler_type="sobol", num_samples=512)
        >>>
        >>> # Try different acquisition functions
        >>> acq_qei = BoTorchAcquisition(
        ...     acquisition_type="qEI",
        ...     sampler=sampler,
        ...     bounds=[[0, 1], [0, 1]]
        ... )
        >>>
        >>> acq_qucb = BoTorchAcquisition(
        ...     acquisition_type="qUCB",
        ...     sampler=sampler,
        ...     bounds=[[0, 1], [0, 1]],
        ...     beta=0.2  # Exploration parameter for UCB
        ... )
        >>>
        >>> # Use in optimizer
        >>> from alf_core import Optimizer
        >>> optimizer = Optimizer(
        ...     acquisition_fn=acq_qei,  # or acq_qucb
        ...     search_fn=ContinuousSearch()
        ... )

    Args:
        acquisition_type: Type of acquisition function. Options:
            - "qEI": Expected Improvement (general purpose)
            - "qLogEI": Log Expected Improvement (See [Ament2023logei]_ for details.)
            - "qNEI": Noisy Expected Improvement (for noisy observations)
            - "qUCB": Upper Confidence Bound (tunable exploration)
        sampler: BoTorchMCSampler configuration or None for analytic acquisition.
            If None, uses default Sobol sampler with 512 samples.
        bounds: Bounds for continuous optimization. List of [lower, upper] for each
            dimension. Example: [[0, 1], [0, 1]] for 2D unit cube.
            Required for continuous optimization mode.
        num_restarts: Number of random restarts for optimization. Default: 10.
        raw_samples: Number of initial random samples for optimization. Default: 512.
        batch_size: Batch size for acquisition (q). Default: 1.
        sequential: If True, optimize candidates sequentially (faster but less diverse).
            If False, jointly optimize (slower but better diversity). Default: False.
        beta: Exploration parameter for qUCB. Higher = more exploration. Default: 0.2.
            Only used when acquisition_type="qUCB".
        optimization_config: Scipy optimizer options forwarded to ``optimize_acqf``.
            If None, uses ``BoTorchAcquisitionOptConfig`` defaults.
        kwargs: Additional keyword arguments passed to the specific acquisition function.

    Raises:
        ValueError: If acquisition_type is not supported.
        NotImplementedError: If acquisition_type is "qKG" (planned but not yet implemented).
    """

    _BATCH_JOINT_POSTERIOR_ERROR = (
        "Batch acquisition (q>1) needs a model with a joint posterior. Use a "
        "joint-posterior surrogate (one exposing a trained `botorch_model`), or "
        "select batches with ALF's native CoreSet/Thompson acquisitions."
    )

    def __init__(
        self,
        acquisition_type: AcquisitionType,
        sampler: BoTorchMCSampler | None = None,
        bounds: list[list[float]] | None = None,
        num_restarts: int = 10,
        raw_samples: int = 512,
        batch_size: int = 1,
        sequential: bool = False,
        beta: float = 0.2,
        optimization_config: BoTorchAcquisitionOptConfig | None = None,
        **kwargs,
    ):
        """Initialize generic BoTorch acquisition function.

        Raises:
            ValueError: If acquisition_type is not supported.
            NotImplementedError: If acquisition_type is "qKG".
        """
        super().__init__()

        # qKG is planned but not implemented; raise early so the error is clear.
        if acquisition_type == "qKG":
            raise NotImplementedError(
                "qKG (Knowledge Gradient) is not yet implemented. Use qEI, qNEI, or qUCB instead."
            )

        # Validate acquisition type
        valid_types = get_args(AcquisitionType)
        if acquisition_type not in valid_types:
            raise ValueError(
                f"Unsupported acquisition_type: {acquisition_type}. "
                f"Must be one of: {', '.join(valid_types)}"
            )

        self.acquisition_type = acquisition_type
        self.bounds = bounds
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.batch_size = batch_size
        self.sequential = sequential
        self.beta = beta
        self.kwargs = kwargs
        self.optimization_config = optimization_config or BoTorchAcquisitionOptConfig()

        # Set up sampler
        if sampler is None:
            # Default: Sobol QMC with 512 samples and fixed seed for reproducibility
            self.sampler_config = BoTorchMCSampler(sampler_type="sobol", num_samples=512, seed=0)
        else:
            self.sampler_config = sampler

        logger.info(
            f"Initialized BoTorch acquisition function: {acquisition_type} "
            f"with sampler: {self.sampler_config}"
        )

    def _require_joint_posterior_for_batch(self, wrapped_model: BotorchModel) -> None:
        """Raise if a batch (q>1) acquisition is requested for a marginal-only model.

        Args:
            wrapped_model: The (possibly wrapped) model that will score candidates.

        Raises:
            ValueError: If `batch_size > 1` and the model cannot provide a joint
                posterior.
        """
        if self.batch_size <= 1:
            return
        if (
            isinstance(wrapped_model, BotorchModelWrapper)
            and not wrapped_model.provides_joint_posterior
        ):
            raise ValueError(self._BATCH_JOINT_POSTERIOR_ERROR)

    def _create_acquisition_function(
        self, model, best_f: float, X_baseline: torch.Tensor | None = None
    ):
        """Create the specific BoTorch acquisition function.

        Args:
            model: Model with a posterior() method compatible with BoTorch.
                Can be a native BoTorch model or a wrapper around an ALF model.
            best_f: Best observed value so far.
            X_baseline: Baseline points for qKG (optional).

        Returns:
            BoTorch acquisition function instance.

        Raises:
            ValueError: If qNEI / log_noisy_expected_improvement requires X_baseline
                but none was provided, or if acquisition_type is unknown.
        """
        # Create sampler
        sampler = self.sampler_config.get_sampler()

        # Create acquisition function based on type
        if self.acquisition_type == "expected_improvement":
            return LogExpectedImprovement(model=model, best_f=best_f)
        elif self.acquisition_type == "probability_of_improvement":
            return ProbabilityOfImprovement(model=model, best_f=best_f)
        elif self.acquisition_type == "upper_confidence_bound":
            return UpperConfidenceBound(model=model, beta=self.beta)
        elif self.acquisition_type == "log_noisy_expected_improvement":
            if X_baseline is None:
                raise ValueError(
                    "log_noisy_expected_improvement requires X_baseline (training data)"
                )
            return qLogNoisyExpectedImprovement(
                model=model,
                X_baseline=X_baseline,
                sampler=sampler,
                **self.kwargs,
            )
        elif self.acquisition_type == "qEI":
            return qExpectedImprovement(
                model=model,
                best_f=best_f,
                sampler=sampler,
                **self.kwargs,
            )
        elif self.acquisition_type == "qLogEI":
            return qLogExpectedImprovement(
                model=model,
                best_f=best_f,
                sampler=sampler,
                **self.kwargs,
            )
        elif self.acquisition_type == "qNEI":
            # qNEI requires X_baseline
            if X_baseline is None:
                raise ValueError("qNEI requires X_baseline (training data)")
            return qNoisyExpectedImprovement(
                model=model,
                X_baseline=X_baseline,
                sampler=sampler,
                **self.kwargs,
            )
        elif self.acquisition_type == "qUCB":
            return qUpperConfidenceBound(
                model=model,
                beta=self.beta,
                sampler=sampler,
                **self.kwargs,
            )
        else:
            raise ValueError(f"Unknown acquisition type: {self.acquisition_type}")

    def __call__(
        self,
        search_candidates: list[Candidate],
        state: State,
    ) -> LabelledCandidates:
        """Compute acquisition values or optimize in continuous space.

        Args:
            search_candidates: Candidates from search function. If empty, performs
                continuous optimization. If provided, scores these candidates.
            state: Task state with surrogate model and training data.

        Returns:
            Candidates with acquisition scores.

        Raises:
            RuntimeError: If surrogate model not available.
            ValueError: If continuous optimization requires bounds.
        """
        if state.surrogate is None:
            raise RuntimeError("Surrogate model is required for acquisition function")

        # Get predictions from surrogate
        train_data = state.dataset.train_dataset
        best_f = float(train_data.labels.max())

        # Mode 1: Score discrete candidates
        if search_candidates:
            return self._score_candidates(search_candidates, state, best_f)

        # Mode 2: Continuous optimization
        return self._optimize_continuous(state, best_f)

    def _score_candidates(
        self,
        candidates: list[Candidate],
        state: State,
        best_f: float,
    ) -> LabelledCandidates:
        """Score a discrete pool of candidates.

        Args:
            candidates: Candidates to score.
            state: Task state.
            best_f: Best observed value.

        Returns:
            Candidates with acquisition scores.

        Raises:
            ValueError: If batch_size(q) is greater than the length of candidates.
            ValueError: If total candidates is not a multiple of batch_size(q).
        """
        logger.info(f"Scoring {len(candidates)} candidates with {self.acquisition_type}")

        raw_model = state.surrogate.model
        wrapped_model = (
            raw_model if isinstance(raw_model, BotorchModel) else BotorchModelWrapper(raw_model)
        )
        self._require_joint_posterior_for_batch(wrapped_model)

        # Infer tensor dtype from model parameters; fall back to float64 (BoTorch default)
        # if the model has no registered PyTorch parameters (e.g. pure ALF BaseModel).
        try:
            infer_dtype = next(wrapped_model.parameters()).dtype
        except StopIteration:
            infer_dtype = torch.float64

        # Get training data for qNEI / log_noisy_expected_improvement
        X_baseline = None
        if self.acquisition_type in ("qNEI", "log_noisy_expected_improvement"):
            X_baseline = candidates_to_tensor(
                state.dataset.train_dataset.candidates, dtype=infer_dtype
            )

        acq_fn = self._create_acquisition_function(wrapped_model, best_f, X_baseline)

        # Evaluate acquisition function
        # For discrete scoring, evaluate each candidate independently
        X = candidates_to_tensor(candidates, dtype=infer_dtype)

        if self.batch_size > X.shape[0]:
            raise ValueError("batch_size(q) greater than the length of candidates")

        if self.batch_size > 1 and X.shape[0] % self.batch_size != 0:
            raise ValueError(
                "Total candidates is not a multiple of batch_size(q)."
                f"total candidates= {X.shape[0]}, q-batch (q) = {self.batch_size}",
            )

        # Convert the input to a format of (b, q, d) where b is the t-batch
        X = X.reshape((-1, self.batch_size, X.shape[1]))

        with torch.no_grad():
            acq_val: Float[torch.Tensor, " b"] = acq_fn(X)
        # Replicate scores: each candidate in a q-batch gets the same score
        # [0.8, 0.6, 0.4] -> [0.8, 0.8, 0.6, 0.6, 0.4, 0.4]
        scores_replicated = np.repeat(acq_val.cpu().numpy(), self.batch_size)
        return LabelledCandidates(
            candidates=candidates,
            labels=np.array(scores_replicated),
        )

    def _optimize_continuous(
        self,
        state: State,
        best_f: float,
    ) -> LabelledCandidates:
        """Optimize acquisition function in continuous space.

        Args:
            state: Task state.
            best_f: Best observed value.

        Returns:
            Optimized candidates with acquisition scores.

        Raises:
            ValueError: If bounds not provided.
        """
        if self.bounds is None:
            raise ValueError(
                "Bounds must be provided for continuous optimization. "
                "Set bounds parameter when initializing BoTorchAcquisition."
            )

        logger.info(f"Optimizing {self.acquisition_type} with {self.num_restarts} restarts")

        # Resolve the surrogate to its joint-posterior BoTorch model when it has one
        # (native model, or ALF model exposing a trained `botorch_model`). Using it
        # directly gives analytic gradients for continuous optimisation. Marginal-only
        # models fall back to the wrapper (numerical gradients via predict()).
        inner_model = resolve_botorch_model(state.surrogate.model)
        model = (
            inner_model if inner_model is not None else BotorchModelWrapper(state.surrogate.model)
        )
        self._require_joint_posterior_for_batch(model)

        # Infer model dtype so bounds/data tensors match (avoids double != float errors).
        # Fall back to float64 (BoTorch default) when the model has no registered parameters
        # (e.g. an ALF BaseModel that is not an nn.Module).
        try:
            model_dtype = next(model.parameters()).dtype
        except StopIteration:
            model_dtype = torch.float64

        # Convert bounds from list[list[float]] to list[tuple[float, float]]
        bounds_tuples = [(b[0], b[1]) for b in self.bounds]
        bounds_tensor = get_bounds_tensor(
            bounds_tuples, device=state.surrogate.model.device, dtype=model_dtype
        )

        # Get training data for qNEI / log_noisy_expected_improvement
        X_baseline = None
        if self.acquisition_type in ("qNEI", "log_noisy_expected_improvement"):
            X_baseline = candidates_to_tensor(
                state.dataset.train_dataset.candidates, dtype=model_dtype
            )

        acq_fn = self._create_acquisition_function(model, best_f, X_baseline)

        # Optimize acquisition function
        candidates_tensor, acq_value = optimize_acqf(
            acq_function=acq_fn,
            bounds=bounds_tensor,
            q=self.batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            gen_candidates=gen_candidates_scipy,
            options={
                "batch_limit": self.optimization_config.batch_limit,
                "maxiter": self.optimization_config.maxiter,
                "nonnegative": self.optimization_config.nonnegative,
                "sample_around_best": self.optimization_config.sample_around_best,
                "sample_around_best_sigma": self.optimization_config.sample_around_best_sigma,
            },
            sequential=self.sequential,
        )

        # Convert back to candidates
        candidates = tensor_to_candidates(candidates_tensor)

        # Create acquisition scores (use the optimized value)
        # Handle both scalar and tensor acquisition values (sequential vs joint optimization)
        if isinstance(acq_value, torch.Tensor) and acq_value.numel() > 1:
            # Sequential optimization returns per-candidate values
            scores = acq_value.cpu().numpy()
        else:
            # Joint optimization returns single value for the batch
            acq_scalar = (
                acq_value.item() if isinstance(acq_value, torch.Tensor) else float(acq_value)
            )
            scores = np.full(len(candidates), acq_scalar)

        return LabelledCandidates(
            candidates=candidates,
            labels=scores,
        )
