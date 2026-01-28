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

"""BoTorch qExpectedImprovement acquisition function for ALF.

This module provides a wrapper around BoTorch's qExpectedImprovement (qEI)
acquisition function, enabling batch acquisition with joint optimization
for diversity and quality.
"""

import logging

import numpy as np
import torch
from alf_core import AcquisitionFunction, Candidate, LabelledCandidates, TaskState
from alf_tools.utils.botorch_utils import (
    predictions_to_posterior,
    tensor_to_candidates,
)
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from torch.distributions import Normal

logger = logging.getLogger("alf-tools")


class BoTorchQEI(AcquisitionFunction):
    """BoTorch qExpectedImprovement acquisition function.

    This acquisition function uses BoTorch's qExpectedImprovement to jointly
    optimize batches of candidates for both quality and diversity. Unlike
    greedy batch selection (where candidates are scored independently), qEI
    considers the joint value of selecting multiple candidates together.

    Key Features:
    - Batch acquisition: Jointly selects q candidates
    - MC sampling: Uses quasi-Monte Carlo for approximation
    - Optimization: Uses BoTorch's optimizers for continuous spaces

    Attributes:
        batch_size: Number of candidates to acquire per round.
        num_restarts: Number of random restarts for optimization.
        raw_samples: Number of random samples for initialization.
        mc_samples: Number of Monte Carlo samples for qEI approximation.
        bounds: Input space bounds as tensor of shape (2, d).
        optimize_sequential: Whether to optimize candidates sequentially (greedy)
            or jointly (default: False for true batch optimization).

    Example:
        >>> # For use with continuous/tabular data (e.g., BoTorch test functions)
        >>> acq_fn = BoTorchQEI(
        ...     batch_size=5,
        ...     bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        ...     num_restarts=10,
        ...     raw_samples=512
        ... )
        >>> # Use in optimization loop
        >>> new_candidates = acq_fn(search_candidates=[], state=state)
    """

    def __init__(
        self,
        batch_size: int = 1,
        bounds: torch.Tensor | None = None,
        num_restarts: int = 10,
        raw_samples: int = 512,
        mc_samples: int = 128,
        optimize_sequential: bool = False,
        seed: int = 42,
    ) -> None:
        """Initialize BoTorchQEI acquisition function.

        Args:
            batch_size: Number of candidates to acquire (q in qEI).
            bounds: Tensor of shape (2, d) with lower bounds in first row and
                upper bounds in second row. Required for optimization mode.
            num_restarts: Number of random restarts for optimization.
            raw_samples: Number of random samples for initialization.
            mc_samples: Number of quasi-Monte Carlo samples for qEI approximation.
            optimize_sequential: If True, optimize candidates sequentially (greedy).
                If False, optimize jointly for true batch acquisition.
            seed: Random seed for reproducibility.
        """
        self.batch_size = batch_size
        self.bounds = bounds
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.mc_samples = mc_samples
        self.optimize_sequential = optimize_sequential
        self.seed = seed

        # Device detection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.bounds is not None:
            self.bounds = self.bounds.to(self.device)

        logger.info(
            f"Initialized BoTorchQEI with batch_size={batch_size}, "
            f"num_restarts={num_restarts}, mc_samples={mc_samples}, "
            f"device={self.device}"
        )

    def __call__(
        self,
        search_candidates: list[Candidate],
        state: TaskState,
    ) -> LabelledCandidates:
        """Compute qEI acquisition values or optimize for new candidates.

        This function operates in two modes:

        1. **Scoring mode** (search_candidates provided):
           Scores each candidate independently using Expected Improvement.
           Used when working with discrete candidate pools.

        2. **Optimization mode** (search_candidates empty):
           Uses BoTorch's optimizer to find optimal candidates in continuous space.
           Requires bounds to be set. Used with continuous optimization (e.g.,
           BoTorch test functions).

        Args:
            search_candidates: List of candidates to score. If empty, will
                optimize new candidates using BoTorch's optimizer (requires bounds).
            state: Task state containing dataset and surrogate model.

        Returns:
            LabelledCandidates with acquisition values (scoring mode) or
            optimal candidates (optimization mode).

        Raises:
            ValueError: If optimization mode is used without bounds.
        """
        if search_candidates:
            # Scoring mode: Score provided candidates
            return self._score_candidates(search_candidates, state)
        else:
            # Optimization mode: Optimize new candidates
            return self._optimize_candidates(state)

    def _score_candidates(
        self,
        search_candidates: list[Candidate],
        state: TaskState,
    ) -> LabelledCandidates:
        """Score candidates using Expected Improvement.

        For discrete candidate pools, we score each candidate independently
        using the analytic EI formula (no Monte Carlo needed).

        Args:
            search_candidates: List of candidates to score.
            state: Task state with dataset and surrogate.

        Returns:
            Candidates with EI scores as labels.
        """
        # Get predictions from surrogate
        predictions = state.surrogate.predict(search_candidates)

        # Get best observed value
        best_f = state.dataset.train_dataset.labels.max()

        # Convert to tensors
        posterior = predictions_to_posterior(predictions, device=self.device)

        # Use analytic EI for scoring (no MC needed for independent evaluation)
        # Manually compute EI using the posterior
        with torch.no_grad():
            # EI formula: (μ - best_f) * Φ(Z) + σ * φ(Z)
            mean = posterior.mean.squeeze(-1)
            variance = posterior.variance.squeeze(-1)
            sigma = torch.sqrt(variance)

            # Compute Z-score
            Z = (mean - best_f) / sigma
            Z = torch.clamp(Z, min=-10, max=10)  # Numerical stability

            # Standard normal CDF and PDF
            normal = Normal(0, 1)
            cdf_Z = normal.cdf(Z)
            pdf_Z = torch.exp(normal.log_prob(Z))

            # EI formula
            ei = (mean - best_f) * cdf_Z + sigma * pdf_Z
            ei = torch.clamp(ei, min=0)  # EI is non-negative

        acquisition_values = ei.cpu().numpy()

        return LabelledCandidates(candidates=search_candidates, labels=acquisition_values)

    def _optimize_candidates(self, state: TaskState) -> LabelledCandidates:
        """Optimize new candidates using BoTorch's continuous optimizer.

        Args:
            state: Task state with dataset and surrogate.

        Returns:
            Optimal candidates found by optimization.

        Raises:
            ValueError: If bounds are not set.
        """
        if self.bounds is None:
            raise ValueError(
                "Bounds must be provided for optimization mode. "
                "Set bounds when initializing BoTorchQEI."
            )

        # Get best observed value
        best_f = state.dataset.train_dataset.labels.max()

        # Create a simple wrapper to get posteriors
        class SurrogateWrapper:
            """Wrapper to make ALF surrogate compatible with BoTorch."""

            def __init__(self, surrogate, device):
                self.surrogate = surrogate
                self.device = device

            def posterior(self, X):
                """Get posterior at X.

                Args:
                    X: Input tensor of shape (n, d).

                Returns:
                    GPyTorchPosterior with predictions at X.
                """
                # Convert tensor to candidates
                candidates = tensor_to_candidates(X.cpu())
                # Get predictions
                predictions = self.surrogate.predict(candidates)
                # Convert to posterior
                return predictions_to_posterior(predictions, device=self.device)

        model = SurrogateWrapper(state.surrogate, self.device)

        # Create qEI acquisition function
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_samples]))
        qei = qExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
        )

        # Optimize acquisition function
        torch.manual_seed(self.seed)
        candidates_tensor, acq_value = optimize_acqf(
            acq_function=qei,
            bounds=self.bounds,
            q=self.batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,
            sequential=self.optimize_sequential,
        )

        # Convert to ALF candidates
        optimal_candidates = tensor_to_candidates(candidates_tensor)

        # Use acquisition values as labels
        # For batch, we get a single value; distribute evenly
        if self.batch_size > 1:
            acq_values = np.ones(self.batch_size) * acq_value.item() / self.batch_size
        else:
            acq_values = np.array([acq_value.item()])

        logger.info(
            f"Optimized {self.batch_size} candidates with qEI value: {acq_value.item():.4f}"
        )

        return LabelledCandidates(candidates=optimal_candidates, labels=acq_values)
