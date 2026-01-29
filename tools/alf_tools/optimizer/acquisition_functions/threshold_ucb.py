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

from typing import Optional

import numpy as np
from alf_core import AcquisitionFunction, Candidate, LabelledCandidates, Predictions, TaskState
from scipy.stats import norm


class ThresholdUCB(AcquisitionFunction):
    """Upper Confidence Bound acquisition with threshold filtering.

    Selects points where the upper confidence bound (mu + beta * sigma)
    exceeds the threshold, then ranks them by acquisition score combining
    probability of improvement and exploration.

    This acquisition function is designed for discovering ALL regions above
    a threshold, rather than finding a single optimum.
    """

    def __init__(
        self,
        threshold: float,
        beta: float = 1.0,
        exploration_weight: float = 0.2,
        diversity_penalty: float = 0.0,
        diversity_length_scale: float = 1.0,
    ) -> None:
        """Initialize the ThresholdUCB acquisition function.

        Args:
            threshold: The target threshold value to exceed.
            beta: Number of standard deviations for UCB filter
                (mu + beta*sigma > threshold). Higher values are more
                optimistic/exploratory. Default is 1.0.
            exploration_weight: Weight for exploration bonus in acquisition
                scoring. Range [0, 1], where 0 = pure exploitation,
                1 = pure exploration. Default is 0.2 (20% exploration,
                80% exploitation).
            diversity_penalty: Weight for diversity penalty to spread points
                apart in batch acquisition. Range [0, inf), where 0 = no
                penalty (allows clustering), higher values = stronger penalty
                for points near already selected points. Default is 0.0.
            diversity_length_scale: Length scale for computing distances in
                diversity penalty. Smaller values make the penalty more
                localized. Default is 1.0.
        """
        super().__init__()
        self.threshold = threshold
        self.beta = beta
        self.exploration_weight = exploration_weight
        self.diversity_penalty = diversity_penalty
        self.diversity_length_scale = diversity_length_scale
        self.selected_in_batch: list[int] = []
        self._current_candidates: Optional[list[Candidate]] = None

    def _compute_diversity_penalty(
        self, candidates: list, selected_indices: list[int]
    ) -> np.ndarray:
        """Compute diversity penalty based on distance to selected points.

        The penalty is computed as a sum of RBF kernels centered at each
        selected point, encouraging selection of points far from already
        selected ones.

        Args:
            candidates: List of candidate points
            selected_indices: Indices of points already selected in this batch

        Returns:
            Array of penalty values (higher = closer to selected points)
        """
        if len(selected_indices) == 0 or self.diversity_penalty == 0.0:
            return np.zeros(len(candidates))

        # Extract features from candidates
        candidate_features = np.array([c.data for c in candidates])
        if candidate_features.ndim == 1:
            candidate_features = candidate_features.reshape(-1, 1)

        # Get selected point features
        selected_features = candidate_features[selected_indices]

        # Compute pairwise distances to all selected points
        # Shape: (n_candidates, n_selected)
        dists = np.linalg.norm(
            candidate_features[:, np.newaxis, :] - selected_features[np.newaxis, :, :], axis=2
        )

        # Apply RBF kernel: exp(-dist^2 / (2 * length_scale^2))
        penalties = np.exp(-(dists**2) / (2 * self.diversity_length_scale**2))

        # Sum penalties from all selected points
        total_penalty = np.sum(penalties, axis=1)

        return total_penalty

    def _get_acquisition_values(self, predictions: Predictions, state: TaskState) -> np.ndarray:
        """Computes acquisition values based on surrogate predictions.

        Process:
        1. UCB Filter: Only consider candidates where
           mu + beta*sigma > threshold
        2. Exploitation: Compute probability of exceeding threshold
           P(f(x) > threshold)
        3. Exploration: Compute normalized uncertainty (sigma)
        4. Combine: acquisition = (1 - w) * PI + w * uncertainty
        5. Diversity: Subtract penalty for points near already selected ones
        6. Mask: Set acquisition to 0 for candidates that fail UCB filter

        Args:
            predictions: The predictions from the surrogate model.
            state: The task state containing the dataset and surrogate
                model.

        Returns:
            The acquisition values for the candidates. Returns 0 for
            candidates where the upper confidence bound does not exceed
            the threshold.

        Raises:
            ValueError: If `empirical_dist` or `variances` is not found
                in predictions.
        """
        if predictions.empirical_dist is not None:
            # Compute mean and std from empirical distribution
            mu = np.mean(predictions.empirical_dist, axis=-1)
            uncertainty = np.std(predictions.empirical_dist, axis=-1)

            # UCB filter: only consider points where
            # mu + beta*sigma > threshold
            ucb = mu + self.beta * uncertainty
            ucb_mask = ucb > self.threshold

            # Exploitation: probability of exceeding threshold
            pi = np.mean(np.maximum(predictions.empirical_dist - self.threshold, 0), -1)

            # Exploration: normalized uncertainty
            max_uncertainty = np.max(uncertainty) if np.max(uncertainty) > 0 else 1.0
            normalized_uncertainty = uncertainty / max_uncertainty

            # Combine exploitation and exploration
            exploitation_weight = 1.0 - self.exploration_weight
            acquisition_values = (
                exploitation_weight * pi + self.exploration_weight * normalized_uncertainty
            )

            # Apply diversity penalty if enabled
            if (
                self.diversity_penalty > 0.0
                and len(self.selected_in_batch) > 0
                and self._current_candidates is not None
            ):
                diversity_penalty = self._compute_diversity_penalty(
                    self._current_candidates, self.selected_in_batch
                )
                acquisition_values = acquisition_values - self.diversity_penalty * diversity_penalty

            # Apply UCB mask: set acquisition to 0 where UCB <= threshold
            return np.where(ucb_mask, acquisition_values, 0.0)

        elif predictions.variances is not None:
            mu = predictions.means
            sigma = np.sqrt(predictions.variances)
            sigma = np.clip(sigma, 1e-9, None)

            # UCB filter: only consider points where
            # mu + beta*sigma > threshold
            ucb = mu + self.beta * sigma
            ucb_mask = ucb > self.threshold

            # Exploitation: probability of improvement over threshold
            z = (mu - self.threshold) / sigma
            pi = norm.cdf(z)

            # Exploration: normalized uncertainty bonus
            max_sigma = np.max(sigma) if np.max(sigma) > 0 else 1.0
            normalized_uncertainty = sigma / max_sigma

            # Combine: exploitation and exploration
            exploitation_weight = 1.0 - self.exploration_weight
            acquisition_values = (
                exploitation_weight * pi + self.exploration_weight * normalized_uncertainty
            )

            # Apply diversity penalty if enabled
            if (
                self.diversity_penalty > 0.0
                and len(self.selected_in_batch) > 0
                and self._current_candidates is not None
            ):
                diversity_penalty = self._compute_diversity_penalty(
                    self._current_candidates, self.selected_in_batch
                )
                acquisition_values = acquisition_values - self.diversity_penalty * diversity_penalty

            # Apply UCB mask: set acquisition to 0 where UCB <= threshold
            acquisition_values = np.where(ucb_mask, acquisition_values, 0.0)

            return np.maximum(acquisition_values, 0.0)
        else:
            raise ValueError(
                "Expected either `empirical_dist` or `variances` in "
                "predictions, but neither was found. Cannot compute "
                "ThresholdUCB acquisition values."
            )

    def __call__(
        self,
        search_candidates: list,
        state: TaskState,
    ):
        """Compute acquisition values for candidates with diversity penalty support.

        This method overrides the default __call__ to support iterative batch
        selection with diversity penalty. When diversity_penalty > 0, it performs
        greedy sequential selection where each point is selected considering
        distance to previously selected points.

        Args:
            search_candidates: List of Candidate objects to score
            state: Current task state containing the dataset and surrogate model

        Returns:
            LabelledCandidates with acquisition values (batch diversity already
            considered if diversity_penalty > 0)
        """
        # Reset batch tracking
        self.selected_in_batch = []
        self._current_candidates = search_candidates

        # If no diversity penalty, use standard approach
        if self.diversity_penalty == 0.0:
            predictions = state.surrogate.predict(search_candidates)
            acquisition_values = self._get_acquisition_values(predictions, state)
            return LabelledCandidates(search_candidates, acquisition_values)

        # With diversity penalty, use iterative selection
        # Get predictions once (they don't change within a batch)
        predictions = state.surrogate.predict(search_candidates)

        # Initialize all acquisition values to compute once
        base_acquisition_values = self._get_acquisition_values(predictions, state)

        # Track which points would be selected with diversity
        # We'll adjust their scores based on the greedy selection order
        acquisition_values = base_acquisition_values.copy()

        # Simulate greedy selection to determine diversity-adjusted scores
        selected_indices = []
        for rank in range(min(state.acq_batch_size, len(search_candidates))):
            if np.max(acquisition_values) <= 0:
                break

            # Select best remaining candidate
            best_idx = np.argmax(acquisition_values)
            selected_indices.append(best_idx)

            # Mark as selected for diversity penalty calculation
            self.selected_in_batch.append(best_idx)

            # Recompute acquisition values with updated diversity penalty
            # This affects the next iteration's selection
            acquisition_values = self._get_acquisition_values(predictions, state)

        # Reset state
        self.selected_in_batch = []
        self._current_candidates = None

        # Return the base acquisition values (before diversity adjustment)
        # The optimizer will use get_top_k which will naturally select
        # the points we identified through greedy selection
        # Actually, we need to return values that reflect the greedy order
        # Assign values based on selection order (higher rank = higher value)
        final_values = np.zeros_like(base_acquisition_values)
        for i, idx in enumerate(selected_indices):
            # Assign decreasing values based on selection order
            final_values[idx] = len(selected_indices) - i + np.max(base_acquisition_values)

        return LabelledCandidates(search_candidates, final_values)
