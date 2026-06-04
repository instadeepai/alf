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

"""Regression metrics and supporting infrastructure.

All metrics are automatically registered in `regression_metric_registry` at
import time via the `@register_requires_variance` and
`@register_no_variance_required` decorators.
"""

import logging
import warnings
from functools import wraps
from typing import Any, Callable

import numpy as np
from alf_core.utils.metrics.base import (
    check_inputs,
    check_variance_validity,
    regression_metric_registry,
    require_min_samples,
)
from jaxtyping import Float
from scipy.stats import norm, pearsonr, spearmanr

logger = logging.getLogger("alf-core")

# np.trapezoid replaced np.trapz in NumPy 2.0; support both.
try:
    _np_trapz = np.trapezoid  # type: ignore[attr-defined]
except AttributeError:
    _np_trapz = np.trapz  # type: ignore[attr-defined]


def register_requires_variance(metric_fn: Callable) -> Callable:
    """Decorator to mark a metric as requiring variance.

    Automatically registers the metric in the global registry and applies input validation.
    The decorated function will receive validated inputs with variance checks.

    Args:
        metric_fn: The metric function to decorate.

    Returns:
        Wrapped metric function with validation and registration.
    """

    @wraps(metric_fn)
    def wrapper(
        means: np.ndarray,
        variances: np.ndarray,
        targets: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        check_inputs(means, targets)
        check_variance_validity(variances, targets)
        return metric_fn(means, variances, targets, *args, **kwargs)

    # Register the metric with variance requirement
    regression_metric_registry.register(metric_fn.__name__, wrapper, requires_variance=True)
    return wrapper


def register_no_variance_required(metric_fn: Callable) -> Callable:
    """Decorator to mark a metric as not requiring variance.

    Automatically registers the metric in the global registry and applies input validation.
    The decorated function will receive validated inputs without variance checks.

    Args:
        metric_fn: The metric function to decorate.

    Returns:
        Wrapped metric function with validation and registration.
    """

    @wraps(metric_fn)
    def wrapper(
        means: np.ndarray,
        variances: np.ndarray | None,
        targets: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        check_inputs(means, targets)
        return metric_fn(means, variances, targets, *args, **kwargs)

    # Register the metric without variance requirement
    regression_metric_registry.register(metric_fn.__name__, wrapper, requires_variance=False)
    return wrapper


def monte_carlo_ranking(
    means: Float[np.ndarray, " b"],
    variances: Float[np.ndarray, " b"],
    num_samples: int = 10000,
    seed: int = 42,
) -> tuple[Float[np.ndarray, " b"], Float[np.ndarray, " b"]]:
    """Compute ranks, their means and variances using Monte Carlo simulation.

    For predicted means and variances, estimates normally distributed means
    using Monte Carlo simulation and computes ranking statistics.

    Args:
        means: Array of shape (b,). Mean predictions.
        variances: Array of shape (b,). Predicted variances.
        num_samples: Number of random samples to draw for the simulation.
            Defaults to 10000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A tuple containing:
        - mean_rank: Mean rank for each candidate
        - rank_variances: Variance of ranks for each candidate
    """
    rng = np.random.default_rng(seed)
    n = len(means)

    # Simulate Gaussian scores
    mean_samples = rng.normal(loc=means, scale=np.sqrt(variances), size=(num_samples, n))

    # Compute hard ranks for each sample
    rank_samples = np.argsort(np.argsort(-mean_samples, axis=1), axis=1) + 1

    # Compute mean and std ranks over samples
    mean_rank = np.mean(rank_samples, axis=0)
    rank_variances = np.var(rank_samples, axis=0)

    return mean_rank, rank_variances


@register_no_variance_required
def mse(
    means: Float[np.ndarray, " b"],
    _: Float[np.ndarray, " b"] | None,
    targets: Float[np.ndarray, " b"],
) -> dict[str, float]:
    """Compute the mean squared error.

    For each sample compute the squared euclidean distance between
    the true label and the mean prediction. Compute the mean over these distances.

    Args:
        means: Array of shape (b,). Mean predictions
        _: Array of shape (b,) or None. Predicted variances
        targets: Array of shape (b,). True labels

    Returns:
        {"mse": MSE float}
    """
    return {"mse": ((targets - means) ** 2).mean(0)}


@register_no_variance_required
@require_min_samples(2)
def spearman(
    means: Float[np.ndarray, " b"],
    _: Float[np.ndarray, " b"] | None,
    targets: Float[np.ndarray, " b"],
) -> dict[str, float]:
    """Compute the spearman correlation.

    This is a non-parametric statistical test which measures
    the strength of the monotonic relationship between two variables.
    We rank the mean predictions and compute the correlation
    coefficient between these ranks and the true ranks.
    This takes a value between [-1,1]

    Args:
        means: Array of shape (b,). Mean predictions
        _: Array of shape (b,) or None. Predicted variances
        targets: Array of shape (b,). True labels

    Returns:
        {"spearman": Spearman correlation float}
    """
    return {"spearman": spearmanr(targets, means)[0]}


@register_no_variance_required
@require_min_samples(2)
def pearson(
    means: Float[np.ndarray, " b"],
    _: Float[np.ndarray, " b"] | None,
    targets: Float[np.ndarray, " b"],
) -> dict[str, float]:
    """Compute the pearson correlation.

    This is a statistical test which measures the strength
    and direction of the linear relationship between variables.
    We compute the correlation coefficient between the mean predictions and the target.
    This takes a value between [-1,1]

    Args:
        means: Array of shape (b,). Mean predictions
        _: Array of shape (b,) or None. Predicted variances
        targets: Array of shape (b,). True labels

    Returns:
        {"pearson": Pearson correlation float}
    """
    return {"pearson": pearsonr(targets, means)[0]}


@register_no_variance_required
def pairwise_xent(
    means: Float[np.ndarray, " b"],
    _: Float[np.ndarray, " b"] | None,
    targets: Float[np.ndarray, " b"],
) -> dict[str, float]:
    """Compute the ranking loss for a pairwise classification problem.

    For each pair of items in the batch, predict which item has the higher target value.
    Derive logits from pairs of predictions and treat them as logits of a binary classifier.

    Args:
        means: Array of shape (b,). Mean predictions
        _: Array of shape (b,) or None. Predicted variances
        targets: Array of shape (b,). True labels

    Returns:
        {"pairwise_xent": Ranking Loss float}
    """
    # Compute pairwise differences
    pairwise_logits = means[:, None] - means[None, :]
    pairwise_targets = targets[:, None] > targets[None, :]

    # Compute binary cross-entropy loss using logits
    log_exponent = np.logaddexp(0, -pairwise_logits)
    binary_cross_entropy = log_exponent - pairwise_logits * pairwise_targets
    ranking_xent = 0.5 * binary_cross_entropy

    # Mask out the diagonal (self-comparisons)
    diag_mask = 1 - np.eye(pairwise_logits.shape[0])
    ranking_xent = (ranking_xent * diag_mask).mean()

    return {"pairwise_xent": ranking_xent}


@register_requires_variance
def expected_calibration_error(
    means: Float[np.ndarray, " b"],
    variances: Float[np.ndarray, " b"],
    targets: Float[np.ndarray, " b"],
    n_grid_points: int = 100,
) -> dict[str, float]:
    """Compute Expected Calibration Error (ECE).

    For each confidence level alpha in a grid from 0 to 1:
    - Find alpha% confidence intervals for all predictions
    - Count percentage of targets that fall within the confidence intervals
    - ECE = area between x=y line and the observed coverage curve

    Lower ECE values indicate better calibration.

    Args:
        means: Array of shape (b,). Mean predictions.
        variances: Array of shape (b,). Predicted variances.
        targets: Array of shape (b,). True labels.
        n_grid_points: Number of grid points for confidence level discretization.
            Defaults to 100.

    Returns:
        Dictionary with key "ece" mapping to the ECE value.
    """
    grid = np.linspace(0, 1, n_grid_points)
    perc = np.zeros(n_grid_points)
    for i, cdf_cutoff in enumerate(grid):
        num_stds = norm.ppf(1 - ((1 - cdf_cutoff) / 2))
        perc[i] = (
            (targets >= means - num_stds * np.sqrt(variances))
            & (targets <= means + num_stds * np.sqrt(variances))
        ).mean()
    ece = np.sum(np.abs(perc - grid)) * (1 / n_grid_points)
    return {"ece": ece}


@register_requires_variance
def rank_expected_calibration_error(
    means: Float[np.ndarray, " b"],
    variances: Float[np.ndarray, " b"],
    targets: Float[np.ndarray, " b"],
) -> dict[str, float]:
    """Compute Expected Calibration Error (ECE) in rank space.

    Computes ECE using Monte Carlo ranking to estimate rank distributions,
    then applies the standard ECE computation in rank space.

    Lower ECE values indicate better calibration.

    Args:
        means: Array of shape (b,). Mean predictions.
        variances: Array of shape (b,). Predicted variances.
        targets: Array of shape (b,). True labels.

    Returns:
        Dictionary with key "rank_ece" mapping to the ECE value.
    """
    mean_rank, rank_variances = monte_carlo_ranking(means, variances, seed=42)
    target_ranks = (-targets).argsort().argsort() + 1

    ece = expected_calibration_error(mean_rank, rank_variances, target_ranks)["ece"]

    return {"rank_ece": ece}


@register_requires_variance
def width(
    _: Float[np.ndarray, " b"],
    variances: Float[np.ndarray, " b"],
    targets: Float[np.ndarray, " b"],
    alpha: float = 0.95,
) -> dict[str, float]:
    """Compute average confidence interval width normalised by dataset range.

    Computes alpha% confidence intervals for all predictions, then calculates
    the average width normalised by the maximum distance between any two targets.
    Lower values are better while maintaining good calibration.

    Args:
        _: Array of shape (b,). Unused parameter (for API consistency).
        variances: Array of shape (b,). Predicted variances.
        targets: Array of shape (b,). True labels.
        alpha: Confidence level (e.g., 0.95 for 95% CI). Defaults to 0.95.

    Returns:
        Dictionary with key "width_{alpha:.2f}" mapping to the normalised
        average width value.

    Raises:
        AssertionError: If alpha is not in [0, 1].
    """
    assert (alpha >= 0) and (alpha <= 1), "alpha should be in [0,1]"

    max_width_dataset = targets.max() - targets.min()
    if max_width_dataset == 0:
        logger.warning(
            "width: all targets are equal (range=0) — calibration width ratio is undefined. "
            "Returning empty dict."
        )
        return {}

    num_stds = norm.ppf(1 - ((1 - alpha) / 2))

    width_ci = 2 * num_stds * np.sqrt(variances)
    average_width_ci = width_ci.mean()

    avg_width_ratio = average_width_ci / max_width_dataset

    return {f"width_{alpha:.2f}": avg_width_ratio}


@register_requires_variance
def rank_width(
    means: Float[np.ndarray, " b"],
    variances: Float[np.ndarray, " b"],
    targets: Float[np.ndarray, " b"],
    alpha: float = 0.95,
) -> dict[str, float]:
    """Compute average confidence interval width in rank space.

    Computes width metric using Monte Carlo ranking to estimate rank distributions,
    then applies the standard width computation in rank space.

    Args:
        means: Array of shape (b,). Mean predictions.
        variances: Array of shape (b,). Predicted variances.
        targets: Array of shape (b,). True labels.
        alpha: Confidence level (e.g., 0.95 for 95% CI). Defaults to 0.95.

    Returns:
        Dictionary with key "rank_width_{alpha:.2f}" mapping to the normalised
        average width value in rank space.

    Raises:
        AssertionError: If alpha is not in [0, 1].
    """
    assert (alpha >= 0) and (alpha <= 1), "alpha should be in [0,1]"

    mean_rank, rank_variances = monte_carlo_ranking(means, variances, seed=42)
    target_ranks = (-targets).argsort().argsort() + 1

    avg_width_ratio = width(mean_rank, rank_variances, target_ranks, alpha)[f"width_{alpha:.2f}"]

    return {f"rank_width_{alpha:.2f}": avg_width_ratio}


@register_requires_variance
def coverage(
    means: Float[np.ndarray, " b"],
    variances: Float[np.ndarray, " b"],
    targets: Float[np.ndarray, " b"],
    alpha: float = 0.95,
) -> dict[str, float]:
    """Compute coverage at alpha% confidence level.

    Computes alpha% confidence intervals for all predictions and measures
    what percentage of targets fall within these intervals. Ideal coverage
    should be close to alpha.

    Args:
        means: Array of shape (b,). Mean predictions.
        variances: Array of shape (b,). Predicted variances.
        targets: Array of shape (b,). True labels.
        alpha: Confidence level (e.g., 0.95 for 95% CI). Defaults to 0.95.

    Returns:
        Dictionary with key "coverage_{alpha:.2f}" mapping to the coverage
        percentage.

    Raises:
        AssertionError: If alpha is not in [0, 1].
    """
    assert (alpha >= 0) and (alpha <= 1), "alpha should be in [0,1]"

    num_stds = norm.ppf(1 - ((1 - alpha) / 2))
    coverage_at_alpha = (
        (targets >= means - num_stds * np.sqrt(variances))
        & (targets <= means + num_stds * np.sqrt(variances))
    ).mean()

    return {f"coverage_{alpha:.2f}": coverage_at_alpha}


@register_requires_variance
def rank_coverage(
    means: Float[np.ndarray, " b"],
    variances: Float[np.ndarray, " b"],
    targets: Float[np.ndarray, " b"],
    alpha: float = 0.95,
) -> dict[str, float]:
    """Compute coverage at alpha% confidence level in rank space.

    Computes coverage metric using Monte Carlo ranking to estimate rank distributions,
    then applies the standard coverage computation in rank space.

    Args:
        means: Array of shape (b,). Mean predictions.
        variances: Array of shape (b,). Predicted variances.
        targets: Array of shape (b,). True labels.
        alpha: Confidence level (e.g., 0.95 for 95% CI). Defaults to 0.95.

    Returns:
        Dictionary with key "rank_coverage_{alpha:.2f}" mapping to the coverage
        percentage in rank space.

    Raises:
        AssertionError: If alpha is not in [0, 1].
    """
    assert (alpha >= 0) and (alpha <= 1), "alpha should be in [0,1]"

    mean_rank, rank_variances = monte_carlo_ranking(means, variances, seed=42)
    target_ranks = (-targets).argsort().argsort() + 1
    coverage_at_alpha = coverage(mean_rank, rank_variances, target_ranks, alpha)[
        f"coverage_{alpha:.2f}"
    ]

    return {f"rank_coverage_{alpha:.2f}": coverage_at_alpha}


@register_requires_variance
def residual_spearman(
    means: Float[np.ndarray, " b"],
    variances: Float[np.ndarray, " b"],
    targets: Float[np.ndarray, " b"],
) -> dict[str, float]:
    """Compute Spearman correlation between residuals and variances.

    Computes absolute residuals (|targets - means|) and measures their Spearman
    correlation with predicted variances. Higher correlation indicates better
    uncertainty estimation.

    Args:
        means: Array of shape (b,). Mean predictions.
        variances: Array of shape (b,). Predicted variances.
        targets: Array of shape (b,). True labels.

    Returns:
        Dictionary with key "residual_spearman" mapping to the correlation
        coefficient.
    """
    residuals = np.abs(targets - means)
    return {"residual_spearman": spearmanr(residuals, variances)[0]}


@register_requires_variance
def residual_pearson(
    means: Float[np.ndarray, " b"],
    variances: Float[np.ndarray, " b"],
    targets: Float[np.ndarray, " b"],
) -> dict[str, float]:
    """Compute Pearson correlation between residuals and standard deviations.

    Computes absolute residuals (|targets - means|) and measures their Pearson
    correlation with predicted standard deviations (sqrt(variances)). Higher
    correlation indicates better uncertainty estimation.

    Args:
        means: Array of shape (b,). Mean predictions.
        variances: Array of shape (b,). Predicted variances.
        targets: Array of shape (b,). True labels.

    Returns:
        Dictionary with key "residual_pearson" mapping to the correlation
        coefficient.
    """
    residuals = np.abs(targets - means)
    return {"residual_pearson": pearsonr(residuals, np.sqrt(variances))[0]}


@register_requires_variance
def regret_ucb_alpha(
    means: Float[np.ndarray, " b"],
    variances: Float[np.ndarray, " b"],
    targets: Float[np.ndarray, " b"],
    alpha: float = 0.1,
    num_acquisitions: int = 100,
) -> dict[str, float]:
    """Compute UCB (Upper Confidence Bound) acquisition regret.

    Computes UCB values (means + alpha * sqrt(variances)) and compares the sum
    of labels from the top num_acquisitions candidates selected by UCB versus
    the best possible sum. Lower regret indicates better acquisition performance.

    Args:
        means: Array of shape (b,). Mean predictions.
        variances: Array of shape (b,). Predicted variances.
        targets: Array of shape (b,). True labels.
        alpha: UCB exploration parameter. Defaults to 0.1.
        num_acquisitions: Number of candidates to acquire. Defaults to 100.

    Returns:
        Dictionary with key "regret_ucb_{alpha:.2f}" mapping to the cumulative
        regret value.

    Raises:
        AssertionError: If num_acquisitions is not a positive integer.
    """
    assert isinstance(num_acquisitions, int), "num_acquisitions should be an integer."
    assert num_acquisitions > 0, "num_acquisitions should be positive"
    # Handle case where num_acquisitions > available items
    if num_acquisitions > len(means):
        warnings.warn(
            f"num_acquisitions ({num_acquisitions}) is greater than the number"
            f"of available items ({len(means)}). Using round({len(means) / 2})"
            f"acquisitions instead.",
            stacklevel=2,
        )
        num_acquisitions = max(1, round(len(means) / 2))

    # With 1 candidate:
    # UCB would select that 1 candidate (the only option)
    # The "optimal" selection would also be that same 1 candidate
    # Regret would always be 0 (no matter how good/bad the model is)
    # The metric provides no useful signal about model quality
    # With 0 candidates:
    # The computation would break or return nonsensical results
    if len(means) < 2:
        warnings.warn(
            f"Dataset size ({len(means)}) is too small to compute UCB regret. Returning NaN.",
            stacklevel=2,
        )
        return {}

    # Compute UCB values
    ucb_values = means + alpha * np.sqrt(variances)

    # Sort true targets to get the best possible selections
    top_target_indices = np.argsort(-targets)[:num_acquisitions]
    best_possible_sum = np.sum(targets[top_target_indices])

    # Sort UCB values to get the actual selections
    top_ucb_indices = np.argsort(-ucb_values)[:num_acquisitions]
    selected_sum = np.sum(targets[top_ucb_indices])

    # Compute cumulative regret
    cumulative_regret = best_possible_sum - selected_sum

    return {f"regret_ucb_{alpha:.2f}": cumulative_regret}


@register_no_variance_required
def top_k_mean(
    _means: Float[np.ndarray, " b"],
    _variances: Float[np.ndarray, " b"] | None,
    targets: Float[np.ndarray, " b"],
    k: int = 10,
) -> dict[str, float]:
    """Compute the mean label of the top-k oracle-labelled candidates.

    Selects the k highest target values and returns their mean.  When k
    exceeds the number of available targets the function falls back to all
    available targets.  Use this as the primary optimisation metric to track
    how quickly an active-learning campaign surfaces high-performing
    candidates across rounds.

    Args:
        _means: Array of shape (b,). Unused (for API consistency with registry).
        _variances: Array of shape (b,) or None. Unused.
        targets: Array of shape (b,). Oracle labels for acquired candidates.
        k: Number of top candidates to consider. Defaults to 10.

    Returns:
        Dictionary with key ``top_{k_eff}_mean`` mapping to the mean value of
        the top-k targets.
    """
    k_eff = min(k, len(targets))
    top_k = np.partition(targets, -k_eff)[-k_eff:]
    return {f"top_{k_eff}_mean": float(top_k.mean())}


@register_no_variance_required
def top_k_max(
    _means: Float[np.ndarray, " b"],
    _variances: Float[np.ndarray, " b"] | None,
    targets: Float[np.ndarray, " b"],
    k: int = 10,
) -> dict[str, float]:
    """Compute the maximum label of the top-k oracle-labelled candidates.

    Selects the k highest target values and returns the maximum.  When k
    exceeds the number of available targets the function falls back to all
    available targets.  Use alongside ``top_k_mean`` to distinguish between
    campaigns that find one very good candidate versus many good ones.

    Args:
        _means: Array of shape (b,). Unused (for API consistency with registry).
        _variances: Array of shape (b,) or None. Unused.
        targets: Array of shape (b,). Oracle labels for acquired candidates.
        k: Number of top candidates to consider. Defaults to 10.

    Returns:
        Dictionary with key ``top_{k_eff}_max`` mapping to the maximum value
        of the top-k targets.
    """
    k_eff = min(k, len(targets))
    top_k = np.partition(targets, -k_eff)[-k_eff:]
    return {f"top_{k_eff}_max": float(top_k.max())}


def auc_top_k(
    round_values: Float[np.ndarray, " n_rounds"],
    best_value: float,
) -> dict[str, float]:
    """Compute the normalised area under the top-k mean curve.

    Integrates the per-round top-k mean values using the trapezoidal rule,
    then normalises the result so that a perfect campaign (one that always
    achieves ``best_value``) scores 1.0.  Lower values indicate that high-
    performing candidates were found later in the campaign.  Use as the
    primary leaderboard ranking metric for sample efficiency.

    Args:
        round_values: Array of shape (n_rounds,).  Per-round top-k mean, where
            entry ``i`` is the top-k mean of all candidates acquired by round
            ``i`` (inclusive).
        best_value: The global best oracle label in the dataset.  Used to
            normalise the AUC to [0, 1].

    Returns:
        Dictionary with key ``auc_top_k`` mapping to the normalised AUC.

    Raises:
        ValueError: If ``round_values`` has fewer than 2 entries or
            ``best_value`` is zero.
    """
    if len(round_values) < 2:
        raise ValueError(f"auc_top_k requires at least 2 rounds, got {len(round_values)}")
    if best_value == 0.0:
        raise ValueError("best_value must be non-zero to normalise the AUC")
    n = len(round_values)
    auc = float(_np_trapz(round_values / best_value, dx=1.0 / (n - 1)))
    return {"auc_top_k": auc}


@register_no_variance_required
def hit_rate(
    _means: Float[np.ndarray, " b"],
    _variances: Float[np.ndarray, " b"] | None,
    targets: Float[np.ndarray, " b"],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the fraction of acquired candidates whose label exceeds a threshold.

    Counts how many oracle-labelled candidates are considered "hits" (i.e.
    their label is at or above ``threshold``) and returns this as a fraction
    of the total.  Suitable for discovery-framing tasks such as drug screening
    or protein fitness optimisation, where "active" or "fit" is a binary
    concept derived from a continuous score.

    Args:
        _means: Array of shape (b,). Unused (for API consistency with registry).
        _variances: Array of shape (b,) or None. Unused.
        targets: Array of shape (b,). Oracle labels for acquired candidates.
        threshold: Minimum label value to count as a hit. Defaults to 0.5.

    Returns:
        Dictionary with key ``hit_rate_{threshold:.3f}`` mapping to the hit
        rate in [0, 1].
    """
    return {f"hit_rate_{threshold:.3f}": float((targets >= threshold).mean())}


@register_requires_variance
def nll_gaussian(
    means: Float[np.ndarray, " b"],
    variances: Float[np.ndarray, " b"],
    targets: Float[np.ndarray, " b"],
) -> dict[str, float]:
    """Compute the mean negative log-likelihood under a Gaussian predictive distribution.

    Evaluates the per-sample NLL assuming the model produces independent
    Gaussian predictions N(μ_i, σ²_i) for each candidate i:

        NLL = 0.5 * mean(log(2π) + log(σ²_i) + (y_i - μ_i)² / σ²_i)

    Lower values indicate that the model places high probability mass on the
    true targets.  Use together with ``expected_calibration_error`` to
    characterise both sharpness and calibration of the surrogate's uncertainty
    estimates.  Variances are clipped to a small positive value before
    computing the logarithm to guard against numerical instability.

    Args:
        means: Array of shape (b,). Mean predictions.
        variances: Array of shape (b,). Predicted variances.  Must be
            non-negative (enforced by the registry decorator).
        targets: Array of shape (b,). True labels.

    Returns:
        Dictionary with key ``nll`` mapping to the mean NLL value.
    """
    eps = np.finfo(float).tiny
    safe_vars = np.maximum(variances, eps)
    nll = 0.5 * np.mean(np.log(2 * np.pi * safe_vars) + (targets - means) ** 2 / safe_vars)
    return {"nll": float(nll)}


def calibration_curve(
    means: Float[np.ndarray, " b"],
    variances: Float[np.ndarray, " b"],
    targets: Float[np.ndarray, " b"],
    n_grid_points: int = 100,
) -> tuple[Float[np.ndarray, " n_grid_points"], Float[np.ndarray, " n_grid_points"]]:
    """Return the expected and observed coverage arrays for a reliability diagram.

    For each confidence level α in a uniform grid from 0 to 1, computes the
    observed fraction of targets that fall within the α-level prediction
    interval.  Plotting observed coverage against expected coverage yields the
    reliability diagram; perfect calibration lies on the diagonal.

    This function exposes the raw arrays used internally by
    ``expected_calibration_error``, enabling callers to render the diagram
    without re-computing the coverage sweep.

    Args:
        means: Array of shape (b,). Mean predictions.
        variances: Array of shape (b,). Predicted variances.
        targets: Array of shape (b,). True labels.
        n_grid_points: Number of confidence levels to evaluate.
            Defaults to 100.

    Returns:
        A tuple ``(expected_coverage, observed_coverage)`` where each array
        has shape (n_grid_points,) and values in [0, 1].
    """
    check_inputs(means, targets)
    check_variance_validity(variances, targets)
    grid = np.linspace(0, 1, n_grid_points)
    observed = np.zeros(n_grid_points)
    for i, alpha in enumerate(grid):
        n_stds = norm.ppf(1 - (1 - alpha) / 2)
        observed[i] = (
            (targets >= means - n_stds * np.sqrt(variances))
            & (targets <= means + n_stds * np.sqrt(variances))
        ).mean()
    return grid, observed


@register_requires_variance
def regret_ucb_alpha_sweep(
    means: Float[np.ndarray, " b"],
    variances: Float[np.ndarray, " b"],
    targets: Float[np.ndarray, " b"],
    alpha: float | list[float] | None = None,
    num_acquisitions: int = 100,
) -> dict[str, float]:
    """Compute UCB regret for multiple alpha values.

    Computes UCB regret for each alpha value in the provided list (or default range).
    Useful for evaluating acquisition function performance across different
    exploration-exploitation trade-offs.

    Args:
        means: Array of shape (b,). Mean predictions.
        variances: Array of shape (b,). Predicted variances.
        targets: Array of shape (b,). True labels.
        alpha: Single float, list of floats, or None. UCB exploration parameter(s).
            If None, uses default list [0.1, 0.3, 0.5, 1.0]. Defaults to None.
        num_acquisitions: Number of candidates to acquire. Defaults to 100.

    Returns:
        Dictionary mapping "regret_ucb_{alpha:.2f}" to regret value for each
        alpha value.

    Raises:
        ValueError: If alpha is an empty list.
        TypeError: If alpha is not a float or list of floats.
    """
    assert variances is not None, "UCB regret requires variances"
    assert np.all(variances >= 0), "All uncertainty values must be non-negative (variances)."

    # Set the default list if alpha was not provided.
    # This is an ugly solution, but setting a mutable object (a list)
    # as a default argument is dangerous in python as the default is only
    # initiated once.
    if alpha is None:
        alpha = [0.1, 0.3, 0.5, 1]

    # Normalize alpha to a list
    if isinstance(alpha, float):
        alpha_list = [alpha]
    elif isinstance(alpha, list):
        if len(alpha) == 0:
            raise ValueError("alpha must be a float or a list of floats")
        alpha_list = alpha
    else:
        raise TypeError("alpha must be a float or a list of floats.")

    regret_alpha_list = {}

    for a in alpha_list:
        # Compute UCB values
        regret_alpha = regret_ucb_alpha(means, variances, targets, a, num_acquisitions)

        regret_alpha_list.update(regret_alpha)
    return regret_alpha_list
