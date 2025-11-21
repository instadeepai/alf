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

import warnings
from functools import wraps
from typing import Any, Callable, Union

import numpy as np
from scipy.stats import norm, pearsonr, spearmanr


def check_inputs(means: np.ndarray, targets: np.ndarray) -> None:
    """Validate that means and targets arrays are compatible and valid.

    Args:
        means: Array of predicted means.
        targets: Array of target values.

    Raises:
        AssertionError: If shapes don't match, arrays are empty, or contain NaN values.
    """
    assert means.shape == targets.shape, (
        f"Means shape {means.shape} and targets shape {targets.shape} don't match"
    )
    assert len(means) != 0, "Empty input arrays"
    assert not (np.any(np.isnan(means))), "Mean prediction array contains NaN values"
    assert not (np.any(np.isnan(targets))), "Target array contains NaN values"


def check_variance_validity(variances: np.ndarray, targets: np.ndarray) -> None:
    """Validate that variance array is valid and compatible with targets.

    Args:
        variances: Array of predicted variances.
        targets: Array of target values.

    Raises:
        AssertionError: If variances is None, contains negative values, length
            doesn't match targets, or contains NaN values.
    """
    assert variances is not None, "This function requires variances but it is None"
    assert np.all(variances >= 0), "All uncertainty values must be non-negative (variances)."
    assert len(variances) == len(targets), (
        f"Length of variances vector ({len(variances)})"
        f"should equal length of targets vector ({len(targets)})"
    )
    assert not (np.any(np.isnan(variances))), "Variance arrays contain NaN values"


class MetricRegistry:
    """Simple registry for metrics with variance requirements."""

    def __init__(self) -> None:
        """Initialize an empty metric registry."""
        self.metrics: dict[str, Callable] = {}
        self.variance_required: dict[str, bool] = {}

    def register(self, name: str, metric_fn: Callable, requires_variance: bool = False) -> None:
        """Register a metric function in the registry.

        Args:
            name: Name identifier for the metric.
            metric_fn: Callable function that computes the metric.
            requires_variance: Whether this metric requires variance information.
        """
        self.metrics[name] = metric_fn
        self.variance_required[name] = requires_variance

    def get_metrics_requiring_variance(self) -> dict[str, Callable]:
        """Get all registered metrics that require variance.

        Returns:
            dict[str, Callable]: Dictionary mapping metric names to their functions.
        """
        return {name: fn for name, fn in self.metrics.items() if self.variance_required[name]}

    def get_metrics_not_requiring_variance(self) -> dict[str, Callable]:
        """Get all registered metrics that don't require variance.

        Returns:
            dict[str, Callable]: Dictionary mapping metric names to their functions.
        """
        return {name: fn for name, fn in self.metrics.items() if not self.variance_required[name]}


# Create the global registry instance
metric_registry = MetricRegistry()


def requires_variance(metric_fn: Callable) -> Callable:
    """Decorator to mark a metric as requiring variance.

    Automatically registers the metric in the global registry and applies input validation.
    The decorated function will receive validated inputs with variance checks.

    Args:
        metric_fn: The metric function to decorate.

    Returns:
        Callable: Wrapped metric function with validation and registration.
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
        check_variance_validity(variances, targets)
        return metric_fn(means, variances, targets, *args, **kwargs)

    # Register the metric with variance requirement
    metric_registry.register(metric_fn.__name__, wrapper, requires_variance=True)
    return wrapper


def no_variance_required(metric_fn: Callable) -> Callable:
    """Decorator to mark a metric as not requiring variance.

    Automatically registers the metric in the global registry and applies input validation.
    The decorated function will receive validated inputs without variance checks.

    Args:
        metric_fn: The metric function to decorate.

    Returns:
        Callable: Wrapped metric function with validation and registration.
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
    metric_registry.register(metric_fn.__name__, wrapper, requires_variance=False)
    return wrapper


def monte_carlo_ranking(
    means: np.ndarray,
    variances: np.ndarray,
    num_samples: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ranks, their means and variances using Monte Carlo simulation.

    For predicted means and variances, estimates normally distributed means
    using Monte Carlo simulation and computes ranking statistics.

    Args:
        means: Array of shape (b,). Mean predictions.
        variances: Array of shape (b,). Predicted variances.
        num_samples: Number of random samples to draw for the simulation.
            Defaults to 10000.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - mean_rank: Mean rank for each candidate
            - rank_variances: Variance of ranks for each candidate
    """
    np.random.seed(42)
    n = len(means)

    # Simulate Gaussian scores
    mean_samples = np.random.normal(loc=means, scale=np.sqrt(variances), size=(num_samples, n))

    # Compute hard ranks for each sample
    rank_samples = np.argsort(np.argsort(-mean_samples, axis=1), axis=1) + 1

    # Compute mean and std ranks over samples
    mean_rank = np.mean(rank_samples, axis=0)
    rank_variances = np.var(rank_samples, axis=0)

    return mean_rank, rank_variances


@no_variance_required
def mse(means: np.ndarray, _: np.ndarray | None, targets: np.ndarray) -> dict[str, float]:
    """Compute the mean squared error.

    For each sample compute the squared euclidean distance between
    the true label and the mean prediction. Compute the mean over these distances.

    Args:
        means: Array of shape (b,). Mean predictions
        _: Array of shape (b,) or None. Predicted variances
        targets: Array of shape (b,). True labels

    Returns:
        dict[str, float]: {"mse": MSE float}
    """
    return {"mse": ((targets - means) ** 2).mean(0)}


@no_variance_required
def spearman(means: np.ndarray, _: np.ndarray | None, targets: np.ndarray) -> dict[str, float]:
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
        dict[str, float]: {"spearman": Spearman correlation float}
    """
    return {"spearman": spearmanr(targets, means)[0]}


@no_variance_required
def pearson(means: np.ndarray, _: np.ndarray | None, targets: np.ndarray) -> dict[str, float]:
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
        dict[str, float]: {"pearson": Pearson correlation float}
    """
    return {"pearson": pearsonr(targets, means)[0]}


@no_variance_required
def pairwise_xent(means: np.ndarray, _: np.ndarray | None, targets: np.ndarray) -> dict[str, float]:
    """Compute the ranking loss for a pairwise classification problem.

    For each pair of items in the batch, predict which item has the higher target value.
    Derive logits from pairs of predictions and treat them as logits of a binary classifier.

    Args:
        means: Array of shape (b,). Mean predictions
        _: Array of shape (b,) or None. Predicted variances
        targets: Array of shape (b,). True labels

    Returns:
        dict[str, float]: {"pairwise_xent": Ranking Loss float}
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


@requires_variance
def expected_calibration_error(
    means: np.ndarray,
    variances: np.ndarray,
    targets: np.ndarray,
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
        dict[str, float]: Dictionary with key "ece" mapping to the ECE value.
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


@requires_variance
def rank_expected_calibration_error(
    means: np.ndarray,
    variances: np.ndarray,
    targets: np.ndarray,
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
        dict[str, float]: Dictionary with key "rank_ece" mapping to the ECE value.
    """
    mean_rank, rank_variances = monte_carlo_ranking(means, variances)
    target_ranks = (-targets).argsort().argsort() + 1

    ece = expected_calibration_error(mean_rank, rank_variances, target_ranks)["ece"]

    return {"rank_ece": ece}


@requires_variance
def width(
    _: np.ndarray,
    variances: np.ndarray,
    targets: np.ndarray,
    alpha: float = 0.95,
) -> dict[str, float]:
    """Compute average confidence interval width normalized by dataset range.

    Computes alpha% confidence intervals for all predictions, then calculates
    the average width normalized by the maximum distance between any two targets.
    Lower values are better while maintaining good calibration.

    Args:
        _: Array of shape (b,). Unused parameter (for API consistency).
        variances: Array of shape (b,). Predicted variances.
        targets: Array of shape (b,). True labels.
        alpha: Confidence level (e.g., 0.95 for 95% CI). Defaults to 0.95.

    Returns:
        dict[str, float]: Dictionary with key "width_{alpha:.2f}" mapping to
            the normalized average width value.

    Raises:
        AssertionError: If alpha is not in [0, 1].
    """
    assert (alpha >= 0) and (alpha <= 1), "alpha should be in [0,1]"

    max_width_dataset = targets.max() - targets.min()

    num_stds = norm.ppf(1 - ((1 - alpha) / 2))

    width_ci = 2 * num_stds * np.sqrt(variances)
    average_width_ci = width_ci.mean()

    avg_width_ratio = average_width_ci / max_width_dataset

    return {f"width_{alpha:.2f}": avg_width_ratio}


@requires_variance
def rank_width(
    means: np.ndarray,
    variances: np.ndarray,
    targets: np.ndarray,
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
        dict[str, float]: Dictionary with key "rank_width_{alpha:.2f}" mapping to
            the normalized average width value in rank space.

    Raises:
        AssertionError: If alpha is not in [0, 1].
    """
    assert (alpha >= 0) and (alpha <= 1), "alpha should be in [0,1]"

    mean_rank, rank_variances = monte_carlo_ranking(means, variances)
    target_ranks = (-targets).argsort().argsort() + 1

    avg_width_ratio = width(mean_rank, rank_variances, target_ranks, alpha)[f"width_{alpha:.2f}"]

    return {f"rank_width_{alpha:.2f}": avg_width_ratio}


@requires_variance
def coverage(
    means: np.ndarray,
    variances: np.ndarray,
    targets: np.ndarray,
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
        dict[str, float]: Dictionary with key "coverage_{alpha:.2f}" mapping to
            the coverage percentage.

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


@requires_variance
def rank_coverage(
    means: np.ndarray,
    variances: np.ndarray,
    targets: np.ndarray,
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
        dict[str, float]: Dictionary with key "rank_coverage_{alpha:.2f}" mapping to
            the coverage percentage in rank space.

    Raises:
        AssertionError: If alpha is not in [0, 1].
    """
    assert (alpha >= 0) and (alpha <= 1), "alpha should be in [0,1]"

    mean_rank, rank_variances = monte_carlo_ranking(means, variances)
    target_ranks = (-targets).argsort().argsort() + 1
    coverage_at_alpha = coverage(mean_rank, rank_variances, target_ranks, alpha)[
        f"coverage_{alpha:.2f}"
    ]

    return {f"rank_coverage_{alpha:.2f}": coverage_at_alpha}


@requires_variance
def residual_spearman(
    means: np.ndarray,
    variances: np.ndarray,
    targets: np.ndarray,
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
        dict[str, float]: Dictionary with key "residual_spearman" mapping to
            the correlation coefficient.
    """
    residuals = np.abs(targets - means)
    return {"residual_spearman": spearmanr(residuals, variances)[0]}


@requires_variance
def residual_pearson(
    means: np.ndarray,
    variances: np.ndarray,
    targets: np.ndarray,
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
        dict[str, float]: Dictionary with key "residual_pearson" mapping to
            the correlation coefficient.
    """
    residuals = np.abs(targets - means)
    return {"residual_pearson": pearsonr(residuals, np.sqrt(variances))[0]}


@requires_variance
def regret_ucb_alpha(
    means: np.ndarray,
    variances: np.ndarray,
    targets: np.ndarray,
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
        dict[str, float]: Dictionary with key "regret_ucb_{alpha:.2f}" mapping to
            the cumulative regret value.

    Raises:
        AssertionError: If num_acquisitions is not a positive integer.
    """
    assert isinstance(num_acquisitions, int), "num_acquisitions should be an integer."
    assert num_acquisitions > 0, "num_acquisitions should be positive"
    # Handle case where num_acquisitions > available items
    if num_acquisitions > len(means):
        warnings.warn(
            f"num_acquisitions ({num_acquisitions}) is greater than the number "
            f"of available items ({len(means)}). Using round({len(means)}/2) acquisitions instead.",
            stacklevel=2,
        )
        num_acquisitions = round(len(means) / 2)

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


@requires_variance
def regret_ucb_alpha_sweep(
    means: np.ndarray,
    variances: np.ndarray,
    targets: np.ndarray,
    alpha: Union[float, list[float]] | None = None,
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
        dict[str, float]: Dictionary mapping "regret_ucb_{alpha:.2f}" to regret value
            for each alpha value.

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
