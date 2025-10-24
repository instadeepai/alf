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
from typing import Any, Callable, Optional, Tuple, Union
from inspect import signature

import numpy as np
from scipy.stats import norm, pearsonr, spearmanr


def check_inputs(means: np.ndarray, targets: np.ndarray) -> None:
    assert (
        means.shape == targets.shape
    ), f"Means shape {means.shape} and targets shape {targets.shape} don't match"
    assert len(means) != 0, "Empty input arrays"
    assert not (np.any(np.isnan(means))), "Mean prediction array contains NaN values"
    assert not (np.any(np.isnan(targets))), "Target array contains NaN values"


def check_variance_validity(variances: np.ndarray, targets: np.ndarray) -> None:
    assert variances is not None, "This function requires variances but it is None"
    assert np.all(
        variances >= 0
    ), "All uncertainty values must be non-negative (variances)."
    assert len(variances) == len(targets), (
        f"Length of variances vector ({len(variances)})"
        f"should equal length of targets vector ({len(targets)})"
    )
    assert not (np.any(np.isnan(variances))), "Variance arrays contain NaN values"


class MetricRegistry:
    """Simple registry for metrics with variance requirements."""
    
    def __init__(self):
        self.metrics: dict[str, Callable] = {}
        self.variance_required: dict[str, bool] = {}
    
    def register(self, name: str, metric_fn: Callable, requires_variance: bool = False):
        """Register a metric function."""
        self.metrics[name] = metric_fn
        self.variance_required[name] = requires_variance
    
    def get_metrics_requiring_variance(self) -> dict[str, Callable]:
        """Get metrics that require variance."""
        return {name: fn for name, fn in self.metrics.items() if self.variance_required[name]}
    
    def get_metrics_not_requiring_variance(self) -> dict[str, Callable]:
        """Get metrics that don't require variance."""
        return {name: fn for name, fn in self.metrics.items() if not self.variance_required[name]}


# Create the global registry instance
metric_registry = MetricRegistry()


def requires_variance(metric_fn: Callable) -> Callable:
    """
    Decorator to mark a metric as requiring variance.
    This automatically registers the metric in the global registry and applies input validation.
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
    """
    Decorator to mark a metric as not requiring variance.
    This automatically registers the metric in the global registry and applies input validation.
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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ranks, their means and variances (ranking uncertainty):

    For predicted means and variances, estimate n normally
    distributed means using Monte Carlo simulation.

    Args:
        means: Array of shape (b,). Mean predictions
        variances: Array of shape (b,) or None. Predicted variances
        num_samples: Integer. Number of random samples to draw for the simulation.

    Returns:
        Tuple: [mean_rank, rank_variances]
    """

    np.random.seed(42)
    n = len(means)

    # Simulate Gaussian scores
    mean_samples = np.random.normal(
        loc=means, scale=np.sqrt(variances), size=(num_samples, n)
    )

    # Compute hard ranks for each sample
    rank_samples = np.argsort(np.argsort(-mean_samples, axis=1), axis=1) + 1

    # Compute mean and std ranks over samples
    mean_rank = np.mean(rank_samples, axis=0)
    rank_variances = np.var(rank_samples, axis=0)

    return mean_rank, rank_variances


@no_variance_required
def mse(
    means: np.ndarray, _: np.ndarray | None, targets: np.ndarray
) -> dict[str, float]:
    """
    Compute the mean squared error.

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
def spearman(
    means: np.ndarray, _: np.ndarray | None, targets: np.ndarray
) -> dict[str, float]:
    """
    Compute the spearman correlation.

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
def pearson(
    means: np.ndarray, _: np.ndarray | None, targets: np.ndarray
) -> dict[str, float]:
    """
    Compute the pearson correlation.

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
def pairwise_xent(
    means: np.ndarray, _: np.ndarray | None, targets: np.ndarray
) -> dict[str, float]:
    """
    Compute the ranking loss for a pairwise classification problem.

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
    """
    Compute Expected Calibration Error (ECE):
    -- requires uncertainty --
    For alpha in grid from 0 -> 1
    Find alpha% confidence intervals for all predictions
    Count % of targets which fall within the confidence intervals
    Compute area between x=y and the curve

    We want to minimize this metric

    Args:
        means: Array of shape (b,). Mean predictions
        _: Array of shape (b,) or None. Predicted variances
        targets: Array of shape (b,). True labels
        n_grid_points: Integer for the coarseness of the grid

    Returns:
        dict: {"ece": ECE float}
    """
    check_variance_validity(variances, targets)

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
    """
    Compute Expected Calibration Error (ECE) in ranks:
    -- requires uncertainty --
    For alpha in grid from 0 -> 1
    Find alpha% confidence intervals for all predictions
    Count % of targets which fall within the confidence intervals
    Compute area between x=y and the curve

    We want to minimize this metric

    Args:
        means: Array of shape (b,). Mean predictions
        _: Array of shape (b,) or None. Predicted variances
        targets: Array of shape (b,). True labels
        n_grid_points: Integer for the coarseness of the grid

    Returns:
        dict: {"ece": ECE float}
    """
    check_variance_validity(variances, targets)

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
    """
    Compute width @ alpha%:
    -- requires uncertainty --
    Find (alpha%) confidence intervals for all predictions
    Compute the average width of all the confidence intervals
    Find the max distance between any two targets
    Compute the ratio:
    avg_ci / max_width

    We want this to be as small as possible whilst still retaining
    good calibration (i.e. ECE, coverage)

    Args:
        means: Array of shape (b,). Mean predictions
        _: Array of shape (b,) or None. Predicted variances
        targets: Array of shape (b,). True labels
        n_grid_points: Integer for the coarseness of the grid

    Returns:
        dict: {"width_{alpha}": Width @ alpha% float}
    """
    check_variance_validity(variances, targets)
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
    """
    Compute width @ alpha% based on ranks:
    -- requires uncertainty --
    Find (alpha%) confidence intervals for all predictions
    Compute the average width of all the confidence intervals
    Find the max distance between any two targets
    Compute the ratio:
    avg_ci / max_width

    We want this to be as small as possible whilst still retaining
    good calibration (i.e. ECE, coverage)

    Args:
        means: Array of shape (b,). Mean predictions
        _: Array of shape (b,) or None. Predicted variances
        targets: Array of shape (b,). True labels
        n_grid_points: Integer for the coarseness of the grid

    Returns:
        dict: {"width_{alpha}": Width @ alpha% float}
    """
    check_variance_validity(variances, targets)
    assert (alpha >= 0) and (alpha <= 1), "alpha should be in [0,1]"

    mean_rank, rank_variances = monte_carlo_ranking(means, variances)
    target_ranks = (-targets).argsort().argsort() + 1

    avg_width_ratio = width(mean_rank, rank_variances, target_ranks, alpha)[
        f"width_{alpha:.2f}"
    ]

    return {f"rank_width_{alpha:.2f}": avg_width_ratio}


@requires_variance
def coverage(
    means: np.ndarray,
    variances: np.ndarray,
    targets: np.ndarray,
    alpha: float = 0.95,
) -> dict[str, float]:
    """
    Compute coverage @ alpha%:
    -- requires uncertainty --
    Find (alpha%) confidence intervals for all predictions
    What % of targets fall within the approximate confidence intervals
    We want this to be as close to alpha as possible

    Args:
        means: Array of shape (b,). Mean predictions
        _: Array of shape (b,) or None. Predicted variances
        targets: Array of shape (b,). True labels
        alpha: float. Confidence level

    Returns:
        dict: {"coverage_{alpha}": Coverage @ alpha% float}
    """
    check_variance_validity(variances, targets)
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
    """
    Compute coverage @ alpha% based on ranks:
    -- requires uncertainty --
    Find (alpha%) confidence intervals for all predictions
    What % of targets fall within the approximate confidence intervals
    We want this to be as close to alpha as possible

    Args:
        means: Array of shape (b,). Mean predictions
        _: Array of shape (b,) or None. Predicted variances
        targets: Array of shape (b,). True labels
        alpha: float. Confidence level

    Returns:
        dict: {"coverage_{alpha}": Coverage @ alpha% float}
    """
    check_variance_validity(variances, targets)
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
    """
    Compute spearman correlation between the residuals and the variances:
    -- requires uncertainty --
    Compute the residuals: abs(targets - means)
    Compute the spearman between residuals and variances

    We expect these two things to be correlated

    Args:
        means: Array of shape (b,). Mean predictions
        _: Array of shape (b,) or None. Predicted variances
        targets: Array of shape (b,). True labels

    Returns:
        dict: {"residual_spearman": Residual Spearman}
    """
    check_variance_validity(variances, targets)

    residuals = np.abs(targets - means)
    return {"residual_spearman": spearmanr(residuals, variances)[0]}


@requires_variance
def residual_pearson(
    means: np.ndarray,
    variances: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """
    Compute pearson correlation between the residuals and the standard deviations
    -- requires uncertainty --
    Compute the residuals: abs(targets - means)
    Compute the pearson between residuals and standard deviations

    We expect these two things to be correlated

    Args:
        means: Array of shape (b,). Mean predictions
        _: Array of shape (b,) or None. Predicted variances
        targets: Array of shape (b,). True labels

    Returns:
        dict: {"residual_pearson": Residual Pearson float}
    """
    check_variance_validity(variances, targets)

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
    """
    Compute UCB regret
    --- requires uncertainty --
    Compute the Upper Confidence Bound (UCB) acquisition function
    on the test dataset.
    ucb_i = means_i + alpha * variances_i
    Compare the best possible num_acquisitions labels
    versus the labels of the sequences we would acquire according to the
    acquisition function

    We want to minimise this metric

    Args:
        means: Array of shape (b,). Mean predictions
        _: Array of shape (b,) or None. Predicted variances
        targets: Array of shape (b,). True labels
        alpha: float. The parameter of the UCB acquisition function
        num_acquisitions: int. The number of acquisitions

    Returns:
        dict: {f"regret_ucb_{alpha}": Regret float}
    """
    check_variance_validity(variances, targets)
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
    alpha: Optional[Union[float, list[float]]] | None = None,
    num_acquisitions: int = 100,
) -> dict[str, float]:
    """
    Compute UCB regret on a list of alphas
    --- requires uncertainty --

    Compute the `regret_ucb_alpha` metric on a list of alpha values

    Args:
        means: Array of shape (b,). Mean predictions
        _: Array of shape (b,) or None. Predicted variances
        targets: Array of shape (b,). True labels
        alpha: float | list[float]. The parameter(s) of the UCB acquisition function
        num_acquisitions: int. The number of acquisitions

    Returns:
        dict: Mapping from f"regret_ucb_{alpha}" to regret value for each alpha.
    """
    assert variances is not None, "UCB regret requires variances"
    assert np.all(
        variances >= 0
    ), "All uncertainty values must be non-negative (variances)."

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


# Backward compatibility
METRICS_REQ_VAR = metric_registry.get_metrics_requiring_variance()
METRICS_NOT_REQ_VAR = metric_registry.get_metrics_not_requiring_variance()