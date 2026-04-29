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
from jaxtyping import Float, Int
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def check_inputs(means: np.ndarray, targets: np.ndarray) -> None:
    """Validate that means and targets arrays are compatible and valid.

    Args:
        means: Array of predicted means.
        targets: Array of target values.

    Raises:
        AssertionError: If shapes don't match, arrays are empty, or contain NaN values.
    """
    assert means.shape == targets.shape, (
        f"Means shape {means.shape} does not match targets shape {targets.shape} "
        f"(both must be (b,))"
    )
    assert len(means) != 0, "Means and targets must not be empty (expected shape (b,) with b > 0)"
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
    assert variances is not None, (
        "variances is None — this metric requires uncertainty estimates; "
        "ensure your model's predict() returns a Predictions object with variances set"
    )
    assert np.all(variances >= 0), (
        f"variances must be non-negative, but {np.sum(variances < 0)} values are negative "
        f"(min={variances.min():.4g})"
    )
    assert len(variances) == len(targets), (
        f"variances has {len(variances)} elements but targets has {len(targets)} — "
        f"both must have shape (b,)"
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
            Dictionary mapping metric names to their functions.
        """
        return {name: fn for name, fn in self.metrics.items() if self.variance_required[name]}

    def get_metrics_not_requiring_variance(self) -> dict[str, Callable]:
        """Get all registered metrics that don't require variance.

        Returns:
            Dictionary mapping metric names to their functions.
        """
        return {name: fn for name, fn in self.metrics.items() if not self.variance_required[name]}


# Create the global registry instances
metric_registry = MetricRegistry()
classification_metric_registry = MetricRegistry()


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
    metric_registry.register(metric_fn.__name__, wrapper, requires_variance=True)
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
    metric_registry.register(metric_fn.__name__, wrapper, requires_variance=False)
    return wrapper


def register_classification_metric(metric_fn: Callable) -> Callable:
    """Decorator to register a classification metric.

    Automatically registers the metric in the classification registry and applies
    basic input validation. The decorated function receives ``(probs, targets)``
    where ``probs`` has shape ``(n_samples, num_classes)`` and ``targets`` has
    shape ``(n_samples,)``.

    Args:
        metric_fn: The metric function to decorate.

    Returns:
        Wrapped metric function with validation and registration.
    """

    @wraps(metric_fn)
    def wrapper(
        probs: Float[np.ndarray, "n_samples num_classes"], targets: Float[np.ndarray, " n_samples"]
    ) -> dict[str, float]:
        assert probs.ndim == 2, (
            f"probs must be with shape (n_samples, num_classes), got shape {probs.shape}"
        )
        assert len(probs) != 0, "Empty input arrays"
        assert probs.shape[0] == targets.shape[0], (
            f"probs and targets batch size mismatch: {probs.shape[0]} vs {targets.shape[0]}"
        )
        return metric_fn(probs, targets)

    classification_metric_registry.register(metric_fn.__name__, wrapper, requires_variance=False)
    return wrapper


def monte_carlo_ranking(
    means: Float[np.ndarray, " b"],
    variances: Float[np.ndarray, " b"],
    num_samples: int = 10000,
) -> tuple[Float[np.ndarray, " b"], Float[np.ndarray, " b"]]:
    """Compute ranks, their means and variances using Monte Carlo simulation.

    For predicted means and variances, estimates normally distributed means
    using Monte Carlo simulation and computes ranking statistics.

    Args:
        means: Array of shape (b,). Mean predictions.
        variances: Array of shape (b,). Predicted variances.
        num_samples: Number of random samples to draw for the simulation.
            Defaults to 10000.

    Returns:
        A tuple containing:
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
    mean_rank, rank_variances = monte_carlo_ranking(means, variances)
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
        Dictionary with key "width_{alpha:.2f}" mapping to the normalized
        average width value.

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
        Dictionary with key "rank_width_{alpha:.2f}" mapping to the normalized
        average width value in rank space.

    Raises:
        AssertionError: If alpha is not in [0, 1].
    """
    assert (alpha >= 0) and (alpha <= 1), "alpha should be in [0,1]"

    mean_rank, rank_variances = monte_carlo_ranking(means, variances)
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

    mean_rank, rank_variances = monte_carlo_ranking(means, variances)
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
        return {f"regret_ucb_{alpha:.2f}": np.nan}

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


@register_requires_variance
def regret_ucb_alpha_sweep(
    means: Float[np.ndarray, " b"],
    variances: Float[np.ndarray, " b"],
    targets: Float[np.ndarray, " b"],
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


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


@register_classification_metric
def accuracy(
    probs: Float[np.ndarray, "n_samples num_classes"],
    targets: Int[np.ndarray, " n_samples"],
) -> dict[str, float]:
    """Compute classification accuracy.

    Args:
        probs: Array of shape (n_samples, num_classes). Predicted class probabilities.
        targets: Array of shape (n_samples,). Integer class labels.

    Returns:
        {"accuracy": accuracy float}
    """
    preds = np.argmax(probs, axis=1)
    return {"accuracy": float(accuracy_score(targets, preds))}


@register_classification_metric
def f1(
    probs: Float[np.ndarray, "n_samples num_classes"],
    targets: Int[np.ndarray, " n_samples"],
) -> dict[str, float]:
    """Compute macro-averaged F1 score.

    Args:
        probs: Array of shape (n_samples, num_classes). Predicted class probabilities.
        targets: Array of shape (n_samples,). Integer class labels.

    Returns:
        {"f1": macro F1 float}
    """
    preds = np.argmax(probs, axis=1)
    return {"f1": float(f1_score(targets, preds, average="macro", zero_division=0))}


@register_classification_metric
def precision(
    probs: Float[np.ndarray, "n_samples num_classes"],
    targets: Int[np.ndarray, " n_samples"],
) -> dict[str, float]:
    """Compute macro-averaged precision.

    Args:
        probs: Array of shape (n_samples, num_classes). Predicted class probabilities.
        targets: Array of shape (n_samples,). Integer class labels.

    Returns:
        {"precision": macro precision float}
    """
    preds = np.argmax(probs, axis=1)
    return {"precision": float(precision_score(targets, preds, average="macro", zero_division=0))}


@register_classification_metric
def recall(
    probs: Float[np.ndarray, "n_samples num_classes"],
    targets: Int[np.ndarray, " n_samples"],
) -> dict[str, float]:
    """Compute macro-averaged recall.

    Args:
        probs: Array of shape (n_samples, num_classes). Predicted class probabilities.
        targets: Array of shape (n_samples,). Integer class labels.

    Returns:
        {"recall": macro recall float}
    """
    preds = np.argmax(probs, axis=1)
    return {"recall": float(recall_score(targets, preds, average="macro", zero_division=0))}


@register_classification_metric
def auc_roc(
    probs: Float[np.ndarray, "n_samples num_classes"],
    targets: Int[np.ndarray, " n_samples"],
) -> dict[str, float]:
    """Compute Area Under the ROC Curve (AUC-ROC).

    For binary classification, uses the positive-class probabilities.
    For multiclass, uses one-vs-rest averaging.

    Args:
        probs: Array of shape (n_samples, num_classes). Predicted class probabilities.
        targets: Array of shape (n_samples,). Integer class labels.

    Returns:
        {"auc_roc": AUC-ROC float}
    """
    if probs.shape[1] == 2:
        return {"auc_roc": float(roc_auc_score(targets, probs[:, 1]))}
    elif len(np.unique(targets)) < 2:
        return {"auc_roc": float("nan")}
    return {"auc_roc": float(roc_auc_score(targets, probs, multi_class="ovr"))}
