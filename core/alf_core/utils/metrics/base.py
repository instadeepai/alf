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

"""Shared validators, decorators, and registry classes used across metric domains."""

import logging
from functools import wraps
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


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
        "ensure your model's predict() returns a Predictions object with variances set "
        "or that the problem type is correctly specified."
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


def require_min_samples(n: int) -> Callable:
    """Decorator factory that requires a minimum number of samples.

    If the decorated function is called with fewer than n samples (based on
    the length of the means array), returns an empty dictionary instead of
    calling the function.

    Args:
        n: Minimum number of samples required.

    Returns:
        A decorator that wraps metric functions.
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> dict[str, float]:
            if len(args[0]) < n:
                logger.warning(
                    f"Insufficient samples for metric {fn.__name__}: got "
                    f"{len(args[0])}, expected at least {n}"
                )
                return {}
            return fn(*args, **kwargs)

        return wrapper

    return decorator


class RegressionMetricRegistry:
    """Simple registry for metrics with variance requirements."""

    def __init__(self) -> None:
        """Initialise an empty metric registry."""
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

    def get_metrics(self, requires_variance: bool) -> dict[str, Callable]:
        """Get registered metrics filtered by variance requirement.

        Args:
            requires_variance: If True, return only metrics that need variance;
                if False, return only metrics that do not.

        Returns:
            Dictionary mapping metric names to their functions.
        """
        return {
            name: fn
            for name, fn in self.metrics.items()
            if self.variance_required[name] == requires_variance
        }


class ClassificationMetricRegistry:
    """Simple registry for classification metrics not requiring variance information."""

    def __init__(self) -> None:
        """Initialise an empty classification metric registry."""
        self.metrics: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None:
        """Register a classification metric function in the registry.

        Args:
            name (str): Name of the metric.
            fn (Callable): The metric function to register.
        """
        self.metrics[name] = fn

    def get_metrics(self) -> dict[str, Callable]:
        """Get all registered metrics that don't require variance.

        Returns:
            Dictionary mapping metric names to their functions.
        """
        return {name: fn for name, fn in self.metrics.items()}


# Create the global registry instances
regression_metric_registry = RegressionMetricRegistry()
classification_metric_registry = ClassificationMetricRegistry()
