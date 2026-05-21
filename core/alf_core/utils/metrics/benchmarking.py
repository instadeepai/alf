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

"""Benchmarking metrics for campaign-level optimisation performance.

This module contains metrics that assess end-to-end optimisation quality
(i.e. how well the acquisition strategy finds high-fitness candidates),
as opposed to surrogate-model quality metrics in ``regression.py``.

Planned metrics (to be implemented in future PRs):

    top_k_mean(means, targets, k) -> dict[str, float]
        Mean fitness of the top-k model-ranked candidates vs oracle ranking.
        Primary optimisation metric / leaderboard score.

    top_k_max(means, targets, k) -> dict[str, float]
        Maximum fitness of the top-k model-ranked candidates.

    auc_top_k(means, targets) -> dict[str, float]
        Area under the top-k-mean curve swept over k in [1, b].
        Sample efficiency proxy for leaderboard ranking.

    hit_rate_at_threshold(means, targets, k, threshold) -> dict[str, float]
        Fraction of top-k model-ranked candidates whose true fitness
        exceeds a user-specified threshold. Discovery framing metric.

    intra_batch_diversity(sequences) -> dict[str, float]
        Mean pairwise Tanimoto or edit-distance diversity within an
        acquired batch. Has a different input signature (no targets).
        Consider a separate decorator for this class of metric.

Uncertainty calibration (NLL, reliability diagrams) belong in regression.py
alongside the existing expected_calibration_error.
"""

from functools import wraps
from typing import Callable


class BenchmarkingMetricRegistry:
    """Registry for campaign-level benchmarking metrics."""

    def __init__(self) -> None:
        """Initialise an empty benchmarking registry."""
        self.metrics: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None:
        """Register a benchmarking metric function.

        Args:
            name: Name identifier for the metric.
            fn: Callable that computes the metric.
        """
        self.metrics[name] = fn

    def get_metrics(self) -> dict[str, Callable]:
        """Return all registered benchmarking metrics.

        Returns:
            Dictionary mapping metric names to their functions.
        """
        return dict(self.metrics)


benchmarking_metric_registry = BenchmarkingMetricRegistry()


def register_benchmarking_metric(metric_fn: Callable) -> Callable:
    """Decorator to register a function in the global benchmarking metric registry.

    The decorated function must accept at least ``(means, targets)`` as its
    first two positional arguments and return a ``dict[str, float]``.

    Args:
        metric_fn: The metric function to decorate.

    Returns:
        The wrapped function, registered in ``benchmarking_metric_registry``.
    """

    @wraps(metric_fn)
    def wrapper(*args, **kwargs):
        return metric_fn(*args, **kwargs)

    benchmarking_metric_registry.register(metric_fn.__name__, wrapper)
    return wrapper
