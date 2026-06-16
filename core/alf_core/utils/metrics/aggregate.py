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

"""Standalone metrics not registered in regression_metric_registry.

Contains multi-round summary metrics (auc_top_k) that operate on per-round
aggregates rather than per-candidate prediction arrays, and so are not
registered in the registry.
"""

import warnings
from typing import TYPE_CHECKING

import numpy as np
from alf_core.utils.enums import ProblemType
from alf_core.utils.metrics.regression import top_k_mean
from jaxtyping import Float

if TYPE_CHECKING:
    from alf_core.dataclasses import State


def auc_top_k(
    round_values: Float[np.ndarray, " n_rounds"],
    best_value: float,
) -> dict[str, float]:
    """Compute the normalised area under the top-k mean curve.

    Integrates the per-round curve using the trapezoidal rule, then divides by
    `best_value` so the AUC lands in [0, 1].  The value 1.0 is only reachable
    when the curve itself is bounded by `best_value` and saturates at it from
    the first round — true for a max-based curve (e.g. classification accuracy
    with `best_value=1.0`), but not for a top-k *mean* curve normalised by the
    global *max*, where the mean of the top k can equal the max only if every
    top-k label equals the global optimum.  For that reason the AUC is best
    read as a relative sample-efficiency ranking within a fixed dataset and
    round budget, not as an absolute "fraction of optimal".  Lower values
    indicate that high-performing candidates were found later in the
    experiment.

    Args:
        round_values: Array of shape (n_rounds,).  Per-round top-k mean, where
            entry `i` is the top-k mean of all candidates acquired by round
            `i` (inclusive).
        best_value: The global best oracle label in the dataset.  Must be
            strictly positive.  Used to normalise the AUC to [0, 1].  If any
            entry in `round_values` exceeds `best_value`, a warning is issued
            and the result is clamped to 1.0.

    Returns:
        Dictionary with key `auc_top_k` mapping to the normalised AUC in
        [0, 1].

    Raises:
        ValueError: If `round_values` has fewer than 2 entries or
            `best_value` is not strictly positive (non-zero).
    """
    if len(round_values) < 2:
        raise ValueError(f"auc_top_k requires at least 2 rounds, got {len(round_values)}")
    # Written as `not (best_value > 0.0)` rather than `best_value <= 0.0` so
    # that NaN (for which all comparisons are False) is also rejected.
    if not (best_value > 0.0):
        raise ValueError(
            f"best_value must be strictly positive (non-zero) to normalise the AUC, "
            f"got {best_value}"
        )
    n = len(round_values)
    normalised = round_values / best_value
    if np.any(normalised < 0.0):
        warnings.warn(
            "Some round_values are negative; they lower the AUC, which is clamped to [0.0, 1.0].",
            stacklevel=2,
        )
    if np.any(normalised > 1.0):
        warnings.warn(
            "Some round_values exceed best_value; the normalised AUC is clamped to "
            "[0.0, 1.0]. Verify that best_value is the true global optimum.",
            stacklevel=2,
        )
    dx = 1.0 / (n - 1)
    auc = float(np.clip(np.sum((normalised[:-1] + normalised[1:]) / 2) * dx, 0.0, 1.0))
    return {"auc_top_k": auc}


_AGGREGATE_METRICS = (auc_top_k,)


def compute_aggregate_metrics(
    round_values: Float[np.ndarray, " n_rounds"],
    best_value: float,
) -> dict[str, float]:
    """Run every aggregate metric over a per-round curve and merge the results.

    Single entry point for end-of-experiment summary metrics: new aggregate
    metrics only need to be added to `_AGGREGATE_METRICS` here, without
    touching the calling task.  Metrics whose requirements are not met (e.g.
    fewer than two rounds) raise ValueError internally and are skipped, so
    the caller does not need to guard against partial failures.

    Args:
        round_values: Array of shape (n_rounds,) with one aggregated value
            per round.
        best_value: The global best oracle label in the dataset, used for
            normalisation.

    Returns:
        Merged dictionary of all aggregate metrics that could be computed.
        Empty when none could be computed.
    """
    metrics: dict[str, float] = {}
    for metric_fn in _AGGREGATE_METRICS:
        try:
            metrics.update(metric_fn(round_values, best_value))
        except ValueError:
            continue
    return metrics


def _sample_efficiency_curve(state: "State") -> list[float]:
    """Build the per-round sample-efficiency curve fed to the aggregate metrics.

    For regression the curve is the top-k mean of all candidates acquired up to
    each round (it should rise as good candidates accumulate); for classification
    it is the per-round test-set accuracy.

    Note: `top_k_mean` uses an effective k of `min(k, n_acquired)`, so while the
    cumulative acquired set is smaller than k the early rounds are averaged over
    fewer candidates and are not directly comparable to later rounds. With small
    acquisition batch sizes this can make the curve non-monotonic and bias the
    downstream AUC; treat the AUC as a relative ranking rather than an absolute
    score in that regime.

    Args:
        state: Final task state after all acquisition rounds.

    Returns:
        One value per acquisition round that produced a valid value.
    """
    if state.problem_type == ProblemType.REGRESSION:
        curve = []
        for round_i in range(1, len(state.history) + 1):
            labels = np.concatenate([c.labels for c in state.history[:round_i]])
            # top_k_mean returns a single dynamically-keyed entry (e.g.
            # "top_10_mean"); take its value for the curve.
            curve.append(float(next(iter(top_k_mean(labels, None, labels).values()))))
        return curve
    return [
        float(metrics.metrics["surrogate/test_accuracy"])
        for metrics in state.metrics_history
        if metrics.round > 0 and "surrogate/test_accuracy" in metrics.metrics
    ]


def compute_experiment_summary(state: "State") -> dict[str, float]:
    """Run all aggregate metrics over a task's per-round sample-efficiency curve.

    State-based entry point for `alf_core.tasks`: builds the per-round curve and
    normaliser from the final task state and delegates to
    `compute_aggregate_metrics`, so the calling task stays free of metric logic
    and new metrics only need adding to `_AGGREGATE_METRICS`.

    Args:
        state: Final task state after all acquisition rounds.

    Returns:
        Merged dictionary of all aggregate metrics that could be computed.
        Empty when none could be computed.
    """
    is_regression = state.problem_type == ProblemType.REGRESSION
    best_value = float(state.dataset.raw_dataset.labels.max()) if is_regression else 1.0
    return compute_aggregate_metrics(np.array(_sample_efficiency_curve(state)), best_value)
