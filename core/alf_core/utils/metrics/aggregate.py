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

import numpy as np
from jaxtyping import Float


def auc_top_k(
    round_values: Float[np.ndarray, " n_rounds"],
    best_value: float,
) -> dict[str, float]:
    """Compute the normalised area under the top-k mean curve.

    Integrates the per-round top-k mean values using the trapezoidal rule,
    then normalises the result so that a perfect experiment (one that always
    achieves `best_value`) scores 1.0.  Lower values indicate that high-
    performing candidates were found later in the experiment.  Use as the
    primary leaderboard ranking metric for sample efficiency.

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
