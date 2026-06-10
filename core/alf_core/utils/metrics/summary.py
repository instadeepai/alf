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

"""Standalone summary metrics that operate across multiple rounds.

Unlike the registered per-round metrics in ``regression.py``, these functions
aggregate information over an entire active learning experiment and are not
registered in ``regression_metric_registry``.
"""

import numpy as np
from alf_core.utils.metrics.base import check_inputs, check_variance_validity
from jaxtyping import Float
from scipy.stats import norm


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
            strictly positive.  Used to normalise the AUC to [0, 1].

    Returns:
        Dictionary with key `auc_top_k` mapping to the normalised AUC.

    Raises:
        ValueError: If `round_values` has fewer than 2 entries or
            `best_value` is not strictly positive (non-zero).
    """
    if len(round_values) < 2:
        raise ValueError(f"auc_top_k requires at least 2 rounds, got {len(round_values)}")
    if best_value <= 0.0:
        raise ValueError(
            f"best_value must be strictly positive (non-zero) to normalise the AUC, "
            f"got {best_value}"
        )
    n = len(round_values)
    normalised = round_values / best_value
    dx = 1.0 / (n - 1)
    auc = float(np.sum((normalised[:-1] + normalised[1:]) / 2) * dx)
    return {"auc_top_k": auc}


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
    `expected_calibration_error`, enabling callers to render the diagram
    without re-computing the coverage sweep.

    Args:
        means: Array of shape (b,). Mean predictions.
        variances: Array of shape (b,). Predicted variances.
        targets: Array of shape (b,). True labels.
        n_grid_points: Number of confidence levels to evaluate.
            Defaults to 100.

    Returns:
        A tuple `(expected_coverage, observed_coverage)` where each array
        has shape (n_grid_points,) and values in [0, 1].
    """
    check_inputs(means, targets)
    check_variance_validity(variances, targets)
    grid = np.linspace(0, 1, n_grid_points)
    observed = np.zeros(n_grid_points)
    std_devs = np.sqrt(variances)
    for i, alpha in enumerate(grid):
        n_stds = norm.ppf(1 - (1 - alpha) / 2)
        half_width = n_stds * std_devs
        observed[i] = ((targets >= means - half_width) & (targets <= means + half_width)).mean()
    return grid, observed
