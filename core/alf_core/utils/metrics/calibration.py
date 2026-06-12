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

"""Uncertainty calibration metrics for probabilistic regression predictions.

Contains the registered calibration metrics (`expected_calibration_error`,
`rank_expected_calibration_error`).
"""

import numpy as np
from alf_core.utils.metrics.regression import monte_carlo_ranking, register_requires_variance
from jaxtyping import Float
from scipy.stats import norm


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
    stds = np.sqrt(variances)
    for i, cdf_cutoff in enumerate(grid):
        num_stds = norm.ppf(1 - ((1 - cdf_cutoff) / 2))
        # At cdf_cutoff=1, num_stds is inf and inf * 0 variance gives NaN;
        # the limiting interval for a zero-variance prediction is [mean, mean].
        with np.errstate(invalid="ignore"):
            half_width = np.nan_to_num(num_stds * stds, nan=0.0)
        perc[i] = ((targets >= means - half_width) & (targets <= means + half_width)).mean()
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
