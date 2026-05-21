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

"""Tests for regression metrics."""

import numpy as np
import pytest
from alf_core.utils.metrics.regression import (
    coverage,
    expected_calibration_error,
    monte_carlo_ranking,
    mse,
    pairwise_xent,
    pearson,
    rank_coverage,
    rank_expected_calibration_error,
    rank_width,
    regret_ucb_alpha,
    regret_ucb_alpha_sweep,
    residual_pearson,
    residual_spearman,
    spearman,
    width,
)


class TestMse:
    """Tests for mse metric."""

    def test_zero_error(self):
        """Perfect predictions give zero MSE."""
        result = mse(np.array([1.0, 2.0, 3.0]), None, np.array([1.0, 2.0, 3.0]))
        assert result == {"mse": pytest.approx(0.0)}

    def test_known_error(self):
        """MSE matches manual computation."""
        result = mse(np.array([0.0, 0.0]), None, np.array([1.0, 3.0]))
        assert result == {"mse": pytest.approx(5.0)}


class TestPearsonGuard:
    """Tests for the require_min_samples guard applied to pearson."""

    def test_one_sample_returns_empty_dict(self):
        """Single sample returns {} instead of NaN."""
        result = pearson(np.array([1.0]), None, np.array([1.0]))
        assert result == {}

    def test_two_samples_returns_pearson_key(self):
        """Two samples computes and returns the pearson key."""
        result = pearson(np.array([1.0, 2.0]), None, np.array([1.0, 2.0]))
        assert "pearson" in result
        assert isinstance(result["pearson"], float)

    def test_three_samples_returns_finite_value(self):
        """Three samples returns a finite pearson value."""
        result = pearson(np.array([1.0, 2.0, 3.0]), None, np.array([1.0, 2.0, 3.0]))
        assert "pearson" in result
        assert np.isfinite(result["pearson"])

    def test_zero_samples_raises(self):
        """Zero samples raises AssertionError from check_inputs."""
        with pytest.raises(AssertionError):
            pearson(np.array([]), None, np.array([]))


class TestSpearmanGuard:
    """Tests for the require_min_samples guard applied to spearman."""

    def test_one_sample_returns_empty_dict(self):
        """Single sample returns {} instead of NaN."""
        result = spearman(np.array([1.0]), None, np.array([1.0]))
        assert result == {}

    def test_two_samples_returns_spearman_key(self):
        """Two samples computes and returns the spearman key."""
        result = spearman(np.array([1.0, 2.0]), None, np.array([1.0, 2.0]))
        assert "spearman" in result
        assert isinstance(result["spearman"], float)

    def test_three_samples_returns_finite_value(self):
        """Three samples returns a finite spearman value."""
        result = spearman(np.array([1.0, 2.0, 3.0]), None, np.array([1.0, 2.0, 3.0]))
        assert "spearman" in result
        assert np.isfinite(result["spearman"])

    def test_zero_samples_raises(self):
        """Zero samples raises AssertionError from check_inputs."""
        with pytest.raises(AssertionError):
            spearman(np.array([]), None, np.array([]))


class TestExpectedCalibrationError:
    """Tests for expected_calibration_error."""

    def test_perfect_calibration_is_near_zero(self):
        """Perfectly calibrated predictions have near-zero ECE."""
        rng = np.random.default_rng(42)
        means = rng.standard_normal(200)
        variances = np.ones(200)
        targets = means + rng.standard_normal(200)
        result = expected_calibration_error(means, variances, targets)
        assert "ece" in result
        assert result["ece"] < 0.15

    def test_returns_float_in_range(self):
        """ECE is a non-negative float."""
        result = expected_calibration_error(
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([0.1, 1.1, 1.9]),
        )
        assert "ece" in result
        assert result["ece"] >= 0.0


class TestRegretUcbAlpha:
    """Tests for regret_ucb_alpha."""

    def test_perfect_model_has_zero_regret(self):
        """A model with means == targets selects optimally."""
        targets = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        result = regret_ucb_alpha(
            targets.copy(), np.zeros(5), targets.copy(), alpha=0.0, num_acquisitions=2
        )
        assert result.get("regret_ucb_0.00", -1) == pytest.approx(0.0)

    def test_small_dataset_returns_empty(self):
        """Dataset of size 1 returns empty dict."""
        result = regret_ucb_alpha(
            np.array([1.0]), np.array([0.1]), np.array([1.0]), num_acquisitions=1
        )
        assert result == {}

    def test_returns_non_negative_regret(self):
        """Regret is always >= 0."""
        means = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        variances = np.ones(5)
        targets = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = regret_ucb_alpha(means, variances, targets, alpha=0.1, num_acquisitions=2)
        key = "regret_ucb_0.10"
        assert key in result
        assert result[key] >= 0.0


class TestRegressionRegistryIntegration:
    """Confirm regression metrics auto-register on module import."""

    def test_no_variance_metrics_registered(self):
        """mse, spearman, pearson, pairwise_xent are in the no-variance registry."""
        from alf_core.utils.metrics._base import regression_metric_registry

        no_var = set(regression_metric_registry.get_metrics(requires_variance=False).keys())
        assert {"mse", "spearman", "pearson", "pairwise_xent"}.issubset(no_var)

    def test_variance_metrics_registered(self):
        """ECE, coverage, width, regret metrics are in the variance-required registry."""
        from alf_core.utils.metrics._base import regression_metric_registry

        var_metrics = set(regression_metric_registry.get_metrics(requires_variance=True).keys())
        assert {
            "expected_calibration_error",
            "coverage",
            "width",
            "regret_ucb_alpha_sweep",
        }.issubset(var_metrics)
