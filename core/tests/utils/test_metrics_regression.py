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

import warnings

import numpy as np
import pytest
from alf_core.utils.metrics import (
    coverage,
    expected_calibration_error,
    hit_rate,
    mse,
    nll_gaussian,
    pairwise_xent,
    pearson,
    rank_coverage,
    rank_expected_calibration_error,
    rank_width,
    regression_metric_registry,
    regret_ucb_alpha,
    regret_ucb_alpha_sweep,
    residual_pearson,
    residual_spearman,
    spearman,
    top_k_max,
    top_k_mean,
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
        """Zero samples raises ValueError from check_inputs."""
        with pytest.raises(ValueError, match="empty"):
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
        """Zero samples raises ValueError from check_inputs."""
        with pytest.raises(ValueError, match="empty"):
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

    def test_zero_variance_exact_predictions_do_not_warn(self):
        """Zero-variance exact predictions compute ECE without NaN warnings.

        At the confidence-1.0 grid point `norm.ppf(1)` is inf, and inf * 0
        variance previously produced NaN interval bounds (RuntimeWarning)
        that mis-counted exact predictions as uncovered.
        """
        means = np.array([1.0, 2.0, 3.0])
        variances = np.zeros(3)
        targets = np.array([1.0, 2.0, 3.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = expected_calibration_error(means, variances, targets)
        # Exact predictions are covered at every confidence level, so the
        # observed coverage is 1.0 everywhere and ECE is the area between
        # the constant-1 curve and the diagonal (~0.5).
        assert result["ece"] == pytest.approx(0.5, abs=0.02)


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
        """Dataset of size 1 warns and returns empty dict."""
        with pytest.warns(UserWarning, match="too small"):
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

    def test_regret_ucb_alpha_zero_acquisitions_raises(self):
        """num_acquisitions=0 raises AssertionError."""
        with pytest.raises(AssertionError):
            regret_ucb_alpha(
                np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                np.ones(5),
                np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
                num_acquisitions=0,
            )


class TestRegressionRegistryIntegration:
    """Confirm regression metrics auto-register on module import."""

    def test_no_variance_metrics_registered(self):
        """mse, spearman, pearson, pairwise_xent are in the no-variance registry."""
        no_var = set(regression_metric_registry.get_metrics(requires_variance=False).keys())
        assert {"mse", "spearman", "pearson", "pairwise_xent"}.issubset(no_var)

    def test_variance_metrics_registered(self):
        """ECE, coverage, width, regret metrics are in the variance-required registry."""
        var_metrics = set(regression_metric_registry.get_metrics(requires_variance=True).keys())
        assert {
            "expected_calibration_error",
            "coverage",
            "width",
            "regret_ucb_alpha_sweep",
        }.issubset(var_metrics)


class TestWidth:
    """Tests for width metric."""

    def test_zero_range_targets_returns_empty(self):
        """All-equal targets means range is zero — width returns {}."""
        result = width(np.array([5.0, 5.0]), np.array([1.0, 1.0]), np.array([5.0, 5.0]))
        assert result == {}

    def test_positive_width_ratio(self):
        """Standard inputs return a non-negative width ratio."""
        result = width(
            np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5, 0.5]), np.array([1.0, 2.0, 3.0])
        )
        assert len(result) > 0
        for v in result.values():
            assert v >= 0.0

    def test_width_invalid_alpha_raises(self):
        """Alpha outside [0, 1] raises AssertionError."""
        with pytest.raises(AssertionError):
            width(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.5, 0.5, 0.5]),
                np.array([1.0, 2.0, 3.0]),
                alpha=1.5,
            )


class TestRegretUcbAlphaSweep:
    """Tests for regret_ucb_alpha_sweep."""

    def test_empty_alpha_list_raises_value_error(self):
        """Empty alpha list raises ValueError."""
        means = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        variances = np.ones(5)
        targets = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        with pytest.raises(ValueError):
            regret_ucb_alpha_sweep(means, variances, targets, alpha=[])

    def test_bad_alpha_type_raises_type_error(self):
        """Non-float, non-list alpha raises TypeError."""
        means = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        variances = np.ones(5)
        targets = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        with pytest.raises(TypeError):
            regret_ucb_alpha_sweep(means, variances, targets, alpha="bad")

    def test_float_alpha_works(self):
        """Single float alpha runs successfully."""
        means = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        variances = np.ones(5)
        targets = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        # The default num_acquisitions (100) exceeds the 5 items, triggering
        # the documented fallback warning.
        with pytest.warns(UserWarning, match="num_acquisitions"):
            result = regret_ucb_alpha_sweep(means, variances, targets, alpha=0.5)
        assert len(result) > 0

    def test_int_alpha_works(self):
        """A scalar int alpha is accepted, consistent with ints inside an alpha list."""
        means = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        variances = np.ones(5)
        targets = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = regret_ucb_alpha_sweep(means, variances, targets, alpha=1, num_acquisitions=2)
        assert set(result) == {"regret_ucb_sweep_1.00"}

    def test_default_alpha_sweep_returns_four_namespaced_keys(self):
        """The default alpha sweep emits one namespaced key per alpha in [0.1, 0.3, 0.5, 1.0]."""
        means = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        variances = np.ones(5)
        targets = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = regret_ucb_alpha_sweep(means, variances, targets, num_acquisitions=2)
        assert set(result) == {
            "regret_ucb_sweep_0.10",
            "regret_ucb_sweep_0.30",
            "regret_ucb_sweep_0.50",
            "regret_ucb_sweep_1.00",
        }

    def test_sweep_keys_do_not_collide_with_standalone_metric(self):
        """Regression test: the sweep previously emitted regret_ucb_{alpha}, colliding
        with the standalone regret_ucb_alpha metric and silently overwriting it when
        both were merged in the metric registry.
        """
        means = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        variances = np.ones(5)
        targets = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        single = regret_ucb_alpha(means, variances, targets, alpha=0.1, num_acquisitions=2)
        sweep = regret_ucb_alpha_sweep(means, variances, targets, num_acquisitions=2)
        assert "regret_ucb_0.10" in single
        assert all(key.startswith("regret_ucb_sweep_") for key in sweep)
        assert set(single).isdisjoint(sweep)


class TestRegretUcbAlphaEdgeCases:
    """Additional edge case tests for regret_ucb_alpha."""

    def test_num_acquisitions_exceeds_dataset_warns_and_returns_result(self):
        """num_acquisitions > len(means) warns and falls back."""
        means = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        variances = np.zeros(5)
        targets = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        with pytest.warns(UserWarning):
            result = regret_ucb_alpha(means, variances, targets, alpha=0.0, num_acquisitions=100)
        assert len(result) > 0


class TestCoverage:
    """Tests for coverage metric."""

    def test_returns_coverage_key(self):
        """Coverage returns a dict with coverage_{alpha:.2f} key."""
        result = coverage(
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([0.0, 1.0, 2.0]),
        )
        assert "coverage_0.95" in result
        assert 0.0 <= result["coverage_0.95"] <= 1.0

    def test_one_sample_returns_coverage(self):
        """Single sample still computes coverage (no min-samples guard)."""
        result = coverage(np.array([1.0]), np.array([1.0]), np.array([1.0]))
        assert "coverage_0.95" in result

    def test_return_type_is_dict_of_float(self):
        """Return value is dict with float values."""
        result = coverage(
            np.array([1.0, 2.0, 3.0]),
            np.array([0.5, 0.5, 0.5]),
            np.array([1.0, 2.0, 3.0]),
        )
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(float(v), float)


class TestRankCoverage:
    """Tests for rank_coverage metric."""

    def test_returns_rank_coverage_key(self):
        """rank_coverage returns a dict with rank_coverage_{alpha:.2f} key."""
        rng = np.random.default_rng(42)
        means = rng.standard_normal(5)
        variances = np.abs(rng.standard_normal(5)) + 0.1
        targets = rng.standard_normal(5)
        result = rank_coverage(means, variances, targets)
        assert "rank_coverage_0.95" in result
        assert 0.0 <= result["rank_coverage_0.95"] <= 1.0

    def test_return_type_is_dict_of_float(self):
        """Return value is dict with float values."""
        rng = np.random.default_rng(0)
        means = rng.standard_normal(4)
        variances = np.abs(rng.standard_normal(4)) + 0.1
        targets = rng.standard_normal(4)
        result = rank_coverage(means, variances, targets)
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(float(v), float)


class TestRankExpectedCalibrationError:
    """Tests for rank_expected_calibration_error metric."""

    def test_returns_rank_ece_key(self):
        """rank_expected_calibration_error returns a dict with rank_ece key."""
        rng = np.random.default_rng(42)
        means = rng.standard_normal(5)
        variances = np.abs(rng.standard_normal(5)) + 0.1
        targets = rng.standard_normal(5)
        result = rank_expected_calibration_error(means, variances, targets)
        assert "rank_ece" in result
        assert result["rank_ece"] >= 0.0

    def test_return_type_is_dict_of_float(self):
        """Return value is dict with float values."""
        rng = np.random.default_rng(1)
        means = rng.standard_normal(4)
        variances = np.abs(rng.standard_normal(4)) + 0.1
        targets = rng.standard_normal(4)
        result = rank_expected_calibration_error(means, variances, targets)
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(float(v), float)


class TestRankWidth:
    """Tests for rank_width metric."""

    def test_returns_rank_width_key(self):
        """rank_width returns a dict with rank_width_{alpha:.2f} key."""
        rng = np.random.default_rng(42)
        means = rng.standard_normal(5)
        variances = np.abs(rng.standard_normal(5)) + 0.1
        targets = rng.standard_normal(5)
        result = rank_width(means, variances, targets)
        assert "rank_width_0.95" in result
        assert result["rank_width_0.95"] >= 0.0

    def test_return_type_is_dict_of_float(self):
        """Return value is dict with float values."""
        rng = np.random.default_rng(2)
        means = rng.standard_normal(4)
        variances = np.abs(rng.standard_normal(4)) + 0.1
        targets = rng.standard_normal(4)
        result = rank_width(means, variances, targets)
        assert isinstance(result, dict)
        for v in result.values():
            assert isinstance(float(v), float)


class TestResidualSpearman:
    """Tests for residual_spearman metric."""

    def test_returns_residual_spearman_key(self):
        """residual_spearman returns a dict with residual_spearman key."""
        rng = np.random.default_rng(42)
        means = rng.standard_normal(5)
        variances = np.abs(rng.standard_normal(5)) + 0.1
        targets = rng.standard_normal(5)
        result = residual_spearman(means, variances, targets)
        assert "residual_spearman" in result

    def test_value_in_range(self):
        """residual_spearman value is in [-1, 1] or NaN for degenerate inputs."""
        rng = np.random.default_rng(10)
        means = rng.standard_normal(10)
        variances = np.abs(rng.standard_normal(10)) + 0.1
        targets = rng.standard_normal(10)
        result = residual_spearman(means, variances, targets)
        val = result["residual_spearman"]
        assert np.isnan(val) or (-1.0 <= val <= 1.0)

    def test_return_type_is_dict(self):
        """Return value is dict."""
        rng = np.random.default_rng(3)
        means = rng.standard_normal(3)
        variances = np.abs(rng.standard_normal(3)) + 0.1
        targets = rng.standard_normal(3)
        result = residual_spearman(means, variances, targets)
        assert isinstance(result, dict)


class TestResidualPearson:
    """Tests for residual_pearson metric."""

    def test_returns_residual_pearson_key(self):
        """residual_pearson returns a dict with residual_pearson key."""
        rng = np.random.default_rng(42)
        means = rng.standard_normal(5)
        variances = np.abs(rng.standard_normal(5)) + 0.1
        targets = rng.standard_normal(5)
        result = residual_pearson(means, variances, targets)
        assert "residual_pearson" in result

    def test_value_in_range(self):
        """residual_pearson value is in [-1, 1] or NaN for degenerate inputs."""
        rng = np.random.default_rng(20)
        means = rng.standard_normal(10)
        variances = np.abs(rng.standard_normal(10)) + 0.1
        targets = rng.standard_normal(10)
        result = residual_pearson(means, variances, targets)
        val = result["residual_pearson"]
        assert np.isnan(val) or (-1.0 <= val <= 1.0)

    def test_return_type_is_dict(self):
        """Return value is dict."""
        rng = np.random.default_rng(4)
        means = rng.standard_normal(3)
        variances = np.abs(rng.standard_normal(3)) + 0.1
        targets = rng.standard_normal(3)
        result = residual_pearson(means, variances, targets)
        assert isinstance(result, dict)


class TestPairwiseXent:
    """Tests for pairwise_xent metric."""

    def test_returns_pairwise_xent_key(self):
        """pairwise_xent returns a dict with pairwise_xent key."""
        result = pairwise_xent(
            np.array([1.0, 2.0, 3.0]),
            None,
            np.array([3.0, 2.0, 1.0]),
        )
        assert "pairwise_xent" in result

    def test_value_is_non_negative(self):
        """pairwise_xent value is non-negative."""
        rng = np.random.default_rng(42)
        means = rng.standard_normal(5)
        targets = rng.standard_normal(5)
        result = pairwise_xent(means, None, targets)
        assert result["pairwise_xent"] >= 0.0

    def test_return_type_is_dict_of_float(self):
        """Return value is dict with float value."""
        result = pairwise_xent(
            np.array([0.0, 1.0, 2.0]),
            None,
            np.array([0.0, 1.0, 2.0]),
        )
        assert isinstance(result, dict)
        assert isinstance(float(result["pairwise_xent"]), float)


class TestTopKMean:
    """Tests for top_k_mean metric."""

    def test_returns_correct_key(self):
        """top_k_mean returns a dict with the correct dynamic key."""
        result = top_k_mean(np.zeros(5), None, np.array([1.0, 2.0, 3.0, 4.0, 5.0]), k=3)
        assert "top_3_mean" in result

    def test_known_value(self):
        """top_k_mean returns the mean of the top-k targets."""
        result = top_k_mean(np.zeros(5), None, np.array([1.0, 2.0, 3.0, 4.0, 5.0]), k=2)
        assert result["top_2_mean"] == pytest.approx(4.5)

    def test_k_exceeds_length_uses_all(self):
        """When k > len(targets), all targets are used."""
        result = top_k_mean(np.zeros(3), None, np.array([1.0, 2.0, 3.0]), k=100)
        assert "top_3_mean" in result
        assert result["top_3_mean"] == pytest.approx(2.0)

    def test_registered_no_variance(self):
        """top_k_mean is registered in the no-variance registry."""
        assert "top_k_mean" in regression_metric_registry.get_metrics(requires_variance=False)


class TestTopKMax:
    """Tests for top_k_max metric."""

    def test_returns_correct_key(self):
        """top_k_max returns a dict with the correct dynamic key."""
        result = top_k_max(np.zeros(5), None, np.array([1.0, 2.0, 3.0, 4.0, 5.0]), k=3)
        assert "top_3_max" in result

    def test_known_value(self):
        """top_k_max returns the maximum of the top-k targets."""
        result = top_k_max(np.zeros(5), None, np.array([1.0, 2.0, 3.0, 4.0, 5.0]), k=3)
        assert result["top_3_max"] == pytest.approx(5.0)

    def test_k_exceeds_length_uses_all(self):
        """When k > len(targets), all targets are used."""
        result = top_k_max(np.zeros(3), None, np.array([1.0, 2.0, 3.0]), k=100)
        assert "top_3_max" in result
        assert result["top_3_max"] == pytest.approx(3.0)

    def test_registered_no_variance(self):
        """top_k_max is registered in the no-variance registry."""
        assert "top_k_max" in regression_metric_registry.get_metrics(requires_variance=False)


class TestHitRate:
    """Tests for hit_rate metric."""

    def test_all_hits(self):
        """All targets above threshold gives hit rate 1.0."""
        result = hit_rate(np.zeros(3), None, np.array([0.9, 0.8, 0.7]), threshold=0.5)
        assert result["hit_rate_0.500"] == pytest.approx(1.0)

    def test_no_hits(self):
        """No targets above threshold gives hit rate 0.0."""
        result = hit_rate(np.zeros(3), None, np.array([0.1, 0.2, 0.3]), threshold=0.5)
        assert result["hit_rate_0.500"] == pytest.approx(0.0)

    def test_partial_hits(self):
        """Half the targets above threshold gives hit rate 0.5."""
        result = hit_rate(np.zeros(4), None, np.array([0.9, 0.1, 0.8, 0.2]), threshold=0.5)
        assert result["hit_rate_0.500"] == pytest.approx(0.5)

    def test_threshold_inclusive(self):
        """Targets exactly at threshold are counted as hits."""
        result = hit_rate(np.zeros(2), None, np.array([0.5, 0.4]), threshold=0.5)
        assert result["hit_rate_0.500"] == pytest.approx(0.5)

    def test_returns_dynamic_key(self):
        """Key reflects the threshold value."""
        result = hit_rate(np.zeros(3), None, np.array([1.0, 2.0, 3.0]), threshold=1.5)
        assert "hit_rate_1.500" in result

    def test_registered_no_variance(self):
        """hit_rate is registered in the no-variance registry."""
        assert "hit_rate" in regression_metric_registry.get_metrics(requires_variance=False)


class TestNllGaussian:
    """Tests for nll_gaussian metric."""

    def test_returns_nll_key(self):
        """nll_gaussian returns a dict with key 'nll'."""
        result = nll_gaussian(
            np.array([0.0, 1.0, 2.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([0.0, 1.0, 2.0]),
        )
        assert "nll" in result

    def test_perfect_predictions_low_nll(self):
        """Predictions matching targets with low variance produce low NLL."""
        means = np.array([1.0, 2.0, 3.0])
        variances = np.full(3, 0.01)
        targets = np.array([1.0, 2.0, 3.0])
        result = nll_gaussian(means, variances, targets)
        assert result["nll"] < 0.0

    def test_nll_increases_with_error(self):
        """Higher prediction error leads to higher NLL."""
        means_good = np.array([1.0, 2.0, 3.0])
        means_bad = np.array([3.0, 0.0, 1.0])
        variances = np.ones(3)
        targets = np.array([1.0, 2.0, 3.0])
        nll_good = nll_gaussian(means_good, variances, targets)["nll"]
        nll_bad = nll_gaussian(means_bad, variances, targets)["nll"]
        assert nll_good < nll_bad

    def test_zero_variance_does_not_raise(self):
        """Zero variance is handled without raising (clipped internally)."""
        result = nll_gaussian(
            np.array([1.0, 2.0]),
            np.array([0.0, 0.0]),
            np.array([1.0, 2.0]),
        )
        assert "nll" in result
        assert np.isfinite(result["nll"])

    def test_registered_requires_variance(self):
        """nll_gaussian is registered in the requires-variance registry."""
        assert "nll_gaussian" in regression_metric_registry.get_metrics(requires_variance=True)
